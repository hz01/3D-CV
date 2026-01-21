from pathlib import Path
from urllib.request import urlretrieve
import zipfile
import pycolmap
import open3d as o3d
import numpy as np

try:
    import torch
except ImportError:
    torch = None


class ColmapPipeline:
    """COLMAP 3D reconstruction pipeline class."""
    
    def __init__(
        self,
        project_root: str = "colmap_cpu_project",
        dataset_url: str = "https://github.com/colmap/colmap/releases/download/3.11.1/south-building.zip",
        num_threads: int = 6,
    ):
        """
        Initialize the COLMAP pipeline.
        
        Args:
            project_root: Root directory for the project
            dataset_url: URL to download the dataset
            num_threads: Number of threads for processing
        """
        self.project_root = Path(project_root)
        self.dataset_url = dataset_url
        self.num_threads = num_threads
        
        # Auto-detect CUDA availability
        self.device = self._detect_device()
        self.use_gpu = (self.device == pycolmap.Device.cuda)
        
        # Paths
        self.out_dir = self.project_root / "outputs"
        self.database_path = self.out_dir / "database.db"
        self.sparse_dir = self.out_dir / "sparse"
        self.dense_dir = self.out_dir / "dense"
        self.sparse_ply_path = self.out_dir / "sparse_points.ply"
        
        # Will be set after dataset download
        self.images_dir = None
        self.sparse_model_dir = None
        
        # Create output directory
        self.out_dir.mkdir(parents=True, exist_ok=True)
    
    def _detect_device(self):
        """
        Detect if CUDA is available, otherwise use CPU.
        
        Returns:
            pycolmap.Device: CUDA device if available, CPU otherwise
        """
        try:
            # Check if CUDA device exists in pycolmap
            if hasattr(pycolmap.Device, 'cuda'):
                # Try to verify CUDA is actually available
                if torch is not None and torch.cuda.is_available():
                    print("CUDA device detected. Using GPU acceleration.")
                    return pycolmap.Device.cuda
                elif torch is None:
                    # torch not available, but pycolmap might still have CUDA support
                    # If pycolmap was built with CUDA, it should work
                    print("CUDA device detected (pycolmap with CUDA support). Using GPU acceleration.")
                    return pycolmap.Device.cuda
                else:
                    print("CUDA not available (no GPU detected). Using CPU.")
                    return pycolmap.Device.cpu
            else:
                print("CUDA not available (pycolmap built without CUDA). Using CPU.")
                return pycolmap.Device.cpu
        except Exception as e:
            # If anything fails, default to CPU
            print(f"CUDA detection failed: {e}. Using CPU.")
            return pycolmap.Device.cpu
    
    def download_and_extract_dataset(self) -> Path:
        """
        Download and extract the dataset.
        
        Returns:
            Path to the images directory
        """
        zip_path = self.project_root / "south-building.zip"
        extract_dir = self.project_root / "south-building"
        
        # 1) Download
        if not zip_path.exists():
            print(f"Downloading:\n{self.dataset_url}")
            urlretrieve(self.dataset_url, zip_path)
            print(f"Saved zip to: {zip_path}")
        else:
            print(f"Zip already exists: {zip_path}")
        
        # 2) Extract
        if not extract_dir.exists():
            print(f"Extracting to: {extract_dir}")
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
            print("Extracted.")
        else:
            print(f"Already extracted: {extract_dir}")
        
        # 3) Find images folder
        candidates = []
        for p in extract_dir.rglob("*"):
            if p.is_dir() and p.name.lower() in ("images", "imgs", "image", "rgb"):
                candidates.append(p)
        
        if not candidates:
            print("Warning: Could not auto-detect an images folder. Browse the extracted directory:")
            print(f"   {extract_dir}")
            self.images_dir = extract_dir
            return extract_dir
        
        def count_images(d: Path) -> int:
            exts = {".jpg", ".jpeg", ".png"}
            return sum(1 for f in d.iterdir() if f.is_file() and f.suffix.lower() in exts)
        
        images_dir = max(candidates, key=count_images)
        print(f"Images folder found: {images_dir} ({count_images(images_dir)} images)")
        self.images_dir = images_dir
        return images_dir
    
    def extract_features(self):
        """Extract features from images."""
        if self.images_dir is None:
            raise ValueError("Images directory not set. Run download_and_extract_dataset() first.")
        
        device_name = "CUDA" if self.use_gpu else "CPU"
        print(f"Extracting features ({device_name})...")
        extraction_options = pycolmap.FeatureExtractionOptions()
        extraction_options.num_threads = self.num_threads
        
        pycolmap.extract_features(
            database_path=str(self.database_path),
            image_path=str(self.images_dir),
            camera_model="SIMPLE_RADIAL",
            extraction_options=extraction_options,
            device=self.device,
        )
        
        print("Feature extraction done")
        print(f"Database created at: {self.database_path}")
    
    def match_features(self):
        """Match features between image pairs."""
        device_name = "CUDA" if self.use_gpu else "CPU"
        print(f"Matching features ({device_name})...")
        
        matching_options = pycolmap.FeatureMatchingOptions()
        matching_options.use_gpu = self.use_gpu
        matching_options.num_threads = self.num_threads
        
        pairing_options = pycolmap.ExhaustivePairingOptions()
        verification_options = pycolmap.TwoViewGeometryOptions()
        
        pycolmap.match_exhaustive(
            database_path=str(self.database_path),
            matching_options=matching_options,
            pairing_options=pairing_options,
            verification_options=verification_options,
            device=self.device,
        )
        
        print("Feature matching done")
        print(f"Matches stored in: {self.database_path}")
    
    def run_sfm(self):
        """Run incremental Structure-from-Motion."""
        if self.images_dir is None:
            raise ValueError("Images directory not set. Run download_and_extract_dataset() first.")
        
        print("Running incremental SfM (full power)...")
        
        self.sparse_dir.mkdir(parents=True, exist_ok=True)
        
        reconstructions = pycolmap.incremental_mapping(
            database_path=str(self.database_path),
            image_path=str(self.images_dir),
            output_path=str(self.sparse_dir),
        )
        
        if len(reconstructions) == 0:
            raise RuntimeError("No reconstruction created")
        
        # Pick the largest reconstruction
        best_id, best_rec = max(
            reconstructions.items(),
            key=lambda kv: kv[1].num_images(),
        )
        
        self.sparse_model_dir = self.sparse_dir / str(best_id)
        
        print("SfM completed")
        print(f"Sparse model directory: {self.sparse_model_dir}")
        print(f"Registered images: {best_rec.num_images()}")
        print(f"3D points: {best_rec.num_points3D()}")
        
        print("\nKey outputs:")
        print(f" - {self.sparse_model_dir / 'cameras.bin'}")
        print(f" - {self.sparse_model_dir / 'images.bin'}")
        print(f" - {self.sparse_model_dir / 'points3D.bin'}")
    
    def visualize_sparse(self):
        """Visualize the sparse reconstruction."""
        if self.sparse_model_dir is None:
            raise ValueError("Sparse model not created. Run run_sfm() first.")
        
        print("Loading sparse reconstruction...")
        
        reconstruction = pycolmap.Reconstruction(self.sparse_model_dir)
        
        print(f"Cameras: {reconstruction.num_images()}")
        print(f"3D points: {reconstruction.num_points3D()}")
        
        # Collect 3D points
        points = []
        colors = []
        
        for p in reconstruction.points3D.values():
            points.append(p.xyz)
            colors.append(p.color / 255.0)  # normalize RGB
        
        points = np.array(points)
        colors = np.array(colors)
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        print("Opening 3D viewer...")
        o3d.visualization.draw_geometries([pcd])
    
    def export_sparse_ply(self, output_path: Path = None):
        """
        Export sparse point cloud to PLY format.
        
        Args:
            output_path: Optional output path. If None, uses default path.
        """
        if self.sparse_model_dir is None:
            raise ValueError("Sparse model not created. Run run_sfm() first.")
        
        if output_path is None:
            output_path = self.sparse_ply_path
        
        reconstruction = pycolmap.Reconstruction(self.sparse_model_dir)
        
        points = []
        colors = []
        
        for p in reconstruction.points3D.values():
            points.append(p.xyz)
            colors.append(p.color)
        
        points = np.array(points)
        colors = np.array(colors)
        
        with open(output_path, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            
            for p, c in zip(points, colors):
                f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        
        print(f"Exported sparse point cloud to: {output_path}")
    
    def run_dense(self):
        """Run dense reconstruction (MVS)."""
        if self.images_dir is None:
            raise ValueError("Images directory not set. Run download_and_extract_dataset() first.")
        if self.sparse_model_dir is None:
            raise ValueError("Sparse model not created. Run run_sfm() first.")
        
        self.dense_dir.mkdir(parents=True, exist_ok=True)
        
        print("Step 3.1: Undistorting images for MVS...")
        pycolmap.undistort_images(
            output_path=str(self.dense_dir),
            input_path=str(self.sparse_model_dir),
            image_path=str(self.images_dir),
            output_type="COLMAP",
        )
        
        device_name = "CUDA" if self.use_gpu else "CPU"
        print(f"Step 3.2: Running PatchMatch stereo ({device_name})...")
        pycolmap.patch_match_stereo(
            workspace_path=str(self.dense_dir),
        )
        
        print("Step 3.3: Fusing depth maps into dense point cloud...")
        fused_ply = self.dense_dir / "fused.ply"
        pycolmap.stereo_fusion(
            workspace_path=str(self.dense_dir),
            output_path=str(fused_ply),
        )
        
        print("Dense reconstruction done")
        print(f"Dense point cloud: {fused_ply}")
