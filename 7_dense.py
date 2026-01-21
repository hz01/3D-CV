from pathlib import Path
import pycolmap

IMAGES_DIR = Path("colmap_cpu_project/south-building/south-building/images")
SPARSE_MODEL_DIR = Path("colmap_cpu_project/outputs/sparse/0")
DENSE_DIR = Path("colmap_cpu_project/outputs/dense")


def run_dense_cpu(images_dir: Path, sparse_model_dir: Path, dense_dir: Path):
    dense_dir.mkdir(parents=True, exist_ok=True)

    print("🧽 Step 3.1: Undistorting images for MVS...")
    pycolmap.undistort_images(
        output_path=str(dense_dir),
        input_path=str(sparse_model_dir),
        image_path=str(images_dir),
        output_type="COLMAP",
    )

    print("🧠 Step 3.2: Running PatchMatch stereo (CPU)...")
    pycolmap.patch_match_stereo(
        workspace_path=str(dense_dir),
    )

    print("🧩 Step 3.3: Fusing depth maps into dense point cloud...")
    fused_ply = dense_dir / "fused.ply"
    pycolmap.stereo_fusion(
        workspace_path=str(dense_dir),
        output_path=str(fused_ply),
    )

    print("✅ Dense reconstruction done")
    print(f"📄 Dense point cloud: {fused_ply}")


if __name__ == "__main__":
    run_dense_cpu(IMAGES_DIR, SPARSE_MODEL_DIR, DENSE_DIR)
