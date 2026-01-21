from pathlib import Path
import pycolmap
import open3d as o3d
import numpy as np

SPARSE_MODEL_DIR = Path("colmap_cpu_project/outputs/sparse/0")


def visualize_sparse_model(model_dir: Path):
    print("📂 Loading sparse reconstruction...")

    # Load COLMAP reconstruction
    reconstruction = pycolmap.Reconstruction(model_dir)

    print(f"📷 Cameras: {reconstruction.num_images()}")
    print(f"☁️  3D points: {reconstruction.num_points3D()}")

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

    print("👀 Opening 3D viewer...")
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    visualize_sparse_model(SPARSE_MODEL_DIR)
