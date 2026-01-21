from pathlib import Path
import pycolmap
import numpy as np

SPARSE_MODEL_DIR = Path("colmap_cpu_project/outputs/sparse/0")
OUT_PLY = Path("colmap_cpu_project/outputs/sparse_points.ply")


def export_sparse_to_ply(model_dir: Path, out_ply: Path):
    reconstruction = pycolmap.Reconstruction(model_dir)

    points = []
    colors = []

    for p in reconstruction.points3D.values():
        points.append(p.xyz)
        colors.append(p.color)

    points = np.array(points)
    colors = np.array(colors)

    with open(out_ply, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")

        for p, c in zip(points, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")

    print(f"✅ Exported sparse point cloud to: {out_ply}")


if __name__ == "__main__":
    export_sparse_to_ply(SPARSE_MODEL_DIR, OUT_PLY)
