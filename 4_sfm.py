from pathlib import Path
import pycolmap

IMAGES_DIR = Path("colmap_cpu_project/south-building/south-building/images")
OUT_DIR = Path("colmap_cpu_project/outputs")
DATABASE_PATH = OUT_DIR / "database.db"
SPARSE_DIR = OUT_DIR / "sparse"


def run_incremental_sfm(database_path: Path, images_dir: Path, sparse_dir: Path):
    print("🧭 Running incremental SfM (full power)...")

    sparse_dir.mkdir(parents=True, exist_ok=True)

    # 🔥 RETURNS A DICT in your PyCOLMAP version
    reconstructions = pycolmap.incremental_mapping(
        database_path=str(database_path),
        image_path=str(images_dir),
        output_path=str(sparse_dir),
    )

    if len(reconstructions) == 0:
        raise RuntimeError("❌ No reconstruction created")

    # Pick the largest reconstruction
    best_id, best_rec = max(
        reconstructions.items(),
        key=lambda kv: kv[1].num_images(),
    )

    model_dir = sparse_dir / str(best_id)

    print("✅ SfM completed")
    print(f"📂 Sparse model directory: {model_dir}")
    print(f"📷 Registered images: {best_rec.num_images()}")
    print(f"☁️  3D points: {best_rec.num_points3D()}")

    print("\nKey outputs:")
    print(f" - {model_dir / 'cameras.bin'}")
    print(f" - {model_dir / 'images.bin'}")
    print(f" - {model_dir / 'points3D.bin'}")


if __name__ == "__main__":
    run_incremental_sfm(DATABASE_PATH, IMAGES_DIR, SPARSE_DIR)
