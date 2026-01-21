from pathlib import Path
import pycolmap

# 🔁 CHANGE THIS to the images path printed by dataset.py
IMAGES_DIR = Path("colmap_cpu_project/south-building/south-building/images")

# Output folder
OUT_DIR = Path("colmap_cpu_project/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_PATH = OUT_DIR / "database.db"


def extract_features_cpu(images_dir: Path, database_path: Path):
    print("🔎 Extracting features (CPU)...")
    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.num_threads = 6  
    pycolmap.extract_features(
        database_path=str(database_path),
        image_path=str(images_dir),
        camera_model="SIMPLE_RADIAL",
        extraction_options=extraction_options,
        device=pycolmap.Device.cpu,
    )

    print("✅ Feature extraction done")
    print(f"Database created at: {database_path}")


if __name__ == "__main__":
    extract_features_cpu(IMAGES_DIR, DATABASE_PATH)
