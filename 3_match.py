from pathlib import Path
import pycolmap

OUT_DIR = Path("colmap_cpu_project/outputs")
DATABASE_PATH = OUT_DIR / "database.db"


def match_features(database_path: Path):
    print("🔗 Matching features (full power)...")

    # 1️⃣ Generic feature matching options
    matching_options = pycolmap.FeatureMatchingOptions()
    matching_options.use_gpu = False
    matching_options.num_threads = 6

    # 2️⃣ Pairing strategy (exhaustive = all image pairs)
    pairing_options = pycolmap.ExhaustivePairingOptions()

    # 3️⃣ Geometric verification (RANSAC, epipolar checks)
    verification_options = pycolmap.TwoViewGeometryOptions()

    pycolmap.match_exhaustive(
        database_path=str(database_path),
        matching_options=matching_options,
        pairing_options=pairing_options,
        verification_options=verification_options,
        device=pycolmap.Device.cpu,
    )

    print("✅ Feature matching done")
    print(f"Matches stored in: {database_path}")


if __name__ == "__main__":
    match_features(DATABASE_PATH)
