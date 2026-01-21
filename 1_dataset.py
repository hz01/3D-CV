from pathlib import Path
from urllib.request import urlretrieve
import zipfile

# Official COLMAP dataset zip (South Building)
DATASET_URL = "https://github.com/colmap/colmap/releases/download/3.11.1/south-building.zip"


def download_and_extract(url: str, out_root: str = "colmap_cpu_project") -> Path:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    zip_path = out_root / "south-building.zip"
    extract_dir = out_root / "south-building"

    # 1) Download
    if not zip_path.exists():
        print(f"⬇️ Downloading:\n{url}")
        urlretrieve(url, zip_path)
        print(f"✅ Saved zip to: {zip_path}")
    else:
        print(f"✅ Zip already exists: {zip_path}")

    # 2) Extract
    if not extract_dir.exists():
        print(f"📦 Extracting to: {extract_dir}")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        print("✅ Extracted.")
    else:
        print(f"✅ Already extracted: {extract_dir}")

    # 3) Best-effort: find images folder
    candidates = []
    for p in extract_dir.rglob("*"):
        if p.is_dir() and p.name.lower() in ("images", "imgs", "image", "rgb"):
            candidates.append(p)

    if not candidates:
        print("⚠️ Could not auto-detect an images folder. Browse the extracted directory:")
        print(f"   {extract_dir}")
        return extract_dir

    def count_images(d: Path) -> int:
        exts = {".jpg", ".jpeg", ".png"}
        return sum(1 for f in d.iterdir() if f.is_file() and f.suffix.lower() in exts)

    images_dir = max(candidates, key=count_images)
    print(f"🖼️ Images folder found: {images_dir} ({count_images(images_dir)} images)")
    return images_dir


if __name__ == "__main__":
    images_dir = download_and_extract(DATASET_URL)
    print("\nDONE.")
    print(f"Use this images path later:\n{images_dir}")
