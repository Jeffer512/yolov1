import argparse
import os
import shutil
from pathlib import Path

# Symmetrical 4px vertical padding parameters (1920x1200 -> 448x280 -> 448x288)
WIDTH = 448
UNPADDED_H = 280
PADDED_H = 288
TOP_PAD_PX = 4
BOTTOM_PAD_PX = 4

def download_dataset(root_dir, api_key):
    """Downloads the raw dataset from the Roboflow platform.

    Args:
        root_dir (Path): Destination path for the raw download.
        api_key (str): Roboflow private API authorization key.
    """
    from roboflow import Roboflow
    
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("roboflow-gw7yv").project("self-driving-car")
    version = project.version(1)
    
    # Download raw assets into the raw directory
    version.download("yolov5", location=str(root_dir))


def split_dataset(root_dir):
    """Splits a consolidated dataset folder into train, valid, and test sets.

    Performs sequential block splitting by sorting filenames lexicographically. 
    This prevents temporal data leakage under the assumption that the file 
    naming convention allows alphabetical sorting to preserve chronological 
    frame sequences.

    Args:
        root_dir (Path): Path to the raw dataset root folder containing the export.
    """
    root_path = Path(root_dir)
    export_images = root_path / "export" / "images"
    export_labels = root_path / "export" / "labels"

    if not export_images.exists() or not export_labels.exists():
        raise FileNotFoundError(
            f"Could not find 'export/images' or 'export/labels' inside {root_dir}. "
            "Aborting split execution."
        )

    # Gather valid image files
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(export_images.glob(ext)))

    # Sort files chronologically by timestamp prefix to enforce block splitting
    image_files.sort()

    # Calculate sequential partition boundaries (70/20/10 ratio)
    total_files = len(image_files)
    train_end = int(total_files * 0.7)
    valid_end = train_end + int(total_files * 0.2)

    splits = {
        "train": image_files[:train_end],
        "valid": image_files[train_end:valid_end],
        "test": image_files[valid_end:]
    }
    
    # Construct splits and move sequential file blocks
    for split, files in splits.items():
        split_img_dir = root_path / split / "images"
        split_lbl_dir = root_path / split / "labels"
        
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Moving {len(files)} files to '{split}' split...")

        for img_path in files:
            label_filename = img_path.stem + ".txt"
            lbl_path = export_labels / label_filename

            # Move image file
            shutil.move(str(img_path), str(split_img_dir / img_path.name))

            # Move corresponding label file if exists
            if lbl_path.exists():
                shutil.move(str(lbl_path), str(split_lbl_dir / label_filename))

    # Purge the empty export directory
    shutil.rmtree(str(export_images.parent))
    print("Dataset splitting complete. Removed empty export directory.")


def preprocess_dataset(src_dir, dest_dir):
    """Resizes, pads, and translates dataset split files offline.

    Processes images and matching annotations from splits. Excludes duplicate 
    files using filename pattern heuristics, resizes images to ``WIDTH`` x 
    ``UNPADDED_H``, pads them symmetrically to ``WIDTH`` x ``PADDED_H``, and 
    translates the vertical coordinates of the bounding boxes.

    Args:
        src_dir (str/Path): Root directory of the source raw split folders.
        dest_dir (str/Path): Target directory for the preprocessed outputs.
    """
    print(f"Beginning offline dataset preprocessing to {WIDTH}x{PADDED_H}...")
    from PIL import Image, ImageOps
    
    src_path = Path(src_dir)
    dest_path = Path(dest_dir)

    for split in ["train", "valid", "test"]:
        src_img_dir = src_path / split / "images"
        src_lbl_dir = src_path / split / "labels"
        
        if not src_img_dir.exists():
            continue
            
        dest_img_dir = dest_path / split / "images"
        dest_lbl_dir = dest_path / split / "labels"
        
        dest_img_dir.mkdir(parents=True, exist_ok=True)
        dest_lbl_dir.mkdir(parents=True, exist_ok=True)

        img_count = 0
        skipped_duplicates = 0
        
        for img_file in src_img_dir.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                
                # Check for and skip duplicates (assume 20-character substrings between .rf. and the extension correspond to duplicates)
                parts = img_file.name.split(".rf.")
                if len(parts) > 1:
                    raw_hash = Path(parts[1]).stem
                    if len(raw_hash) == 20:
                        skipped_duplicates += 1
                        continue
                
                # Symmetrical resize and padding preprocessing
                with Image.open(img_file) as img:
                    resized = img.resize((WIDTH, UNPADDED_H), Image.Resampling.BILINEAR)
                    padded = ImageOps.expand(resized, border=(0, TOP_PAD_PX, 0, BOTTOM_PAD_PX), fill=(0, 0, 0))
                    padded.save(dest_img_dir / img_file.name, quality=95)

                # Reconstruct and shift vertical bounding box coordinates
                lbl_file = src_lbl_dir / (img_file.stem + ".txt")
                if lbl_file.exists():
                    new_lines = []
                    with open(lbl_file, "r") as f:
                        for line in f.readlines():
                            class_id, x, y, w, h = map(float, line.split())
                            
                            # Apply 4-pixel vertical offset and scale height proportionally
                            y_new = ((y * UNPADDED_H) + TOP_PAD_PX) / PADDED_H
                            h_new = h * (UNPADDED_H / PADDED_H)
                            
                            new_lines.append(f"{int(class_id)} {x:.8f} {y_new:.8f} {w:.8f} {h_new:.8f}\n")
                            
                    with open(dest_lbl_dir / lbl_file.name, "w") as f:
                        f.writelines(new_lines)
                        
                img_count += 1
                
        print(f"[{split}] Processed {img_count} unique images. Skipped {skipped_duplicates} duplicates.")

    # Move the dataset configuration YAML to the preprocessed directory
    if (src_path / "data.yaml").exists():
        shutil.move(str(src_path / "data.yaml"), str(dest_path / "data.yaml"))
        
    print("Offline preprocessing complete.")


def main(args):
    """Orchestrates the full data ingestion, splitting, and preprocessing pipeline."""
    raw_dir = Path(args.raw_dir)
    preprocessed_dir = Path(args.processed_dir)

    # Skip process if target output folder already exists
    if preprocessed_dir.exists():
        print(f"Preprocessed dataset directory '{preprocessed_dir}' already exists. Skipping setup.")
        return

    # Trigger raw dataset download if missing
    if not raw_dir.exists():
        print(f"Dataset directory '{raw_dir}' not found.")
        print("Initiating programmatic dataset download from Roboflow...")
        
        api_key = args.api_key or os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            raise ValueError(
                "Raw dataset folder missing. Please provide a Roboflow API key via "
                "'--api_key' or set it in your local environment."
            )
        download_dataset(raw_dir, api_key)

    # Partition dataset chronologically if export folder exists
    export_images = raw_dir / "export" / "images"
    if export_images.exists():
        split_dataset(raw_dir)

    # Preprocess splits into optimized target layout
    preprocess_dataset(raw_dir, preprocessed_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv1 Data Ingestion and Preprocessing Pipeline")
    parser.add_argument("--api_key", type=str, default=None, help="Roboflow Private API Key")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Path to raw dataset directory")
    parser.add_argument("--processed_dir", type=str, default="data/processed", help="Path to preprocessed dataset directory")
    args = parser.parse_args()
    
    main(args)