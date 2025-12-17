import os
import random
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

def train_val_split(src_root='dataset/garbage classification',
                    dst_root='dataset/data',
                    val_ratio=0.2,
                    seed=42,
                    max_workers=4):  # adjust based on CPU cores
    """
    Splits dataset into train and validation folders (80/20 default),
    resizes images to 256x256, and saves them in the destination folder.
    Parallel processing is used for speed.
    """
    random.seed(seed)
    src_root = Path(src_root)
    dst_root = Path(dst_root)

    class_names = [d.name for d in src_root.iterdir() if d.is_dir()]

    def process_image(f, split, cls):
        out_dir = dst_root / split / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        dest_file = out_dir / f.name
        if not dest_file.exists():
            try:
                img = Image.open(f).convert("RGB")
                img = img.resize((256, 256))  # optional resize
                img.save(dest_file)
            except Exception as e:
                print(f"Error processing {f}: {e}")

    for cls in class_names:
        cls_path = src_root / cls
        files = list(cls_path.glob('*'))
        random.shuffle(files)
        n_val = int(len(files) * val_ratio)
        val_files = files[:n_val]
        train_files = files[n_val:]

        # Parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for f in train_files:
                executor.submit(process_image, f, 'train', cls)
            for f in val_files:
                executor.submit(process_image, f, 'val', cls)

    print(f"Train/Validation split complete!")
    print(f"Check folders: {dst_root}/train and {dst_root}/val")


if __name__ == '__main__':
    train_val_split()
