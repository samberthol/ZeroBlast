import argparse
import logging
import os
import random
import tarfile
import tempfile
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from tqdm import tqdm
from google.cloud import storage
from ultralytics import YOLO

from utils import download_blob, parse_gcs_url

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_and_extract_shard(gs_path, extract_root):
    filename = Path(gs_path).name
    local_archive = extract_root / filename
    try:
        download_blob(gs_path, str(local_archive))
        with tarfile.open(local_archive, "r:gz") as tar:
            tar.extractall(path=extract_root)
        os.remove(local_archive)
        return True
    except Exception as e:
        logging.error(f"Failed to process {gs_path}: {e}")
        return False

def prepare_yolo_dataset(gcs_url, local_dir, val_percent=0.1):
    bucket_name, prefix = parse_gcs_url(gcs_url)
    storage_client = storage.Client()
    blobs = [f"gs://{bucket_name}/{b.name}" for b in storage_client.list_blobs(bucket_name, prefix=prefix) if b.name.endswith('.tar.gz')]
    
    raw_extract = Path(local_dir) / "raw"
    raw_extract.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Downloading {len(blobs)} archives...")
    with ThreadPoolExecutor(max_workers=8) as exe:
        list(tqdm(exe.map(lambda b: download_and_extract_shard(b, raw_extract), blobs), total=len(blobs)))
    
    # YOLO structure: images/train, images/val, labels/train, labels/val
    yolo_root = Path(local_dir) / "yolo_data"
    for split in ['train', 'val']:
        (yolo_root / 'images' / split).mkdir(parents=True, exist_ok=True)
        (yolo_root / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    image_files = list((raw_extract / 'images').glob('*.npy'))
    random.shuffle(image_files)
    
    n_val = int(len(image_files) * val_percent)
    val_files = image_files[:n_val]
    train_files = image_files[n_val:]
    
    def move_to_yolo(files, split):
        for img_path in tqdm(files, desc=f"Preparing {split}"):
            # Convert .npy to .png for YOLO (YOLOv11 can be picky with .npy if not configured)
            # However, since we have 16-bit float data, we might want to scale it.
            # For simplicity in this plan, we'll save as .npy and see if YOLO11 handles it, 
            # or convert to 8-bit for standard pretrained YOLO weights.
            # Let's convert to 8-bit grayscale for better transfer learning.
            img = np.load(img_path)
            # Normalize -300 to 300 to 0-255
            img_8bit = ((np.clip(img, -300, 300) + 300) / 600 * 255).astype(np.uint8)
            
            from PIL import Image
            out_img = yolo_root / 'images' / split / f"{img_path.stem}.png"
            Image.fromarray(img_8bit).save(out_img)
            
            # Label
            label_path = raw_extract / 'labels' / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy(label_path, yolo_root / 'labels' / split / f"{img_path.stem}.txt")
            else:
                # Create empty label file for background patches
                open(yolo_root / 'labels' / split / f"{img_path.stem}.txt", 'a').close()

    move_to_yolo(train_files, 'train')
    move_to_yolo(val_files, 'val')
    
    # Create dataset.yaml
    yaml_content = f"""
path: {yolo_root.absolute()}
train: images/train
val: images/val
names:
  0: UXO
"""
    yaml_path = yolo_root / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
        
    return yaml_path

def train_yolo(data_yaml, epochs, batch_size, imgsz, model_name='yolo11n.pt'):
    model = YOLO(model_name)
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        save=True,
        device=0 if torch.cuda.is_available() else 'cpu'
    )
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--data-bucket', type=str, required=True)
    parser.add_argument('--imgsz', type=int, default=256)
    parser.add_argument('--model', type=str, default='yolo11n.pt')
    parser.add_argument('--checkpoint-dir', type=str, required=True)
    args = parser.parse_args()

    random.seed(42)
    
    with tempfile.TemporaryDirectory() as tmp:
        yaml_path = prepare_yolo_dataset(args.data_bucket, tmp)
        train_yolo(yaml_path, args.epochs, args.batch_size, args.imgsz, args.model)
        
        # Upload best model to GCS
        best_model = Path("runs/detect/train/weights/best.pt")
        if best_model.exists():
            from utils import upload_blob
            upload_blob(str(best_model), f"{args.checkpoint_dir.rstrip('/')}/best.pt")
            logging.info(f"Uploaded best model to {args.checkpoint_dir}/best.pt")
