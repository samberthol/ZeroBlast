import argparse
import logging
import os
import random
import tarfile
import re
import tempfile
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from google.cloud import storage

from model import HRNetSegmentation
from utils import download_blob, upload_blob, parse_gcs_url

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Data Prep Helpers ---
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

def prepare_local_dataset(gcs_url, local_dir):
    bucket_name, prefix = parse_gcs_url(gcs_url)
    if not bucket_name:
        raise ValueError(f"Invalid GCS URL: {gcs_url}")
    
    if not prefix.endswith('/'):
        prefix += '/'
    
    storage_client = storage.Client()
    iter_blobs = storage_client.list_blobs(bucket_name, prefix=prefix)
    
    # v57/v58: Zone exclusion is handled at the dataprep stage.
    blobs = [f"gs://{bucket_name}/{b.name}" for b in iter_blobs if b.name.endswith('.tar.gz')]
    
    if not blobs:
        raise RuntimeError(f"No .tar.gz files found in {gcs_url}")
        
    logging.info(f"Found {len(blobs)} archives. Downloading and extracting to {local_dir}...")
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    
    with ThreadPoolExecutor(max_workers=8) as exe:
        futures = [exe.submit(download_and_extract_shard, b, local_dir) for b in blobs]
        for f in tqdm(futures, total=len(blobs), desc="Preparing Data"):
            f.result()
            
    images_dir = local_dir / "images"
    masks_dir = local_dir / "masks"
    if not images_dir.exists() or not masks_dir.exists():
        raise RuntimeError("Dataset extraction failed: 'images' or 'masks' directories missing.")
        
    return images_dir, masks_dir

def save_model(model, model_dir, filename='best_model.pth'):
    local_path = f'/tmp/{filename}'
    torch.save(model.state_dict(), local_path)
    
    if model_dir.startswith("gs://"):
        logging.info(f"Uploading model to {model_dir}/{filename}...")
        upload_blob(local_path, f"{model_dir.rstrip('/')}/{filename}")
    else:
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(local_path, os.path.join(model_dir, filename))
    
    if os.path.exists(local_path):
        os.remove(local_path)

# --- Data Loading ---
class DipoleDataset(Dataset):
    def __init__(self, images_dir, masks_dir, augment=False, ids=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.augment = augment
        
        all_ids = sorted([f.stem for f in self.images_dir.glob('*.npy')]) if not ids else ids
        if not all_ids:
            raise RuntimeError(f'No images found in {images_dir}.')
        self.ids = all_ids
        logging.info(f'Loaded {len(self.ids)} image/mask pairs.')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        image = np.load(self.images_dir / f'{name}.npy').astype(np.float32)
        mask = np.load(self.masks_dir / f'{name.replace("_image", "_mask")}.npy').astype(np.float32)

        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask)
        
        image = torch.clamp(image, -300, 300)
        image = TF.normalize(image, [0], [300]) # Roughly Normalize
        
        # HRNet handles 256x256 naturally. No resize needed.
        
        return {'image': image, 'mask': mask}

# --- Fractal Noise Generation (Priority 1) ---
def generate_fractal_noise(batch_size, channels, height, width, device, alpha=1.5):
    """Generates 2D fractal (1/f^alpha) noise using IFFT."""
    noise = torch.randn(batch_size, channels, height, width, device=device)
    fft = torch.fft.fftn(noise, dim=(-2, -1))
    freq_y = torch.fft.fftfreq(height, device=device).view(1, 1, height, 1)
    freq_x = torch.fft.fftfreq(width, device=device).view(1, 1, 1, width)
    rho = torch.sqrt(freq_y**2 + freq_x**2)
    rho[rho == 0] = 1.0 
    f_filter = 1.0 / (rho ** alpha)
    f_filter[..., 0, 0] = 0 
    filtered_fft = fft * f_filter
    fractal_noise = torch.fft.ifftn(filtered_fft, dim=(-2, -1)).real
    return (fractal_noise / fractal_noise.std()) * 0.05

# --- GPU Augmentation ---
def augment_batch(images, masks):
    if random.random() > 0.5:
        images = TF.hflip(images)
        masks = TF.hflip(masks)
    
    # v61: Replace Gaussian Noise with Fractal Noise (Priority 1)
    noise = generate_fractal_noise(images.shape[0], images.shape[1], images.shape[2], images.shape[3], images.device, alpha=1.5)
    images = images + noise
    images = torch.clamp(images, -1, 1)
    return images, masks

# --- Loss & Evaluation ---
class AdaptiveWingLoss(nn.Module):
    def __init__(self, omega=14.0, theta=0.5, epsilon=1.0, alpha=2.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred_logits, target):
        y = target
        y_hat = torch.sigmoid(pred_logits)
        delta_y = (y - y_hat).abs()
        
        # AWing parameters for y_hat near y
        A = self.omega * (1.0 / (1.0 + torch.pow(self.theta / self.epsilon, self.alpha - y))) * (self.alpha - y) * (torch.pow(self.theta / self.epsilon, self.alpha - y - 1.0)) * (1.0 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(1.0 + torch.pow(self.theta / self.epsilon, self.alpha - y))
        
        loss = torch.where(
            delta_y < self.theta,
            self.omega * torch.log(1.0 + torch.pow(delta_y / self.epsilon, self.alpha - y)),
            A * delta_y - C
        )
        return loss.mean()

class FocalMSELoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75):
        super(FocalMSELoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred_logits, target_heatmap):
        pred_prob = torch.sigmoid(pred_logits)
        mse = F.mse_loss(pred_prob, target_heatmap, reduction='none')
        p_t = torch.where(target_heatmap > 0.5, pred_prob, 1 - pred_prob)
        focal_weight = (1 - p_t).pow(self.gamma)
        alpha_t = torch.where(target_heatmap > 0.5, self.alpha, 1 - self.alpha)
        focal_mse_loss = alpha_t * focal_weight * mse
        return focal_mse_loss.mean()

@torch.inference_mode()
def evaluate(net, dataloader, device, criterion):
    net.eval()
    num_val_batches = len(dataloader)
    total_loss = 0
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation', leave=False):
        image, mask_true = batch['image'].to(device), batch['mask'].unsqueeze(1).to(device)
        pred_logits = net.predict(image)
        total_loss += criterion(pred_logits, mask_true).item()
    net.train()
    return total_loss / max(num_val_batches, 1)

def train_model(model, device, images_dir, masks_dir, epochs, batch_size, lr, val_percent, amp, checkpoint_dir, weight_decay):
    dataset = DipoleDataset(images_dir, masks_dir, augment=False)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_set, shuffle=False, batch_size=batch_size, num_workers=8, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = AdaptiveWingLoss() # v61: Switched to AWing (Priority 2)
    
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'].to(device), batch['mask'].unsqueeze(1).to(device)
                images, true_masks = augment_batch(images, true_masks)

                with torch.autocast(device.type, enabled=amp):
                    logits4, logits3, logits2 = model(images)
                    loss4 = criterion(logits4, true_masks)
                    loss3 = criterion(logits3, F.interpolate(true_masks, size=logits3.shape[2:]))
                    loss2 = criterion(logits2, F.interpolate(true_masks, size=logits2.shape[2:]))
                    loss = (0.5 * loss4) + (0.3 * loss3) + (0.2 * loss2)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss': f'{loss.item():.4f}'})

        val_loss = evaluate(model, val_loader, device, criterion)
        scheduler.step()
        logging.info(f'Validation Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, checkpoint_dir, 'best_model.pth')
            logging.info(f'New best model saved with loss {best_val_loss:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500) 
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--data-bucket', type=str, required=True)
    parser.add_argument('--checkpoint-dir', type=str, default='gs://pyro-models-us-hostproject1-355311/checkpoints/v62_hrnet')
    parser.add_argument('--amp', action='store_true', default=True)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = False # Stability fix for generic CUDA errors
    model = HRNetSegmentation(n_channels=1, n_classes=1).to(device)

    with tempfile.TemporaryDirectory() as tmp:
        images_dir, masks_dir = prepare_local_dataset(args.data_bucket, tmp)
        train_model(model, device, images_dir, masks_dir, args.epochs, args.batch_size, args.lr, 0.1, args.amp, args.checkpoint_dir, args.weight_decay)
