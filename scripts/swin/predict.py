import argparse
import logging
import os
import shutil
import numpy as np
import rasterio
import torch
import torchvision.transforms.functional as TF
from rasterio.windows import Window
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from model import SwinUNet
from utils import download_blob, upload_blob, resample_raster, parse_gcs_url

PATCH_SIZE = 256
MODEL_INPUT_SIZE = 224
STRIDE = 128
TARGET_RES = 0.10
GAUSSIAN_SIGMA = 64.0

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@torch.inference_mode()
def predict_on_raster(input_raster_path, output_raster_path, model_path, batch_size=16):
    local_input = input_raster_path
    local_model = model_path
    
    if input_raster_path.startswith("gs://"):
        local_input = "/tmp/input_raster.tif"
        download_blob(input_raster_path, local_input)
        
    if model_path.startswith("gs://"):
        local_model = "/tmp/model.pth"
        download_blob(model_path, local_model)

    # --- Resolution Normalization ---
    with rasterio.open(local_input) as src:
        res_x = src.transform[0]
        res_y = abs(src.transform[4])
        logging.info(f"Input resolution: {res_x:.4f}m x {res_y:.4f}m")
        
    if abs(res_x - TARGET_RES) > 1e-3 or abs(res_y - TARGET_RES) > 1e-3:
        local_input = resample_raster(local_input, "/tmp/resampled_input.tif", target_res=TARGET_RES)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # v57: SwinUNet (cloned from v56)
    model = SwinUNet(n_channels=1, n_classes=1, pretrained=False)
    
    state = torch.load(local_model, map_location=device)
    # Handle both full state dicts and partial ones
    state = state['model'] if 'model' in state else (state['model_state_dict'] if 'model_state_dict' in state else state)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    logging.info(f"Starting inference on {local_input}...")
    local_output = "/tmp/output_raster.tif"
    
    with rasterio.open(local_input) as src:
        meta = src.meta.copy()
        meta.update(count=1, dtype='float32', nodata=-9999.0, compress='lzw')
        src_nodata = src.nodata if src.nodata is not None else 32767.0
        
        pred_agg = np.zeros((src.height, src.width), dtype=np.float32)
        count_agg = np.zeros((src.height, src.width), dtype=np.uint8)

        for r_offset in tqdm(range(0, src.height, STRIDE)):
            patches = []
            windows = []
            for c_offset in range(0, src.width, STRIDE):
                window = Window(c_offset, r_offset, min(PATCH_SIZE, src.width-c_offset), min(PATCH_SIZE, src.height-r_offset))
                
                if window.width < PATCH_SIZE or window.height < PATCH_SIZE:
                    padded = np.full((PATCH_SIZE, PATCH_SIZE), src_nodata, dtype=np.float32)
                    d = src.read(1, window=window)
                    padded[:window.height, :window.width] = d
                    patch = padded
                else:
                    patch = src.read(1, window=window).astype(np.float32)

                # --- Pre-processing (v38 Style: High-Pass Trend Subtraction) ---
                nodata_mask = (patch == src_nodata) | np.isnan(patch)
                valid = patch[~nodata_mask]
                mean_val = np.mean(valid) if valid.size > 0 else 0
                patch[nodata_mask] = mean_val
                
                low_freq = gaussian_filter(patch, sigma=GAUSSIAN_SIGMA)
                patch_hp = np.clip(patch - low_freq, -300, 300)
                
                patch_tensor = torch.from_numpy(patch_hp).unsqueeze(0) # (1, H, W)
                patch_tensor = TF.normalize(patch_tensor, [0], [300])
                
                # --- v57 Specific: Resize to 224x224 ---
                patch_tensor = TF.resize(patch_tensor, [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE], antialias=True)
                
                patches.append(patch_tensor)
                windows.append(window)

            if not patches: continue

            # Create batches manually to handle edge cases
            for i in range(0, len(patches), batch_size):
                batch_tensors = torch.stack(patches[i:i+batch_size]).to(device)
                logits, _, _ = model(batch_tensors)
                probs = torch.sigmoid(logits)
                
                # Resize back to PATCH_SIZE if needed, or handle window properly
                probs = TF.resize(probs, [PATCH_SIZE, PATCH_SIZE], antialias=True)
                probs = probs.cpu().numpy().squeeze(1)
                
                if len(probs.shape) == 2: # handle batch_size=1
                    probs = np.expand_dims(probs, axis=0)

                for j in range(probs.shape[0]):
                    w = windows[i+j]
                    p = probs[j, :w.height, :w.width]
                    pred_agg[w.row_off:w.row_off+w.height, w.col_off:w.col_off+w.width] += p
                    count_agg[w.row_off:w.row_off+w.height, w.col_off:w.col_off+w.width] += 1

        valid = count_agg > 0
        pred_agg[valid] /= count_agg[valid]
        pred_agg[~valid] = meta['nodata']
        
        with rasterio.open(local_output, 'w', **meta) as dst:
            dst.write(pred_agg, 1)

    if output_raster_path.startswith("gs://"):
        upload_blob(local_output, output_raster_path)
    else:
        shutil.copy2(local_output, output_raster_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-raster", required=True)
    parser.add_argument("--output-raster", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    predict_on_raster(args.input_raster, args.output_raster, args.model, args.batch_size)
