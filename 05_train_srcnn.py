"""
05_train_srcnn.py
-----------------
Step 5: Train a minimal SRCNN (Super-Resolution CNN) in PyTorch.

Architecture: Dong et al. 2014, 3 convolutional layers
  Conv(1->64, 9x9, pad=4) + ReLU  -- patch extraction
  Conv(64->32, 1x1) + ReLU        -- non-linear mapping
  Conv(32->1, 5x5, pad=2)         -- reconstruction

Trains on grayscale Y-channel patches extracted from existing HR/LR frame pairs.
Evaluates on full test frames and compares per-frame PSNR/SSIM vs bicubic baseline.

Outputs:
  output/models/srcnn.pth          -- trained model weights
  output/results/srcnn_metrics.csv -- per-frame PSNR/SSIM comparison table
"""

import cv2
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
HR_DIR      = "output/frames/hr"
LR_DIR      = "output/frames/lr"
MODELS_DIR  = "output/models"
RESULTS_DIR = "output/results"

PATCH_SIZE        = 32
PATCHES_PER_FRAME = 300
BATCH_SIZE        = 64
EPOCHS            = 100
LR_RATE           = 1e-3
TRAIN_SPLIT       = 0.8

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────
class SRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, 1)
        self.conv3 = nn.Conv2d(32, 1, 5, padding=2)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return self.conv3(x)


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class PatchDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs  = torch.from_numpy(np.stack(inputs))   # (N, 1, P, P)
        self.targets = torch.from_numpy(np.stack(targets))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def bgr_to_y(bgr_img):
    """Return Y channel (luminance) of a BGR image, float32 normalized to [0, 1]."""
    ycrcb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCrCb)
    return ycrcb[:, :, 0].astype(np.float32) / 255.0


def load_frame_pair(filename):
    """Load HR/LR pair and return (bicubic_y, hr_y) as float32 [0,1] arrays."""
    hr_bgr = cv2.imread(os.path.join(HR_DIR, filename))
    lr_bgr = cv2.imread(os.path.join(LR_DIR, filename))
    h, w   = hr_bgr.shape[:2]
    bicubic_bgr = cv2.resize(lr_bgr, (w, h), interpolation=cv2.INTER_CUBIC)
    return bgr_to_y(bicubic_bgr), bgr_to_y(hr_bgr)


def extract_patches(bicubic_y, hr_y, n_patches, patch_size):
    """Extract n_patches random (input, target) patch pairs from a frame."""
    h, w = hr_y.shape
    inputs, targets = [], []
    for _ in range(n_patches):
        r = np.random.randint(0, h - patch_size)
        c = np.random.randint(0, w - patch_size)
        inputs.append(bicubic_y[r:r+patch_size, c:c+patch_size][np.newaxis])
        targets.append(hr_y[r:r+patch_size, c:c+patch_size][np.newaxis])
    return inputs, targets


def infer_full_frame(model, bicubic_y, device):
    """Run SRCNN on a full Y-channel image. Returns float32 [0,1] numpy array."""
    model.eval()
    with torch.no_grad():
        t = torch.from_numpy(bicubic_y[np.newaxis, np.newaxis]).to(device)
        out = model(t).squeeze().cpu().numpy()
    return np.clip(out, 0.0, 1.0)


def compute_psnr_ssim(ref_uint8, pred_uint8):
    """Compute PSNR and SSIM between two uint8 grayscale images."""
    p = float(psnr_metric(ref_uint8, pred_uint8, data_range=255))
    s = float(ssim_metric(ref_uint8, pred_uint8, data_range=255))
    return p, s


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Frame split ──
    filenames   = sorted(os.listdir(HR_DIR))
    n           = len(filenames)
    split       = int(n * TRAIN_SPLIT)
    train_files = filenames[:split]
    test_files  = filenames[split:]
    print(f"Frames: {n} total | {len(train_files)} train | {len(test_files)} test")

    # ── Extract training patches ──
    print("Extracting training patches...")
    all_inputs, all_targets = [], []
    for fname in train_files:
        bicubic_y, hr_y = load_frame_pair(fname)
        inp_patches, tgt_patches = extract_patches(
            bicubic_y, hr_y, PATCHES_PER_FRAME, PATCH_SIZE
        )
        all_inputs.extend(inp_patches)
        all_targets.extend(tgt_patches)

    dataset = PatchDataset(all_inputs, all_targets)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"  {len(dataset)} patches | {len(loader)} batches/epoch")

    # ── Train ──
    model     = SRCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE)
    criterion = nn.MSELoss()

    print(f"\nTraining SRCNN for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            avg_loss = total_loss / len(loader)
            print(f"  Epoch {epoch:3d}/{EPOCHS} — avg loss: {avg_loss:.6f}")

    model_path = os.path.join(MODELS_DIR, "srcnn.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

    # ── Evaluate on test frames ──
    print("\nEvaluating on test frames...")
    rows = []
    for fname in test_files:
        bicubic_y, hr_y = load_frame_pair(fname)

        hr_uint8      = (hr_y * 255).astype(np.uint8)
        bicubic_uint8 = (bicubic_y * 255).astype(np.uint8)
        srcnn_y       = infer_full_frame(model, bicubic_y, device)
        srcnn_uint8   = (srcnn_y * 255).astype(np.uint8)

        p_bic, s_bic = compute_psnr_ssim(hr_uint8, bicubic_uint8)
        p_src, s_src = compute_psnr_ssim(hr_uint8, srcnn_uint8)

        rows.append({
            "filename":     fname,
            "psnr_bicubic": round(p_bic, 4),
            "ssim_bicubic": round(s_bic, 4),
            "psnr_srcnn":   round(p_src, 4),
            "ssim_srcnn":   round(s_src, 4),
        })

    df = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS_DIR, "srcnn_metrics.csv")
    df.to_csv(out_csv, index=False)

    print(f"\n{'Metric':<20} {'Bicubic':>10} {'SRCNN':>10} {'Delta':>10}")
    print("-" * 52)
    m_p_bic = df["psnr_bicubic"].mean()
    m_p_src = df["psnr_srcnn"].mean()
    m_s_bic = df["ssim_bicubic"].mean()
    m_s_src = df["ssim_srcnn"].mean()
    print(f"{'Mean PSNR (dB)':<20} {m_p_bic:>10.2f} {m_p_src:>10.2f} {m_p_src - m_p_bic:>+10.2f}")
    print(f"{'Mean SSIM':<20} {m_s_bic:>10.4f} {m_s_src:>10.4f} {m_s_src - m_s_bic:>+10.4f}")
    print(f"\nSaved per-frame metrics to: {out_csv}")
