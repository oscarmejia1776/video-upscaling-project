"""
06_visualize_srcnn.py
---------------------
Step 6: Visualize SRCNN results against the bicubic baseline.

For 5 evenly-spaced test frames generates a 4-panel color comparison:
  Low-Resolution | Bicubic | SRCNN | Ground Truth HR

Also generates a summary metrics table image comparing bicubic vs SRCNN.

Outputs:
  output/images/srcnn_comparison_01.png ... _05.png
  output/images/srcnn_metrics_table.png
"""

import cv2
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
HR_DIR      = "output/frames/hr"
LR_DIR      = "output/frames/lr"
IMAGES_DIR  = "output/images"
RESULTS_DIR = "output/results"
MODEL_PATH  = "output/models/srcnn.pth"
TRAIN_SPLIT = 0.8
DISPLAY_H   = 540   # resize frames to this height for manageable figure size

os.makedirs(IMAGES_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# Model (must match 05_train_srcnn.py exactly)
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
# Helpers
# ─────────────────────────────────────────────
def reconstruct_color(lr_bgr, hr_bgr, model, device):
    """
    Apply SRCNN to the Y channel of the bicubic-upscaled LR frame, then
    merge with the bicubic Cb/Cr channels to produce a full-color result.
    Returns (bicubic_bgr, srcnn_bgr) as uint8 BGR images at HR resolution.
    """
    h, w = hr_bgr.shape[:2]
    bicubic_bgr   = cv2.resize(lr_bgr, (w, h), interpolation=cv2.INTER_CUBIC)
    bicubic_ycrcb = cv2.cvtColor(bicubic_bgr, cv2.COLOR_BGR2YCrCb)

    bicubic_y = bicubic_ycrcb[:, :, 0].astype(np.float32) / 255.0

    model.eval()
    with torch.no_grad():
        t   = torch.from_numpy(bicubic_y[np.newaxis, np.newaxis]).to(device)
        out = model(t).squeeze().cpu().numpy()
    srcnn_y = np.clip(out, 0.0, 1.0)

    srcnn_ycrcb          = bicubic_ycrcb.copy()
    srcnn_ycrcb[:, :, 0] = (srcnn_y * 255).astype(np.uint8)
    srcnn_bgr            = cv2.cvtColor(srcnn_ycrcb, cv2.COLOR_YCrCb2BGR)

    return bicubic_bgr, srcnn_bgr


def to_display(bgr_img, height, interp=cv2.INTER_LINEAR):
    """Resize a BGR image to a fixed display height, preserving aspect ratio."""
    h, w = bgr_img.shape[:2]
    scale = height / h
    return cv2.resize(bgr_img, (int(w * scale), height), interpolation=interp)


def save_comparison(lr_bgr, bicubic_bgr, srcnn_bgr, hr_bgr, save_path, idx):
    """Save a 4-panel side-by-side comparison image."""
    lr_disp  = to_display(lr_bgr,      DISPLAY_H, cv2.INTER_NEAREST)  # blocky look
    bic_disp = to_display(bicubic_bgr, DISPLAY_H)
    src_disp = to_display(srcnn_bgr,   DISPLAY_H)
    hr_disp  = to_display(hr_bgr,      DISPLAY_H)

    panels = [lr_disp, bic_disp, src_disp, hr_disp]
    titles = ["Low-Resolution", "Bicubic", "SRCNN", "Ground Truth HR"]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for ax, panel, title in zip(axes, panels, titles):
        ax.imshow(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=13)
        ax.axis("off")

    plt.suptitle(f"Super-Resolution Comparison — Sample {idx}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ──
    model = SRCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    print(f"Loaded SRCNN from {MODEL_PATH}")

    # ── Identify test frames (same split as step 5) ──
    filenames  = sorted(os.listdir(HR_DIR))
    split      = int(len(filenames) * TRAIN_SPLIT)
    test_files = filenames[split:]

    sample_indices = np.linspace(0, len(test_files) - 1, 5, dtype=int)
    sample_files   = [test_files[i] for i in sample_indices]

    # ── Comparison images ──
    print("Generating comparison images...")
    for i, fname in enumerate(sample_files, start=1):
        hr_bgr = cv2.imread(os.path.join(HR_DIR, fname))
        lr_bgr = cv2.imread(os.path.join(LR_DIR, fname))

        bicubic_bgr, srcnn_bgr = reconstruct_color(lr_bgr, hr_bgr, model, device)

        save_path = os.path.join(IMAGES_DIR, f"srcnn_comparison_{i:02d}.png")
        save_comparison(lr_bgr, bicubic_bgr, srcnn_bgr, hr_bgr, save_path, i)
        print(f"  Saved {save_path}")

    # ── Summary metrics table ──
    print("Generating metrics table...")
    df = pd.read_csv(os.path.join(RESULTS_DIR, "srcnn_metrics.csv"))

    m_p_bic = df["psnr_bicubic"].mean()
    m_s_bic = df["ssim_bicubic"].mean()
    m_p_src = df["psnr_srcnn"].mean()
    m_s_src = df["ssim_srcnn"].mean()

    table_data = [
        ["Method",  "Mean PSNR (dB)", "Mean SSIM"],
        ["Bicubic", f"{m_p_bic:.2f}",  f"{m_s_bic:.4f}"],
        ["SRCNN",   f"{m_p_src:.2f}",  f"{m_s_src:.4f}"],
    ]

    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis("off")
    tbl = ax.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.4, 2.0)
    for j in range(3):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    plt.suptitle("Bicubic vs SRCNN — Super-Resolution Metrics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    table_path = os.path.join(IMAGES_DIR, "srcnn_metrics_table.png")
    plt.savefig(table_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved {table_path}")

    print("\nAll done. SRCNN outputs saved to output/images/ and output/results/")
