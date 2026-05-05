import cv2
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paths
HR_DIR       = "output/frames/hr"
LR_DIR       = "output/frames/lr"
IMAGES_DIR   = "output/images"
RESULTS_DIR  = "output/results"
os.makedirs(IMAGES_DIR, exist_ok=True)

NUM_COMPARISONS = 5   # how many side-by-side images to save

# Pick a few evenly-spaced frames to showcase
hr_files = sorted(os.listdir(HR_DIR))

if not hr_files:
    print("No HR frames found. Run 01_extract_frames.py first.")
    exit()

# Choose frames spread across the whole dataset
indices = np.linspace(0, len(hr_files) - 1, NUM_COMPARISONS, dtype=int)
selected_files = [hr_files[i] for i in indices]

print(f"Saving {NUM_COMPARISONS} side-by-side comparison images...")

for i, filename in enumerate(selected_files):
    hr_path = os.path.join(HR_DIR, filename)
    lr_path = os.path.join(LR_DIR, filename)

    hr_bgr = cv2.imread(hr_path)
    lr_bgr = cv2.imread(lr_path)

    if hr_bgr is None or lr_bgr is None:
        print(f"  [SKIP] Could not read {filename}")
        continue

    # Bicubic upscale LR → HR size
    h, w = hr_bgr.shape[:2]
    bicubic = cv2.resize(lr_bgr, (w, h), interpolation=cv2.INTER_CUBIC)

    # Also create a "zoomed" LR at HR size just for display (nearest neighbor = pixelated look)
    lr_display = cv2.resize(lr_bgr, (w, h), interpolation=cv2.INTER_NEAREST)

    # Convert BGR → RGB for matplotlib
    lr_rgb      = cv2.cvtColor(lr_display, cv2.COLOR_BGR2RGB)
    bicubic_rgb = cv2.cvtColor(bicubic,    cv2.COLOR_BGR2RGB)
    hr_rgb      = cv2.cvtColor(hr_bgr,     cv2.COLOR_BGR2RGB)

    # Crop a zoomed patch from the center-ish of the frame to show detail difference
    # We crop from the HR/bicubic images (same size) and the LR display
    crop_h = h // 4
    crop_w = w // 4
    cy = h // 3 
    cx = w // 2
    y1, y2 = cy - crop_h // 2, cy + crop_h // 2
    x1, x2 = cx - crop_w // 2, cx + crop_w // 2

    lr_crop      = lr_rgb[y1:y2,      x1:x2]
    bicubic_crop = bicubic_rgb[y1:y2,  x1:x2]
    hr_crop      = hr_rgb[y1:y2,       x1:x2]

    # Build the figure — top row: full frames, bottom row: zoomed crops
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"Frame Comparison: {filename}", fontsize=13, fontweight="bold")

    # Top row — full frames
    axes[0][0].imshow(lr_rgb);      axes[0][0].set_title("Low-Resolution Input",    fontsize=11)
    axes[0][1].imshow(bicubic_rgb); axes[0][1].set_title("Bicubic Upscaled",         fontsize=11)
    axes[0][2].imshow(hr_rgb);      axes[0][2].set_title("High-Resolution Original", fontsize=11)

    # Bottom row — zoomed crops (detail comparison)
    axes[1][0].imshow(lr_crop);      axes[1][0].set_title("LR — Zoomed Crop",       fontsize=11)
    axes[1][1].imshow(bicubic_crop); axes[1][1].set_title("Bicubic — Zoomed Crop",  fontsize=11)
    axes[1][2].imshow(hr_crop);      axes[1][2].set_title("HR — Zoomed Crop",        fontsize=11)

    for row in axes:
        for ax in row:
            ax.axis("off")

    plt.tight_layout()
    save_path = os.path.join(IMAGES_DIR, f"comparison_{i+1:02d}_{filename.replace('.png','')}.png")
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  Saved: {save_path}")

# Save a results summary table as an image
# (handy to drop straight into your slides)
results_path = os.path.join(RESULTS_DIR, "classifier_results.csv")
if os.path.exists(results_path):
    results_df = pd.read_csv(results_path)

    fig, ax = plt.subplots(figsize=(11, 3))
    ax.axis("off")

    col_labels = list(results_df.columns)
    cell_text  = results_df.values.tolist()

    table = ax.table(
        cellText  = cell_text,
        colLabels = col_labels,
        cellLoc   = "center",
        loc       = "center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)

    # Shade the header row
    for col_idx in range(len(col_labels)):
        table[0, col_idx].set_facecolor("#4472C4")
        table[0, col_idx].set_text_props(color="white", fontweight="bold")

    plt.title("Classifier Results Summary", fontsize=12, pad=15)
    plt.tight_layout()
    table_path = os.path.join(IMAGES_DIR, "results_table.png")
    plt.savefig(table_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nResults table image saved to: {table_path}")
else:
    print("\nNo classifier_results.csv found — run 03_train_models.py first.")

# Save bicubic metrics table as image
bicubic_path = os.path.join(RESULTS_DIR, "bicubic_metrics.csv")
if os.path.exists(bicubic_path):
    bdf = pd.read_csv(bicubic_path)

    fig, ax = plt.subplots(figsize=(6, 1.5))
    ax.axis("off")
    table2 = ax.table(
        cellText  = bdf.values.tolist(),
        colLabels = list(bdf.columns),
        cellLoc   = "center",
        loc       = "center",
    )
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1.2, 2.0)
    for col_idx in range(len(bdf.columns)):
        table2[0, col_idx].set_facecolor("#70AD47")
        table2[0, col_idx].set_text_props(color="white", fontweight="bold")

    plt.title("Bicubic Upscaling Baseline Metrics", fontsize=11, pad=15)
    plt.tight_layout()
    bm_path = os.path.join(IMAGES_DIR, "bicubic_metrics_table.png")
    plt.savefig(bm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Bicubic metrics table saved to: {bm_path}")

print("\nVisualization complete! Check output/images/ for all saved figures.")
