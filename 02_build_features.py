import cv2
import numpy as np
import pandas as pd
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Paths
HR_DIR      = "output/frames/hr"
LR_DIR      = "output/frames/lr"
RESULTS_DIR = "output/results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def compute_features(lr_frame_gray, hr_frame_gray, bicubic_frame_gray):

    # Basic pixel statistics on the LR frame
    mean_pixel  = float(np.mean(lr_frame_gray))
    std_pixel   = float(np.std(lr_frame_gray))
    min_pixel   = float(np.min(lr_frame_gray))
    max_pixel   = float(np.max(lr_frame_gray))
    contrast    = max_pixel - min_pixel

    # Sharpness: Laplacian variance (higher = sharper)
    laplacian   = cv2.Laplacian(lr_frame_gray, cv2.CV_64F)
    sharpness   = float(laplacian.var())

    # Edge energy: mean gradient magnitude (Sobel)
    sobel_x     = cv2.Sobel(lr_frame_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y     = cv2.Sobel(lr_frame_gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient    = float(np.mean(np.sqrt(sobel_x**2 + sobel_y**2)))

    # Reconstruction quality: comparing bicubic vs original HR
    mse_val     = float(np.mean((bicubic_frame_gray.astype(float) - hr_frame_gray.astype(float))**2))

    # PSNR — standard image quality metric (higher is better)
    psnr_val    = float(psnr(hr_frame_gray, bicubic_frame_gray, data_range=255))

    # SSIM — structural similarity (closer to 1.0 is better)
    ssim_val    = float(ssim(hr_frame_gray, bicubic_frame_gray, data_range=255))

    return {
        "mean_pixel" : mean_pixel,
        "std_pixel"  : std_pixel,
        "min_pixel"  : min_pixel,
        "max_pixel"  : max_pixel,
        "contrast"   : contrast,
        "sharpness"  : sharpness,
        "gradient"   : gradient,
        "mse"        : mse_val,
        "psnr"       : psnr_val,
        "ssim"       : ssim_val,
    }


# Main — loop over all frame pairs
if __name__ == "__main__":
    hr_files = sorted(os.listdir(HR_DIR))
    lr_files = sorted(os.listdir(LR_DIR))

    if not hr_files:
        print("No frames found. Run 01_extract_frames.py first.")
        exit()

    print(f"Building features for {len(hr_files)} frame pairs...")

    rows = []
    for filename in hr_files:
        hr_path = os.path.join(HR_DIR, filename)
        lr_path = os.path.join(LR_DIR, filename)

        if not os.path.exists(lr_path):
            print(f"  [SKIP] No matching LR frame for {filename}")
            continue

        # Load frames in grayscale (simpler features, smaller data)
        hr_frame = cv2.imread(hr_path, cv2.IMREAD_GRAYSCALE)
        lr_small = cv2.imread(lr_path, cv2.IMREAD_GRAYSCALE)

        if hr_frame is None or lr_small is None:
            print(f"  [SKIP] Could not read {filename}")
            continue

        # Bicubic upscale: resize LR back to HR dimensions
        h, w = hr_frame.shape
        bicubic_upscaled = cv2.resize(lr_small, (w, h), interpolation=cv2.INTER_CUBIC)

        # Compute all features and quality metrics
        feat = compute_features(lr_small, hr_frame, bicubic_upscaled)
        feat["filename"] = filename
        rows.append(feat)

    # Build DataFrame 
    df = pd.DataFrame(rows)

    #  Create binary quality label based on median MSE
    # Frames with MSE above the median are "harder" to reconstruct
    median_mse = df["mse"].median()
    df["quality_label"] = (df["mse"] > median_mse).astype(int)

    print(f"\nMedian bicubic MSE: {median_mse:.4f}")
    print(f"  Label 0 (easy to reconstruct): {(df['quality_label'] == 0).sum()} frames")
    print(f"  Label 1 (hard to reconstruct): {(df['quality_label'] == 1).sum()} frames")

    # Save the dataset
    out_path = os.path.join(RESULTS_DIR, "features.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved features to: {out_path}")
    print(f"Columns: {list(df.columns)}")

    # Print a quick summary of bicubic baseline metrics
    print("\n── Bicubic Upscaling Baseline (Classical Method) ──")
    print(f"  Mean MSE  : {df['mse'].mean():.4f}")
    print(f"  Mean PSNR : {df['psnr'].mean():.2f} dB")
    print(f"  Mean SSIM : {df['ssim'].mean():.4f}")
