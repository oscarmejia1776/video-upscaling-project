# AI-Based Video Upscaling — Final Project

**Group:** Taylor O'Neal · Dipen Patel · Phillip Mejia

---

## Project Goal

Build a machine learning pipeline for video super-resolution. We downscale high-quality video frames to simulate low-resolution input, train classical ML classifiers to characterize frame reconstruction difficulty, and implement SRCNN — a convolutional neural network — to reconstruct high-resolution frames. A final script applies the trained model to upscale any input video.

---

## Setup

### 1. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Dataset
Four short video clips (12–30 seconds each) sourced from [Pexels.com](https://www.pexels.com/videos/) are included in `data/videos/`. Videos were selected at 1080p and 4K resolutions across 25fps, 30fps, and 60fps to provide varied frame data.

---

## How to Run

### Full pipeline (all 6 steps)

```bash
./run_all.sh
```

This executes the following six scripts incrementally, in order, stopping if any step fails:

| Script | What it does |
|--------|-------------|
| `01_extract_frames.py` | Reads each video, extracts every 10th frame, saves HR originals and 4x downscaled LR versions as paired PNG files |
| `02_build_features.py` | Computes image features (brightness, sharpness, contrast, edge detail) for each frame pair, measures bicubic MSE/PSNR/SSIM, and saves a labeled CSV dataset |
| `03_train_models.py` | Loads the CSV, splits 80/20 train/test, trains 6 classical ML classifiers, and prints + saves a full results table |
| `04_visualize.py` | Generates side-by-side frame comparison images, confusion matrices, accuracy bar chart, and table figures for slides |
| `05_train_srcnn.py` | Trains a 3-layer SRCNN on Y-channel patches extracted from the training frames, evaluates PSNR/SSIM vs bicubic on test frames, and saves the model |
| `06_visualize_srcnn.py` | Loads the trained SRCNN, generates 4-panel comparison images (LR / Bicubic / SRCNN / HR), and saves a summary metrics table |

### Upscale a video with the trained SRCNN

Once `05_train_srcnn.py` has been run (model saved to `output/models/srcnn.pth`), you can upscale any video:

```bash
# Upscale the first video found in data/videos/ (default)
python 07_upscale_video.py

# Upscale a specific video file
python 07_upscale_video.py data/videos/15506611_1920_1080_25fps.mp4

# Upscale any other MP4 (e.g. a short 480p test clip)
python 07_upscale_video.py path/to/your_clip.mp4
```

Output is saved to `output/upscaled/<filename>_srcnn.mp4` at 4× the input resolution.

> **Tip:** For a quick test on CPU, use a short (5–10 second) 480p clip. A 480p clip runs roughly 5× faster per frame than a 1080p clip. A 5-second clip at 30fps (~150 frames) finishes in about 4–8 minutes on CPU.

---

## Output Files

| File | Description |
|------|-------------|
| `output/results/features.csv` | Feature dataset with labels |
| `output/results/classifier_results.csv` | Accuracy / F1 / Precision / Recall for all 6 models |
| `output/results/bicubic_metrics.csv` | MSE / PSNR / SSIM for bicubic upscaling |
| `output/results/srcnn_metrics.csv` | Per-frame PSNR / SSIM comparison: bicubic vs SRCNN |
| `output/models/srcnn.pth` | Trained SRCNN model weights |
| `output/images/comparison_*.png` | Side-by-side LR vs Bicubic vs HR frame comparisons |
| `output/images/srcnn_comparison_*.png` | 4-panel LR vs Bicubic vs SRCNN vs HR comparisons |
| `output/images/confusion_matrices.png` | Confusion matrices for all 6 classifiers |
| `output/images/accuracy_comparison.png` | Bar chart comparing model accuracy |
| `output/images/results_table.png` | Classifier results table image |
| `output/images/bicubic_metrics_table.png` | Bicubic metrics table image |
| `output/images/srcnn_metrics_table.png` | Bicubic vs SRCNN metrics table image |
| `output/upscaled/*_srcnn.mp4` | Upscaled video output from `07_upscale_video.py` |

---

## Project Structure

```
video_upscaling_project/
├── data/
│   └── videos/              ← source .mp4 files (from Pexels.com)
├── output/
│   ├── frames/
│   │   ├── hr/              ← high-resolution extracted frames
│   │   └── lr/              ← low-resolution (downscaled 4x) frames
│   ├── images/              ← comparison plots and table images
│   ├── models/              ← trained SRCNN weights (srcnn.pth)
│   ├── results/             ← CSV result tables
│   └── upscaled/            ← upscaled video output
├── 01_extract_frames.py
├── 02_build_features.py
├── 03_train_models.py
├── 04_visualize.py
├── 05_train_srcnn.py
├── 06_visualize_srcnn.py
├── 07_upscale_video.py
├── run_all.sh
├── requirements.txt
└── README.md
```
