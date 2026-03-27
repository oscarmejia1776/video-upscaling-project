# AI-Based Video Upscaling — Mid-Evaluation Baseline

**Group:** Taylor O'Neal · Dipen Patel · Phillip Mejia

---

## Project Goal

Build a baseline pipeline for video super-resolution. We downscale high-quality video frames to simulate low-resolution input, then evaluate how well classical ML models can characterize frame reconstruction difficulty — motivating the need for deep learning approaches in the final project.

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

Run the full pipeline with a single command:

```bash
./run_all.sh
```

This executes the following four scripts incrementally, in order, stopping if any step fails:

| Script | What it does |
|--------|-------------|
| `01_extract_frames.py` | Reads each video, extracts every 10th frame, saves HR originals and 4x downscaled LR versions as paired PNG files |
| `02_build_features.py` | Computes image features (brightness, sharpness, contrast, edge detail) for each frame pair, measures bicubic MSE/PSNR/SSIM, and saves a labeled CSV dataset |
| `03_train_models.py` | Loads the CSV, splits 80/20 train/test, trains 6 classical ML classifiers, and prints + saves a full results table |
| `04_visualize.py` | Generates side-by-side frame comparison images, confusion matrices, accuracy bar chart, and table figures for slides |

---

## Output Files

| File | Description |
|------|-------------|
| `output/results/features.csv` | Feature dataset with labels |
| `output/results/classifier_results.csv` | Accuracy / F1 / Precision / Recall for all 6 models |
| `output/results/bicubic_metrics.csv` | MSE / PSNR / SSIM for bicubic upscaling |
| `output/images/comparison_*.png` | Side-by-side LR vs Bicubic vs HR frame comparisons |
| `output/images/confusion_matrices.png` | Confusion matrices for all 6 classifiers |
| `output/images/accuracy_comparison.png` | Bar chart comparing model accuracy |
| `output/images/results_table.png` | Clean results table image for slides |
| `output/images/bicubic_metrics_table.png` | Bicubic metrics table image for slides |

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
│   └── results/             ← CSV result tables
├── 01_extract_frames.py
├── 02_build_features.py
├── 03_train_models.py
├── 04_visualize.py
├── run_all.sh
├── requirements.txt
└── README.md
```
