#!/bin/bash
# Run the full video upscaling baseline pipeline in order.
# Each step only runs if the previous one succeeded.

echo "========================================="
echo " Video Upscaling — Baseline Pipeline"
echo "========================================="

echo ""
echo "[1/6] Extracting frames from videos..."
python 01_extract_frames.py || { echo "ERROR in step 1, stopping."; exit 1; }

echo ""
echo "[2/6] Building feature dataset..."
python 02_build_features.py || { echo "ERROR in step 2, stopping."; exit 1; }

echo ""
echo "[3/6] Training classifiers..."
python 03_train_models.py || { echo "ERROR in step 3, stopping."; exit 1; }

echo ""
echo "[4/6] Generating visualizations..."
python 04_visualize.py || { echo "ERROR in step 4, stopping."; exit 1; }

echo ""
echo "[5/6] Training SRCNN..."
python 05_train_srcnn.py || { echo "ERROR in step 5, stopping."; exit 1; }

echo ""
echo "[6/6] Visualizing SRCNN results..."
python 06_visualize_srcnn.py || { echo "ERROR in step 6, stopping."; exit 1; }

echo ""
echo "========================================="
echo " All done! Check output/ for results."
echo "========================================="
