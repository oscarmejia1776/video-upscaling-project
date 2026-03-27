# Mid-Evaluation Presentation Guide
**AI-Based Video Upscaling and Super-Resolution**
**Due: March 28, 2026 | ~5 minute video**

---

## Demo Checklist

Before recording, confirm you have the following ready to show:

- [ ] PowerPoint slides open and ready to present
- [ ] `data/videos/` folder open showing the 4 source MP4 files
- [ ] `output/frames/hr/` and `output/frames/lr/` showing extracted frame pairs
- [ ] `output/results/features.csv` open (terminal or spreadsheet)
- [ ] `03_train_models.py` open in editor — scroll to train/test split and classifier section
- [ ] `output/images/results_table.png` ready to show
- [ ] `output/images/comparison_01_frame_0000.png` (or any comparison image) ready to show
- [ ] `output/images/confusion_matrices.png` ready to show

---

## Slide-by-Slide Outline + Script

---

### Slide 1 — Title Slide

**Content:**
- Title: *AI-Based Video Upscaling and Super-Resolution*
- Subtitle: *Mid-Evaluation Baseline*
- Group members: Taylor O'Neal, Dipen Patel, Phillip Mejia
- Date: March 2026

**What to say:**
> "Hi, we're [names], and our project is called AI-Based Video Upscaling and Super-Resolution. Today we'll be walking through our mid-evaluation baseline, covering our project goal, the data and approach we used, and the results we got."

---

### Slide 2 — Project Goal

**Content:**
- Bullet: Low-resolution video is common in older recordings, compressed media, and surveillance footage
- Bullet: Traditional methods like bicubic upscaling increase pixel count but don't restore real detail
- Bullet: Goal: use machine learning to better understand and eventually improve how we reconstruct low-resolution video frames
- Optional: include one small side-by-side image (LR vs HR) to make it visual

**What to say:**
> "The problem we're addressing is low-resolution video. Whether it's old footage, heavily compressed streams, or surveillance video — the quality is often poor and hard to recover with traditional methods. Standard approaches like bicubic interpolation can resize an image, but they're essentially just guessing at the missing pixels using math. Our goal is to use machine learning to better characterize this problem and ultimately build a smarter solution. For the mid-evaluation, we're establishing a strong baseline to show what the problem looks like before we apply any deep learning."

---

### Slide 3 — Dataset

**Content:**
- Bullet: 4 short video clips sourced from Pexels.com (free, high-quality stock footage)
- Bullet: Resolutions: 1080p and 4K | Frame rates: 25fps, 30fps, 60fps
- Bullet: Total: 201 extracted frame pairs
- Bullet: Each frame pair = one original HR frame + one 4x downscaled LR version

**What to say:**
> "For our dataset, we used four short video clips downloaded from Pexels.com — a free stock video site. The videos range from about 12 to 30 seconds and include a mix of resolutions and frame rates. We extracted every 10th frame from each video, giving us 201 frame pairs total. For each pair, we have the original high-resolution frame and a version we artificially downscaled by a factor of 4 to simulate low-resolution input. This gives us the paired data we need for supervised learning."

---

### Slide 4 — Preprocessing & Feature Engineering

**Content:**
- Bullet: Frames extracted using OpenCV (`01_extract_frames.py`)
- Bullet: HR frames saved as-is; LR frames created by 4x bicubic downscale
- Bullet: For each frame pair, computed 7 image features:
  - Mean pixel value, Std deviation, Min/Max pixel value
  - Contrast, Sharpness (Laplacian variance), Gradient magnitude
- Bullet: Also computed bicubic reconstruction metrics: MSE, PSNR, SSIM
- Bullet: Label assigned: 0 = easy to reconstruct, 1 = hard to reconstruct (based on median MSE)

**What to say:**
> "Once we had the frames, we ran them through a feature extraction step. For each frame pair, we calculated seven simple image statistics — things like brightness, contrast, sharpness, and edge intensity. We also measured how well bicubic upscaling reconstructed the HR frame from the LR version using three standard metrics: MSE, PSNR, and SSIM. We then used the median MSE to assign a binary label to each frame — easy to reconstruct, or hard to reconstruct. This gave us a clean tabular dataset of 201 samples that we could plug straight into scikit-learn classifiers."

---

### Slide 5 — Approach: Classical ML Baseline

**Content:**
- Bullet: Problem framed as binary classification: predict whether a frame is easy or hard to reconstruct
- Bullet: 80/20 train/test split → 160 training samples, 41 test samples
- Bullet: Trained 6 classical ML classifiers from scikit-learn:
  - Logistic Regression, Decision Tree, Random Forest
  - SVC, Gaussian Naive Bayes, K-Nearest Neighbors
- Bullet: Evaluation metrics: Accuracy, Precision, Recall, F1 Score, MSE, MAE

**What to say:**
> "Our approach for the baseline is a standard classification workflow, similar to what we've covered in class. We framed the problem as: given these 7 image features, can a model predict whether bicubic upscaling will do a good or bad job on this frame? We split the 201 samples 80/20 into training and test sets, then trained six classifiers — Logistic Regression, Decision Tree, Random Forest, SVC, Gaussian Naive Bayes, and K-Nearest Neighbors. We evaluated each model using accuracy, precision, recall, and F1 score."

---

### Slide 6 — Results

**Content:**
- Insert screenshot of `output/images/results_table.png`
- Insert screenshot of `output/images/bicubic_metrics_table.png`
- Key callouts:
  - Bicubic baseline: PSNR = 34.51 dB, SSIM = 0.9502
  - Best classifiers (KNN, Decision Tree, Random Forest): ~97.6% accuracy, ~0.977 F1

**What to say:**
> "Here are our results. The bicubic upscaling baseline achieved a mean PSNR of 34.51 decibels and an SSIM of 0.95. PSNR above 30 is considered acceptable, but there's clear room for improvement — especially on frames with high motion or fine texture. On the classification side, three of our six models — K-Nearest Neighbors, Decision Tree, and Random Forest — all achieved around 97.6% accuracy and an F1 score of about 0.977. This tells us our 7 image features are actually very strong predictors of reconstruction difficulty, which is an interesting finding on its own."

---

### Slide 7 — Visualizations

**Content:**
- Insert one side-by-side comparison image from `output/images/comparison_01_frame_0000.png`
  - Label the three panels: Low-Resolution | Bicubic Upscaled | Original HR
- Insert `output/images/confusion_matrices.png` or `output/images/accuracy_comparison.png`

**What to say:**
> "Here you can see what the actual frame quality looks like visually. On the left is the low-resolution input, in the middle is the bicubic reconstruction, and on the right is the original high-resolution frame. You can see the bicubic version is smoother but loses fine detail compared to the original. The confusion matrices and accuracy chart show how cleanly most of our classifiers separated the two classes — the tree-based models and KNN had almost no misclassifications."

---

### Slide 8 — Next Steps (Final Project)

**Content:**
- Bullet: Train a CNN-based super-resolution model (e.g., SRCNN) to actually reconstruct HR frames
- Bullet: Compare deep learning output vs bicubic baseline using PSNR/SSIM
- Bullet: Explore parameter tuning, cross-validation, and feature selection
- Bullet: Possibly explore pre-trained models (Real-ESRGAN) for perceptual quality comparison

**What to say:**
> "For the final project, we'll go beyond classification and build a model that actually reconstructs the high-resolution frame from the low-resolution input. Our plan is to implement SRCNN — a simple convolutional neural network designed specifically for super-resolution — and compare its PSNR and SSIM scores directly against our bicubic baseline. We'll also look into parameter tuning and cross-validation on the classification side, and potentially bring in a pre-trained model like Real-ESRGAN for a visual quality comparison. The baseline we've built today gives us a clear benchmark to beat."

---

## Recording Tips

- Aim for **4:30 – 5:00 minutes** total. Practice once before recording.
- Screen record with slides open, then switch to your file explorer / code editor for the demo portion.
- For the demo, you don't need to run anything — just navigate to the folders and scroll through the files slowly.
- Suggested flow: Slides 1–5 (~2.5 min) → live demo of files/code (~1.5 min) → Slides 6–8 (~1 min)
- Only one group member needs to record and submit.
