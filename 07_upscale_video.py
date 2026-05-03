"""
07_upscale_video.py
-------------------
Apply the trained SRCNN model to every frame of a video and write a 4x
upscaled MP4 output.

Usage:
    python 07_upscale_video.py                        # uses first video in data/videos/
    python 07_upscale_video.py path/to/video.mp4      # specific input file

Output:
    output/upscaled/<input_stem>_srcnn.mp4

NOTE: Processing runs on CPU and takes roughly 1-2 seconds per frame.
A 30-second clip at 30 fps (~900 frames) will take 15-30 minutes.
"""

import cv2
import glob
import numpy as np
import os
import sys
import time
import torch
import torch.nn as nn

MODEL_PATH  = "output/models/srcnn.pth"
OUTPUT_DIR  = "output/upscaled"
SCALE       = 4

os.makedirs(OUTPUT_DIR, exist_ok=True)


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


def upscale_frame(bgr_frame, model, device):
    """
    4x upscale a single BGR frame using bicubic + SRCNN Y-channel refinement.
    Returns a BGR uint8 image at 4x the input resolution.
    """
    h, w = bgr_frame.shape[:2]
    out_h, out_w = h * SCALE, w * SCALE

    # Bicubic upscale to target size
    bicubic = cv2.resize(bgr_frame, (out_w, out_h), interpolation=cv2.INTER_CUBIC)

    # Convert to YCrCb and extract Y channel
    ycrcb   = cv2.cvtColor(bicubic, cv2.COLOR_BGR2YCrCb)
    y_float = ycrcb[:, :, 0].astype(np.float32) / 255.0

    # Run SRCNN on Y channel
    with torch.no_grad():
        t   = torch.from_numpy(y_float[np.newaxis, np.newaxis]).to(device)
        out = model(t).squeeze().cpu().numpy()
    srcnn_y = np.clip(out, 0.0, 1.0)

    # Merge SRCNN Y with bicubic Cb/Cr and convert back to BGR
    result          = ycrcb.copy()
    result[:, :, 0] = (srcnn_y * 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_YCrCb2BGR)


if __name__ == "__main__":
    # ── Resolve input video ──
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        candidates = sorted(glob.glob("data/videos/*.mp4"))
        if not candidates:
            print("No .mp4 files found in data/videos/. Pass a video path as an argument.")
            sys.exit(1)
        input_path = candidates[0]

    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        sys.exit(1)

    stem        = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(OUTPUT_DIR, f"{stem}_srcnn.mp4")

    # ── Load video metadata ──
    cap       = cv2.VideoCapture(input_path)
    fps       = cap.get(cv2.CAP_PROP_FPS)
    in_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_w, out_h = in_w * SCALE, in_h * SCALE

    print(f"Input : {input_path}")
    print(f"        {in_w}x{in_h} @ {fps:.1f} fps — {n_frames} frames")
    print(f"Output: {output_path}")
    print(f"        {out_w}x{out_h} @ {fps:.1f} fps")
    est_minutes = n_frames * 1.5 / 60
    print(f"\nEstimated time on CPU: ~{est_minutes:.0f} minutes ({n_frames} frames @ ~1.5 s/frame)")
    print("Processing...\n")

    # ── Load model ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SRCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    # ── Open writer ──
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    # ── Process frames ──
    start     = time.time()
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        writer.write(upscale_frame(frame, model, device))
        processed += 1

        if processed % 50 == 0:
            elapsed  = time.time() - start
            per_frame = elapsed / processed
            remaining = (n_frames - processed) * per_frame
            print(f"  Frame {processed}/{n_frames} — "
                  f"{per_frame:.2f} s/frame — "
                  f"~{remaining/60:.1f} min remaining")

    cap.release()
    writer.release()

    elapsed = time.time() - start
    print(f"\nDone! Processed {processed} frames in {elapsed/60:.1f} minutes.")
    print(f"Output saved to: {output_path}")
