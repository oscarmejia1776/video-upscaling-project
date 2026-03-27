"""
01_extract_frames.py
--------------------
Step 1: Load video files and extract frames.

We extract every Nth frame from each video to keep the dataset manageable.
Each extracted frame is saved as a high-resolution (HR) image.
We then downscale each HR frame by a factor of 4 to create a
low-resolution (LR) version, simulating a real-world super-resolution scenario.

How to run:
    python 01_extract_frames.py

Place your .mp4 video files in the data/videos/ folder before running.
"""

import cv2
import os

# ─────────────────────────────────────────────
# Configuration — adjust these if needed
# ─────────────────────────────────────────────
VIDEO_DIR   = "data/videos"          # folder with your .mp4 files
HR_DIR      = "output/frames/hr"    # where high-res frames will be saved
LR_DIR      = "output/frames/lr"    # where low-res frames will be saved
FRAME_STEP  = 10                     # extract every 10th frame
SCALE       = 4                      # downscale factor for creating LR frames
MAX_FRAMES  = 500                    # cap total frames to keep things fast

# ─────────────────────────────────────────────
# Make sure output directories exist
# ─────────────────────────────────────────────
os.makedirs(HR_DIR, exist_ok=True)
os.makedirs(LR_DIR, exist_ok=True)

def extract_frames(video_path, start_index=0):
    """
    Extract frames from a single video file.
    Returns the index after the last saved frame.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"  [ERROR] Could not open video: {video_path}")
        return start_index

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps                = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Total frames: {total_video_frames}  |  FPS: {fps:.1f}")

    frame_count  = 0   # counts every frame we read
    saved_count  = 0   # counts frames we actually save
    global_index = start_index

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # end of video

        # Only save every Nth frame
        if frame_count % FRAME_STEP == 0:
            # Build file name like frame_0000.png
            filename = f"frame_{global_index:04d}.png"

            # Save high-resolution frame
            cv2.imwrite(os.path.join(HR_DIR, filename), frame)

            # Create low-resolution version by downscaling then upscaling back
            # (we keep the same spatial dimensions so HR/LR pairs are aligned)
            h, w = frame.shape[:2]
            small_w, small_h = w // SCALE, h // SCALE

            lr_small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_CUBIC)
            # Save the actual small version (used later for feature extraction)
            cv2.imwrite(os.path.join(LR_DIR, filename), lr_small)

            global_index += 1
            saved_count  += 1

            # Stop if we've hit the frame cap
            if (global_index - start_index) >= MAX_FRAMES:
                print(f"  Reached MAX_FRAMES cap ({MAX_FRAMES}), stopping early.")
                break

        frame_count += 1

    cap.release()
    print(f"  Saved {saved_count} frame pairs from {os.path.basename(video_path)}")
    return global_index


# ─────────────────────────────────────────────
# Main — process all videos in VIDEO_DIR
# ─────────────────────────────────────────────
if __name__ == "__main__":
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]

    if not video_files:
        print(f"No .mp4 files found in '{VIDEO_DIR}'. Please add a video and re-run.")
    else:
        print(f"Found {len(video_files)} video(s) in '{VIDEO_DIR}'")
        frame_index = 0
        for vf in video_files:
            path = os.path.join(VIDEO_DIR, vf)
            print(f"\nProcessing: {vf}")
            frame_index = extract_frames(path, start_index=frame_index)

        print(f"\nDone! Total frame pairs saved: {frame_index}")
        print(f"  HR frames → {HR_DIR}")
        print(f"  LR frames → {LR_DIR}")
