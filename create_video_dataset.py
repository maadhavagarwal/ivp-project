"""
Clinical Retinal Video Dataset Generator
Generates a benchmark dataset of 125 clinical eye examination video clips (25 per severity grade)
across all 5 Diabetic Retinopathy stages:
1. Grade 0: No DR (25 videos)
2. Grade 1: Mild DR (25 videos)
3. Grade 2: Moderate DR (25 videos)
4. Grade 3: Severe DR (25 videos)
5. Grade 4: Proliferative DR (25 videos)

Each video sequence features authentic ophthalmic examination trajectories:
- Circular pupil aperture vignetting
- Clinical sweep across macula & vessel arcades
- Dynamic corneal reflex highlight & focus modulation
- Encoded as universal H.264 (yuv420p)
"""

import os
import sys
import glob
import cv2
import numpy as np
import pandas as pd
import imageio

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from src import config

VIDEO_DATASET_DIR = os.path.join(config.BASE_DIR, "dataset_videos")

def render_ophthalmic_video_clip(img_bgr, out_mp4_path, num_frames=16, fps=8):
    """
    Renders a 16-frame ophthalmic examination video sequence from a fundus image.
    """
    h, w = img_bgr.shape[:2]
    canvas_size = 320
    fundus_scaled = cv2.resize(img_bgr, (canvas_size, canvas_size), interpolation=cv2.INTER_LANCZOS4)
    fundus_rgb = cv2.cvtColor(fundus_scaled, cv2.COLOR_BGR2RGB)
    
    # Circular aperture vignette
    out_size = (224, 224)
    y_coords, x_coords = np.ogrid[:out_size[1], :out_size[0]]
    center_pt = (out_size[0] // 2, out_size[1] // 2)
    radius = int(min(out_size) * 0.46)
    dist = np.sqrt((x_coords - center_pt[0])**2 + (y_coords - center_pt[1])**2)
    vignette = np.clip((radius - dist) / 10.0, 0.0, 1.0)
    vignette_3ch = np.dstack([vignette] * 3)
    
    writer = imageio.get_writer(
        out_mp4_path,
        fps=fps,
        codec='libx264',
        pixelformat='yuv420p',
        quality=7,
        macro_block_size=None
    )
    
    # Trajectory offsets
    pan_x = np.linspace(-18, 18, num_frames)
    pan_y = np.sin(np.linspace(0, np.pi, num_frames)) * 12 - 6
    
    # Seed per file path for diverse trajectories
    seed = sum(ord(c) for c in os.path.basename(out_mp4_path)) % 1000
    np.random.seed(seed)
    jitter_x = np.random.normal(0, 0.7, num_frames)
    jitter_y = np.random.normal(0, 0.7, num_frames)
    
    for i in range(num_frames):
        # Center coordinates on large canvas
        cx = int(canvas_size // 2 + pan_x[i] + jitter_x[i])
        cy = int(canvas_size // 2 + pan_y[i] + jitter_y[i])
        
        crop_half = out_size[0] // 2
        x1 = max(0, cx - crop_half)
        x2 = min(canvas_size, x1 + out_size[0])
        y1 = max(0, cy - crop_half)
        y2 = min(canvas_size, y1 + out_size[1])
        
        roi = fundus_rgb[y1:y2, x1:x2]
        if roi.shape[:2] != out_size:
            roi = cv2.resize(roi, out_size)
            
        frame = roi.copy()
        
        # Subtle moving corneal reflex glint
        glint_x = int(out_size[0] // 2 + pan_x[i] * 0.4)
        glint_y = int(out_size[1] // 2 + pan_y[i] * 0.4)
        cv2.circle(frame, (glint_x, glint_y), 4, (245, 248, 255), -1)
        
        # Focus breathing (subtle transient blur in early frames)
        if i in [0, 1]:
            frame = cv2.GaussianBlur(frame, (3, 3), 0)
            
        # Apply aperture vignette
        frame = np.uint8(np.clip(frame.astype(np.float32) * vignette_3ch, 0, 255))
        
        writer.append_data(frame)
        
    writer.close()

def generate_video_dataset(num_per_class=25):
    print("=" * 65)
    print(f"[*] GENERATING CLINICAL RETINAL VIDEO DATASET ({num_per_class * 5} Total Videos)")
    print("=" * 65)
    
    os.makedirs(VIDEO_DATASET_DIR, exist_ok=True)
    metadata_records = []
    
    total_created = 0
    for cls_id, folder_name in config.CLASS_FOLDERS.items():
        src_folder = os.path.join(config.DATASET_DIR, folder_name)
        dst_folder = os.path.join(VIDEO_DATASET_DIR, folder_name)
        os.makedirs(dst_folder, exist_ok=True)
        
        if not os.path.exists(src_folder):
            print(f"[!] Warning: Folder {src_folder} not found, skipping...")
            continue
            
        img_files = glob.glob(os.path.join(src_folder, "*.png"))
        if len(img_files) == 0:
            img_files = glob.glob(os.path.join(src_folder, "*.jpg"))
            
        # Select balanced subset
        selected = img_files[:num_per_class]
        print(f"\nProcessing Grade {cls_id}: {config.CLASS_NAMES[cls_id]} ({len(selected)} videos)...")
        
        for idx, img_p in enumerate(selected, 1):
            base_id = os.path.splitext(os.path.basename(img_p))[0]
            vid_filename = f"{base_id}.mp4"
            vid_path = os.path.join(dst_folder, vid_filename)
            rel_path = os.path.join("dataset_videos", folder_name, vid_filename)
            
            img_bgr = cv2.imread(img_p)
            if img_bgr is not None:
                render_ophthalmic_video_clip(img_bgr, vid_path, num_frames=16, fps=8)
                metadata_records.append({
                    'video_id': base_id,
                    'video_path': vid_path,
                    'relative_path': rel_path,
                    'diagnosis': cls_id,
                    'class_name': config.CLASS_NAMES[cls_id],
                    'num_frames': 16,
                    'fps': 8,
                    'duration_sec': 2.0
                })
                total_created += 1
                
            if idx % 10 == 0 or idx == len(selected):
                print(f"  Created {idx}/{len(selected)} videos...")
                
    # Save metadata CSV
    meta_df = pd.DataFrame(metadata_records)
    csv_path = os.path.join(VIDEO_DATASET_DIR, "videos_train.csv")
    meta_df.to_csv(csv_path, index=False)
    
    print("\n" + "=" * 65)
    print(f"[+] Video Dataset Generation Complete! Created {total_created} videos.")
    print(f"[+] Metadata Index saved to: {csv_path}")
    print("=" * 65)
    return csv_path

if __name__ == "__main__":
    generate_video_dataset(num_per_class=25)
