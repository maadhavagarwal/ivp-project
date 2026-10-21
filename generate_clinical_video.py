"""
Realistic Clinical Ophthalmoscope Video Generator
Generates a 15-second (375 frames @ 25 FPS) authentic ophthalmic fundus examination video:
1. Pupil acquisition & illumination ramp-up
2. Optic disc centering & fine focus adjustment
3. Smooth clinical pan to the Macula & Fovea
4. Arcade sweeps across vascular quadrants (lesion inspection)
5. Physiological micro-saccades, subtle corneal reflex glint, and natural blink artifact
6. Encoded with standard H.264 (yuv420p) for universal browser & player playback
"""

import os
import sys
import cv2
import numpy as np
import imageio

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from src import config

def create_clinical_eye_video(output_path="sample_eye_scan.mp4", duration_sec=15, fps=25):
    print("=" * 65)
    print("[*] GENERATING REALISTIC CLINICAL OPHTHALMOSCOPE EYE SCAN VIDEO")
    print("=" * 65)
    
    total_frames = int(duration_sec * fps)
    vid_w, vid_h = 640, 480
    
    # Select fundus image with clear pathology (Moderate DR)
    img_path = os.path.join(config.DATASET_DIR, "Moderate", "000c1434d8d7.png")
    if not os.path.exists(img_path):
        img_path = os.path.join(config.DATASET_DIR, "Mild", "0024cdab0c1e.png")
    
    fundus_bgr = cv2.imread(img_path)
    if fundus_bgr is None:
        raise FileNotFoundError(f"Could not load source image from {img_path}")
        
    # High-resolution super-sampled canvas for smooth subpixel interpolation
    canvas_size = 900
    fundus_large = cv2.resize(fundus_bgr, (canvas_size, canvas_size), interpolation=cv2.INTER_LANCZOS4)
    fundus_rgb = cv2.cvtColor(fundus_large, cv2.COLOR_BGR2RGB)
    
    # Precompute aperture mask (circular field of view with soft optical vignette)
    aperture_radius = int(min(vid_w, vid_h) * 0.44)
    aperture_center = (vid_w // 2, vid_h // 2)
    
    y_coords, x_coords = np.ogrid[:vid_h, :vid_w]
    dist_from_center = np.sqrt((x_coords - aperture_center[0])**2 + (y_coords - aperture_center[1])**2)
    
    vignette = np.clip((aperture_radius - dist_from_center) / 18.0, 0.0, 1.0)
    vignette_3ch = np.dstack([vignette] * 3)
    
    # Setup H.264 Video Writer via imageio-ffmpeg
    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec='libx264',
        pixelformat='yuv420p',
        quality=8,
        macro_block_size=None
    )
    
    # Physiological micro-tremor & drift trajectory
    np.random.seed(42)
    tremor_x = np.cumsum(np.random.normal(0, 0.4, total_frames))
    tremor_y = np.cumsum(np.random.normal(0, 0.4, total_frames))
    tremor_x = np.convolve(tremor_x, np.ones(15)/15, mode='same')
    tremor_y = np.convolve(tremor_y, np.ones(15)/15, mode='same')
    
    # Keyframe Examination Waypoints (target X offset, target Y offset, zoom level, focus blur, brightness)
    # Native center = (450, 450)
    def get_trajectory_params(t_norm):
        # t_norm in [0.0 .. 1.0]
        if t_norm < 0.15:
            # Phase 1: Entry & Pupil Alignment (0 - 2.25s)
            p = t_norm / 0.15
            dx = 40.0 * (1.0 - p)
            dy = -30.0 * (1.0 - p)
            zoom = 1.1 + 0.15 * p
            blur = max(0.0, (1.0 - p) * 7.0)
            brightness = 0.2 + 0.8 * p
        elif t_norm < 0.38:
            # Phase 2: Optic Disc Centering & Focus Wheel Adjustment (2.25s - 5.7s)
            p = (t_norm - 0.15) / 0.23
            dx = -45.0 * np.sin(p * np.pi)
            dy = -10.0 * np.sin(p * np.pi)
            zoom = 1.25 + 0.05 * np.sin(p * np.pi)
            # Transient focus wheel adjustment
            blur = 2.5 * np.sin(p * np.pi * 2) if 0.3 < p < 0.7 else 0.0
            brightness = 1.0
        elif t_norm < 0.62:
            # Phase 3: Smooth Pan to Macula & Fovea (5.7s - 9.3s)
            p = (t_norm - 0.38) / 0.24
            # Smooth ease-in-out cosine interpolation
            ease = 0.5 - 0.5 * np.cos(p * np.pi)
            dx = -25.0 + ease * 60.0
            dy = -10.0 + ease * 15.0
            zoom = 1.25 + 0.1 * ease
            blur = 0.0
            brightness = 1.0
        elif t_norm < 0.82:
            # Phase 4: Vascular Arcade Sweeps (9.3s - 12.3s)
            p = (t_norm - 0.62) / 0.20
            sweep_angle = p * 2.0 * np.pi
            dx = 35.0 + 30.0 * np.cos(sweep_angle)
            dy = 5.0 + 25.0 * np.sin(sweep_angle)
            zoom = 1.30
            blur = 0.0
            brightness = 1.0
        elif t_norm < 0.90:
            # Phase 5a: Saccade & Patient Blink (12.3s - 13.5s)
            p = (t_norm - 0.82) / 0.08
            dx = 15.0 * (1.0 - p)
            dy = 0.0
            zoom = 1.25
            # Blink darkness profile between frame 310 and 325
            blink = max(0.0, 1.0 - 4.0 * abs(p - 0.5))
            brightness = 1.0 - 0.92 * blink
            blur = 4.0 * blink
        else:
            # Phase 5b: Final Centered Diagnostic Inspection (13.5s - 15.0s)
            p = (t_norm - 0.90) / 0.10
            dx = 0.0
            dy = 0.0
            zoom = 1.25
            blur = 0.0
            brightness = 1.0
            
        return dx, dy, zoom, blur, brightness

    print(f"Rendering {total_frames} clinical examination frames at {vid_w}x{vid_h} ({fps} FPS)...")
    
    for f in range(total_frames):
        t_norm = f / float(total_frames)
        dx, dy, zoom, blur, brightness = get_trajectory_params(t_norm)
        
        # Add micro-tremor
        tx = dx + tremor_x[f]
        ty = dy + tremor_y[f]
        
        # Crop centered ROI from fundus_large
        crop_w = int(canvas_size / (zoom * 1.5))
        crop_h = int(crop_w * (vid_h / vid_w))
        
        cx = int(canvas_size / 2 + tx)
        cy = int(canvas_size / 2 + ty)
        
        x1 = max(0, cx - crop_w // 2)
        x2 = min(canvas_size, x1 + crop_w)
        y1 = max(0, cy - crop_h // 2)
        y2 = min(canvas_size, y1 + crop_h)
        
        roi = fundus_rgb[y1:y2, x1:x2]
        frame_resized = cv2.resize(roi, (vid_w, vid_h), interpolation=cv2.INTER_LINEAR)
        
        # Apply corneal reflex glint (subtle specular reflection moving with ophthalmoscope angle)
        glint_x = int(vid_w / 2 + tx * 0.4 + 15 * np.cos(t_norm * 4 * np.pi))
        glint_y = int(vid_h / 2 + ty * 0.4 + 10 * np.sin(t_norm * 4 * np.pi))
        cv2.circle(frame_resized, (glint_x, glint_y), 8, (240, 245, 255), -1)
        # Soften glint edge
        cv2.GaussianBlur(frame_resized, (3, 3), 0, dst=frame_resized)
        
        # Apply focus blur if active
        if blur > 0.5:
            k = int(blur) * 2 + 1
            frame_resized = cv2.GaussianBlur(frame_resized, (k, k), 0)
            
        # Apply illumination & brightness
        if brightness != 1.0:
            frame_resized = np.uint8(np.clip(frame_resized.astype(np.float32) * brightness, 0, 255))
            
        # Apply circular ophthalmoscope aperture vignette
        frame_aperture = np.uint8(frame_resized.astype(np.float32) * vignette_3ch)
        
        # Add subtle sensor noise (simulating high-speed CCD/CMOS in low-light ophthalmoscope)
        noise = np.random.normal(0, 1.8, frame_aperture.shape).astype(np.float32)
        frame_final = np.uint8(np.clip(frame_aperture.astype(np.float32) + noise, 0, 255))
        
        writer.append_data(frame_final)
        
        if (f + 1) % 75 == 0 or f == total_frames - 1:
            print(f"  Rendered {f + 1}/{total_frames} frames ({(f+1)/fps:.1f}s)...")
            
    writer.close()
    print(f"\n[+] Clinical Eye Video successfully generated at: {output_path}")
    print(f"[+] Duration: {duration_sec}s | Total Frames: {total_frames} | Codec: H.264 (libx264)")

if __name__ == "__main__":
    create_clinical_eye_video("sample_eye_scan.mp4", duration_sec=15, fps=25)
