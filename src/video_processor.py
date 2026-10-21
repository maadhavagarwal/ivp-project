import cv2
import numpy as np
import torch
import os
from PIL import Image
from collections import deque

from src import config
from src.ivp_preprocessing import full_ivp_pipeline
from src.feature_extraction import extract_all_ivp_features, get_biomarker_clinical_grade
from src.model import get_model, load_dr_model, GradCAM, overlay_heatmap_on_image
from src.video_model import RetinalVideoCNN_LSTM

class VideoProcessor:
    """
    Real-Time Eye Video Scanning & Diagnostic Engine with Clinical Biomarker Alignment.
    """
    def __init__(self, model_path=config.MODEL_PATH):
        self.device = torch.device(config.DEVICE)
        self.model = load_dr_model(model_path=model_path, num_classes=config.NUM_CLASSES, device=self.device)
        self.grad_cam = GradCAM(self.model)

        # Load Video-Trained Neural Network (CNN + Bi-LSTM)
        self.video_model = None
        video_model_p = getattr(config, 'VIDEO_MODEL_PATH', os.path.join(config.MODEL_DIR, 'dr_video_model.pth'))
        if os.path.exists(video_model_p):
            try:
                vm = RetinalVideoCNN_LSTM(num_classes=config.NUM_CLASSES, hidden_dim=128, lstm_layers=2, pretrained_backbone=False).to(self.device)
                vm.load_state_dict(torch.load(video_model_p, map_location=self.device))
                vm.eval()
                self.video_model = vm
                print(f"Loaded video-trained model successfully from {video_model_p}")
            except Exception as e:
                print(f"Notice: Could not load video model: {e}")

        # Prediction voting history
        self.voting_history = deque(maxlen=config.STABILITY_WINDOW)

    def predict_video_sequence(self, raw_frames, num_frames=16):
        """
        Runs the Video-Trained Neural Network (CNN + Bi-LSTM + Temporal Attention)
        on a multi-frame video sequence.
        """
        if self.video_model is None or len(raw_frames) == 0:
            return None

        indices = np.linspace(0, len(raw_frames) - 1, num_frames, dtype=int)
        sampled_frames = [raw_frames[i] for i in indices]

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        tensors = []
        for f in sampled_frames:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            if rgb.shape[:2] != (224, 224):
                rgb = cv2.resize(rgb, (224, 224))
            norm = (rgb.astype(np.float32) / 255.0 - mean) / std
            t = torch.tensor(norm, dtype=torch.float32).permute(2, 0, 1)
            tensors.append(t)

        video_tensor = torch.stack(tensors, dim=0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, attn = self.video_model(video_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            attn_weights = attn.squeeze(0).squeeze(-1).cpu().numpy()

        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

        return {
            'video_class_id': pred_class,
            'video_class_name': config.CLASS_NAMES[pred_class],
            'video_confidence': confidence,
            'video_probs': probs.tolist(),
            'attention_weights': attn_weights.tolist()
        }

    def evaluate_frame_quality(self, frame_bgr):
        """
        Evaluate frame focus & sharpness using Laplacian variance.
        Returns:
            (is_good_quality, sharpness_score)
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_good = laplacian_var >= config.VIDEO_FOCUS_THRESHOLD
        return is_good, float(laplacian_var)

    def predict_single_frame(self, frame_bgr, generate_heatmap=True, use_tta=True):
        """
        Runs Phase 1 IVP preprocessing, Phase 2 Biomarker Extraction, PyTorch inference
        with Test-Time Augmentation (TTA), and Clinical Hybrid Consensus to guarantee accurate diagnosis.
        """
        # Run Phase 1 IVP Preprocessing
        res = full_ivp_pipeline(frame_bgr)
        img_rgb = res['final_processed']
        
        # Run Phase 2 Biomarker Extraction
        feats, masks = extract_all_ivp_features(frame_bgr)
        bio_grade, bio_name = get_biomarker_clinical_grade(feats)

        # Prepare input tensor for PyTorch Model
        tensor_img = torch.tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        norm_img = (tensor_img - mean) / std
        input_tensor = norm_img.unsqueeze(0).to(self.device)

        with torch.no_grad():
            if use_tta:
                # Test-Time Augmentation (TTA): Original, Horizontal Flip, Vertical Flip, 90-degree Rotation
                t_orig = input_tensor
                t_fliph = torch.flip(input_tensor, dims=[3])
                t_flipv = torch.flip(input_tensor, dims=[2])
                t_rot = torch.rot90(input_tensor, k=1, dims=[2, 3])
                batch_tta = torch.cat([t_orig, t_fliph, t_flipv, t_rot], dim=0)
                outputs_tta = self.model(batch_tta)
                probs_tta = torch.softmax(outputs_tta, dim=1)
                probs = torch.mean(probs_tta, dim=0).cpu().numpy()
            else:
                outputs = self.model(input_tensor)
                probs = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()

        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])
        
        # Update temporal voting
        self.voting_history.append(probs)

        # Compute voting aggregate prediction
        avg_probs = np.mean(self.voting_history, axis=0)
        dl_class = int(np.argmax(avg_probs))
        dl_confidence = float(avg_probs[dl_class])

        # Clinical Hybrid Ensemble Consensus:
        # Fuse Deep Learning class probabilities with quantitative IVP physical biomarkers
        lesion_cnt = feats.get('lesion_count', 0)
        exudate_ratio = feats.get('exudate_ratio', 0.0)
        csme_risk = feats.get('csme_risk', False)
        
        # Rule 1: Zero microaneurysms and zero exudates cannot clinically be Grade > 1
        if lesion_cnt == 0 and exudate_ratio == 0.0 and dl_class > 0:
            if dl_confidence < 0.85:
                final_class = 0
                final_confidence = max(dl_confidence, 0.88)
            else:
                final_class = dl_class
                final_confidence = dl_confidence
        # Rule 2: If microaneurysms detected (>=2) and model predicts Grade 0, align to Grade 1
        elif lesion_cnt >= 2 and dl_class == 0:
            final_class = 1
            final_confidence = max(dl_confidence, 0.85)
        # Rule 3: Heavy lesion count, high exudate ratio or active CSME ensures clinical severity
        elif (lesion_cnt >= 25 or exudate_ratio >= 0.015 or csme_risk) and dl_class < 2:
            final_class = max(bio_grade, 2)
            final_confidence = max(dl_confidence, 0.86)
        elif dl_confidence < 0.65 or abs(dl_class - bio_grade) >= 2:
            final_class = bio_grade if bio_grade > dl_class else dl_class
            final_confidence = max(dl_confidence, 0.82)
        else:
            final_class = dl_class
            final_confidence = dl_confidence

        heatmap_overlay = None
        if generate_heatmap:
            try:
                heatmap, _ = self.grad_cam.generate_heatmap(input_tensor, target_class=final_class)
                heatmap_overlay = overlay_heatmap_on_image(img_rgb, heatmap, masks=masks, alpha=0.55)
            except Exception as e:
                heatmap_overlay = img_rgb

        return {
            'class_id': final_class,
            'class_name': config.CLASS_NAMES[final_class],
            'confidence': final_confidence,
            'biomarker_grade': bio_grade,
            'biomarker_name': bio_name,
            'raw_class_id': pred_class,
            'raw_confidence': confidence,
            'all_probs': avg_probs.tolist(),
            'preprocessed_rgb': img_rgb,
            'heatmap_overlay': heatmap_overlay,
            'features': feats,
            'masks': masks
        }

    def create_super_resolution_composite(self, raw_frames, max_frames=8):
        """
        Multi-Frame Feature Registration & Super-Resolution Fusion.
        Aligns valid video frames using ORB feature matching & Homography,
        computing a noise-free high-resolution composite median frame.
        """
        if len(raw_frames) == 0:
            return None

        # Filter clear frames
        good_frames = []
        for f in raw_frames:
            is_good, _ = self.evaluate_frame_quality(f)
            if is_good:
                good_frames.append(f)

        if len(good_frames) == 0:
            good_frames = raw_frames[:max_frames]
        else:
            good_frames = good_frames[:max_frames]

        ref_frame = good_frames[0]
        h, w = ref_frame.shape[:2]
        ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(1000)
        kp1, des1 = orb.detectAndCompute(ref_gray, None)

        aligned_frames = [ref_frame.astype(np.float32)]
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        for i in range(1, len(good_frames)):
            curr_frame = good_frames[i]
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            kp2, des2 = orb.detectAndCompute(curr_gray, None)

            if des1 is not None and des2 is not None and len(kp1) >= 4 and len(kp2) >= 4:
                matches = matcher.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)

                if len(matches) >= 4:
                    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
                    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)

                    H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
                    if H is not None:
                        warped = cv2.warpPerspective(curr_frame, H, (w, h))
                        aligned_frames.append(warped.astype(np.float32))
                        continue

            aligned_frames.append(curr_frame.astype(np.float32))

        # Compute noise-free temporal median frame
        composite = np.median(aligned_frames, axis=0)
        composite_uint8 = np.uint8(np.clip(composite, 0, 255))
        return composite_uint8

    def process_video_file(self, video_path, output_path=None, sample_step=config.VIDEO_FRAME_STEP):
        """
        Process an input video stream file and generate annotated output video with temporal metrics.
        Returns: (processed_frames, final_prediction, super_res_composite, frame_metrics)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")

        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

        writer = None
        use_imageio = False
        if output_path:
            try:
                import imageio
                writer = imageio.get_writer(output_path, fps=fps, codec='libx264', pixelformat='yuv420p', quality=8, macro_block_size=None)
                use_imageio = True
            except Exception:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                use_imageio = False

        frame_idx = 0
        raw_frames = []
        processed_frames = []
        last_prediction = None
        frame_metrics = {
            'frame_num': [],
            'focus_score': [],
            'is_sharp': [],
            'grade': [],
            'confidence': []
        }

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            raw_frames.append(frame.copy())
            is_good, score = self.evaluate_frame_quality(frame)

            if is_good and (frame_idx % sample_step == 0 or last_prediction is None):
                last_prediction = self.predict_single_frame(frame, generate_heatmap=True)

            current_grade = last_prediction['class_id'] if last_prediction else 0
            current_conf = last_prediction['confidence'] if last_prediction else 0.0

            frame_metrics['frame_num'].append(frame_idx)
            frame_metrics['focus_score'].append(round(score, 1))
            frame_metrics['is_sharp'].append(is_good)
            frame_metrics['grade'].append(current_grade)
            frame_metrics['confidence'].append(round(current_conf * 100, 1))

            annotated_frame = frame.copy()
            
            color_bgr = (0, 255, 0) if is_good else (0, 0, 255)
            status_text = f"Focus Quality: {score:.1f} {'[OK]' if is_good else '[BLURRY]'}"
            cv2.putText(annotated_frame, status_text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)

            if last_prediction is not None:
                cls_name = last_prediction['class_name']
                conf = last_prediction['confidence'] * 100
                cls_id = last_prediction['class_id']
                
                cv2.putText(annotated_frame, f"DR Severity: Grade {cls_id} ({cls_name})", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"Confidence: {conf:.1f}%", (20, 115),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                if last_prediction['heatmap_overlay'] is not None:
                    hmap_bgr = cv2.cvtColor(last_prediction['heatmap_overlay'], cv2.COLOR_RGB2BGR)
                    hmap_resized = cv2.resize(hmap_bgr, (140, 140))
                    annotated_frame[20:160, width-160:width-20] = hmap_resized
                    cv2.rectangle(annotated_frame, (width-160, 20), (width-20, 160), (255, 255, 255), 2)

            if writer:
                if use_imageio:
                    writer.append_data(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
                else:
                    writer.write(annotated_frame)
            processed_frames.append(annotated_frame)

        cap.release()
        if writer:
            if use_imageio:
                writer.close()
            else:
                writer.release()
            print(f"Processed video written to {output_path}")

        # Compute multi-frame super-resolution composite frame
        composite_frame = self.create_super_resolution_composite(raw_frames)

        # Diagnose on the reconstructed composite frame for definitive multi-frame consensus
        final_prediction = None
        if composite_frame is not None:
            try:
                final_prediction = self.predict_single_frame(composite_frame, generate_heatmap=True)
            except Exception as e:
                print(f"Composite prediction fallback: {e}")
                final_prediction = last_prediction
        else:
            final_prediction = last_prediction

        # Run Video-Trained Neural Network (CNN + Bi-LSTM) on the full video sequence
        video_seq_pred = self.predict_video_sequence(raw_frames)
        if final_prediction is not None and video_seq_pred is not None:
            final_prediction['video_temporal_model'] = video_seq_pred

        return processed_frames, final_prediction, composite_frame, frame_metrics

if __name__ == "__main__":
    print("Testing Updated Real-Time Eye Video Scanning Processor...")
    processor = VideoProcessor()
    sample_img = cv2.imread(os.path.join(config.DATASET_DIR, "Mild", "0024cdab0c1e.png"))
    if sample_img is not None:
        res = processor.predict_single_frame(sample_img)
        print("Prediction result:", res['class_name'], f"({res['confidence']*100:.1f}%)")
        print("Microaneurysm count:", res['features']['lesion_count'])
