import os
import sys
import traceback
import cv2
import numpy as np

print("="*60)
print("RUNNING COMPREHENSIVE SYSTEM INTEGRITY AUDIT")
print("="*60)

errors_found = []
warnings_found = []

# 1. Config Check
try:
    from src import config
    print("[1] Config check: OK")
    print("    - Model path exists:", os.path.exists(config.MODEL_PATH))
    if not os.path.exists(config.MODEL_PATH):
        errors_found.append(f"Model file missing at: {config.MODEL_PATH}")
    print("    - RF Model path exists:", os.path.exists(config.RF_MODEL_PATH))
    if not os.path.exists(config.RF_MODEL_PATH):
        warnings_found.append(f"RF Model file missing at: {config.RF_MODEL_PATH}")
    print("    - Dataset directory exists:", os.path.exists(config.DATASET_DIR))
    print("    - Train CSV exists:", os.path.exists(os.path.join(config.BASE_DIR, "train.csv")))
except Exception as e:
    errors_found.append(f"Config module failed: {e}")

# 2. Preprocessing Check
try:
    from src.ivp_preprocessing import full_ivp_pipeline
    dummy = np.full((300, 300, 3), 100, dtype=np.uint8)
    res = full_ivp_pipeline(dummy)
    assert 'final_processed' in res, "Missing final_processed in pipeline output"
    print("[2] IVP Preprocessing: OK")
except Exception as e:
    errors_found.append(f"IVP Preprocessing failed: {e}\n{traceback.format_exc()}")

# 3. Feature Extraction Check
try:
    from src.feature_extraction import extract_all_ivp_features, get_biomarker_clinical_grade
    feats, masks = extract_all_ivp_features(dummy)
    grade, name = get_biomarker_clinical_grade(feats)
    print("[3] Feature Extraction: OK")
    print(f"    - Extracted {len(feats)} features and {len(masks)} masks")
except Exception as e:
    errors_found.append(f"Feature Extraction failed: {e}\n{traceback.format_exc()}")

# 4. Model Loading & Grad-CAM Check
try:
    import torch
    from src.model import load_dr_model, GradCAM, overlay_heatmap_on_image
    device = torch.device('cpu')
    model = load_dr_model(config.MODEL_PATH, num_classes=5, device=device)
    grad_cam = GradCAM(model)
    dummy_tensor = torch.randn(1, 3, 224, 224)
    cam, target = grad_cam.generate_heatmap(dummy_tensor, target_class=0)
    assert cam.shape == (224, 224), f"Cam shape mismatch: {cam.shape}"
    blended = overlay_heatmap_on_image(dummy, cam)
    print("[4] Model & Grad-CAM: OK")
except Exception as e:
    errors_found.append(f"Model & Grad-CAM failed: {e}\n{traceback.format_exc()}")

# 5. VideoProcessor Check
try:
    from src.video_processor import VideoProcessor
    vp = VideoProcessor()
    # Test single frame prediction
    single_res = vp.predict_single_frame(dummy, generate_heatmap=True)
    assert 'class_id' in single_res, "class_id missing"
    assert 'confidence' in single_res, "confidence missing"
    print("[5] VideoProcessor predict_single_frame: OK")
    
    # Test video processing on sample_eye_scan.mp4 if exists
    sample_vid = os.path.join(config.BASE_DIR, "sample_eye_scan.mp4")
    if os.path.exists(sample_vid):
        test_out = os.path.join(config.OUTPUT_DIR, "test_proc_video.mp4")
        pf, fp, comp, fm = vp.process_video_file(sample_vid, output_path=test_out, sample_step=5)
        print(f"[5b] VideoProcessor process_video_file: OK (Processed {len(pf)} frames)")
    else:
        warnings_found.append("sample_eye_scan.mp4 not found in root")
except Exception as e:
    errors_found.append(f"VideoProcessor failed: {e}\n{traceback.format_exc()}")

# 6. PDF and HTML Report Generator Check
try:
    from src.pdf_generator import generate_html_report, generate_pdf_summary_image
    p_info = {'name': 'Audit Test', 'age': '45', 'sex': 'Female', 'eye': 'Left Eye (OS)'}
    res_dict = {
        'class_id': 2,
        'confidence': 0.88,
        'features': {
            'vessel_density': 0.12,
            'frangi_score': 0.08,
            'gabor_score': 32.1,
            'exudate_ratio': 0.015,
            'exudate_count': 5,
            'lesion_count': 12,
            'fovea_lesion_distance_dd': 0.9,
            'csme_risk': True
        }
    }
    h_out = os.path.join(config.OUTPUT_DIR, "audit_report.html")
    p_out = os.path.join(config.OUTPUT_DIR, "audit_report.pdf")
    generate_html_report(p_info, res_dict, h_out)
    generate_pdf_summary_image(p_info, res_dict, dummy, dummy, p_out)
    assert os.path.exists(h_out), "HTML report not created"
    assert os.path.exists(p_out), "PDF report not created"
    print("[6] PDF & HTML Report Generator: OK")
except Exception as e:
    errors_found.append(f"Report generator failed: {e}\n{traceback.format_exc()}")

# 7. Training Pipeline Check
try:
    from src.train import FocalLossWithLabelSmoothing, DRDataset
    fl = FocalLossWithLabelSmoothing()
    pred_t = torch.randn(4, 5, requires_grad=True)
    targ_t = torch.tensor([0, 1, 2, 3])
    loss = fl(pred_t, targ_t)
    loss.backward()
    print("[7] Training loss & autograd: OK")
except Exception as e:
    errors_found.append(f"Training pipeline check failed: {e}\n{traceback.format_exc()}")

# 8. Evaluation Benchmark Script Check
try:
    import evaluate
    print("[8] evaluate.py import: OK")
except Exception as e:
    errors_found.append(f"evaluate.py failed: {e}\n{traceback.format_exc()}")

print("\n" + "="*60)
print("AUDIT RESULTS SUMMARY:")
print("="*60)
if errors_found:
    print(f"FAILED: {len(errors_found)} critical error(s) found:")
    for err in errors_found:
        print("  -", err)
else:
    print("ALL 8 CORE SYSTEM MODULES PASSED RUNTIME EXECUTION TESTS.")

if warnings_found:
    print(f"WARNINGS ({len(warnings_found)}):")
    for w in warnings_found:
        print("  -", w)
