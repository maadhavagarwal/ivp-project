"""
Model & IVP Evaluation Benchmark Script
Evaluates the Diabetic Retinopathy system across all 5 clinical stages:
1. Multi-Class Accuracy & Balanced Accuracy
2. Cohen's Quadratic Weighted Kappa (QWK)
3. Precision, Recall, and F1-Score per severity grade
4. Generates and saves visual Confusion Matrix heatmap
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import cv2
import torch
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
import matplotlib.pyplot as plt

# Ensure UTF-8 stdout encoding on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from src import config
from src.video_processor import VideoProcessor

def run_evaluation(num_samples_per_class=10, save_confusion_matrix=True):
    print("=" * 65)
    print("[*] DIABETIC RETINOPATHY DIAGNOSTIC BENCHMARK EVALUATION")
    print("=" * 65)
    
    processor = VideoProcessor()
    
    # Check dataset existence
    df_path = os.path.join(config.BASE_DIR, "train.csv")
    if not os.path.exists(df_path):
        print(f"Error: Dataset CSV not found at {df_path}")
        return

    df = pd.read_csv(df_path)
    print(f"Loaded train.csv: {len(df)} total patient records.")
    
    # Locate image paths per class folder
    records = []
    for cls_id, folder_name in config.CLASS_FOLDERS.items():
        folder_path = os.path.join(config.DATASET_DIR, folder_name)
        if not os.path.exists(folder_path):
            continue
        
        # Filter df for this class
        sub_df = df[df['diagnosis'] == cls_id]
        found_in_class = 0
        
        for _, row in sub_df.iterrows():
            img_name = f"{row['id_code']}.png"
            full_path = os.path.join(folder_path, img_name)
            if os.path.exists(full_path):
                records.append({
                    'id_code': row['id_code'],
                    'path': full_path,
                    'gt_class': cls_id,
                    'class_name': config.CLASS_NAMES[cls_id]
                })
                found_in_class += 1
                if num_samples_per_class > 0 and found_in_class >= num_samples_per_class:
                    break

    eval_df = pd.DataFrame(records)
    print(f"Selected {len(eval_df)} balanced test samples across all 5 DR grades.")
    print("Running IVP & Deep Learning Inference with Test-Time Augmentation (TTA)...")
    
    y_true = []
    y_pred = []
    y_bio = []
    y_raw_dl = []
    
    for idx, row in eval_df.iterrows():
        img = cv2.imread(row['path'])
        if img is None:
            continue
            
        processor.reset_history()
        res = processor.predict_single_frame(img, generate_heatmap=False, use_tta=True, is_video_stream=False)
        
        y_true.append(row['gt_class'])
        y_pred.append(res['class_id'])
        y_bio.append(res['biomarker_grade'])
        y_raw_dl.append(res['raw_class_id'])
        
        if (idx + 1) % 10 == 0 or (idx + 1) == len(eval_df):
            print(f"  Processed {idx + 1}/{len(eval_df)} images...")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 1. Multi-Class Exact Accuracy
    acc = np.mean(y_true == y_pred) * 100
    
    # 2. Binary Screening Accuracy (Healthy vs Retinopathy - Grade 0 vs 1-4)
    bin_true = (y_true > 0).astype(int)
    bin_pred = (y_pred > 0).astype(int)
    bin_acc = np.mean(bin_true == bin_pred) * 100

    # 3. Clinical Adjacent-Grade Diagnostic Agreement (+- 1 Stage)
    adj_acc = np.mean(np.abs(y_true - y_pred) <= 1) * 100

    # 4. Referable DR Screening Accuracy (Non-Referable Gr 0-1 vs Referable Gr 2-4)
    ref_true = (y_true >= 2).astype(int)
    ref_pred = (y_pred >= 2).astype(int)
    ref_acc = np.mean(ref_true == ref_pred) * 100

    # 5. Quadratic Weighted Kappa (Medical DR Standard)
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    
    # 6. Classification Report
    target_names = [f"Grade {i}: {config.CLASS_NAMES[i]}" for i in range(5)]
    labels_present = sorted(list(set(y_true).union(set(y_pred))))
    labels_names = [target_names[i] for i in labels_present]
    
    report_str = classification_report(y_true, y_pred, labels=labels_present, target_names=labels_names, zero_division=0)
    
    print("\n" + "=" * 65)
    print("[*] BENCHMARK EVALUATION RESULTS (PHOTO FUNDUS IMAGES)")
    print("=" * 65)
    print(f"[+] Binary DR Screening Accuracy (Healthy vs DR)   : {bin_acc:.2f}%  (Target >= 90%)")
    print(f"[+] Clinical Adjacent-Grade Diagnostic Reliability : {adj_acc:.2f}%")
    print(f"[+] Referable DR Screening Accuracy (Urgent Action): {ref_acc:.2f}%")
    print(f"[+] Exact 5-Class Multi-Grade Accuracy             : {acc:.2f}%")
    print(f"[+] Quadratic Weighted Kappa (QWK Score)           : {qwk:.4f}")
    print("\nDetailed Clinical Grade Classification Report:")
    print(report_str)
    
    # 7. Save Confusion Matrix
    if save_confusion_matrix:
        cm = confusion_matrix(y_true, y_pred, labels=range(5))
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Diabetic Retinopathy Classification Confusion Matrix', fontsize=12, pad=15)
        plt.colorbar()
        
        tick_marks = np.arange(5)
        class_short_names = ["No DR (0)", "Mild (1)", "Moderate (2)", "Severe (3)", "Proliferative (4)"]
        plt.xticks(tick_marks, class_short_names, rotation=35, ha="right", fontsize=9)
        plt.yticks(tick_marks, class_short_names, fontsize=9)
        plt.xlabel('Predicted Clinical Grade', fontweight='bold', labelpad=10)
        plt.ylabel('Ground Truth Grade', fontweight='bold', labelpad=10)

        # Annotate numbers
        thresh = cm.max() / 2.
        for i in range(5):
            for j in range(5):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black",
                         fontweight='bold')
                         
        plt.tight_layout()
        cm_path = os.path.join(config.OUTPUT_DIR, "evaluation_confusion_matrix.png")
        plt.savefig(cm_path, dpi=200)
        plt.close()
        print(f"[+] Confusion Matrix plot saved to: {cm_path}")
        
    # Save text summary
    summary_path = os.path.join(config.OUTPUT_DIR, "evaluation_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Diabetic Retinopathy Evaluation Benchmark Summary\n")
        f.write(f"Evaluated Samples: {len(y_true)}\n")
        f.write(f"Binary DR Screening Accuracy (Healthy vs DR): {bin_acc:.2f}%\n")
        f.write(f"Clinical Adjacent-Grade (+-1 Stage) Reliability: {adj_acc:.2f}%\n")
        f.write(f"Exact 5-Class Accuracy: {acc:.2f}%\n")
        f.write(f"QWK Score: {qwk:.4f}\n\n")
        f.write(report_str)
    print(f"[+] Full summary saved to: {summary_path}")

def run_video_evaluation(samples_per_grade=5):
    """
    Evaluates real-time retinal eye scan video clips from dataset_videos/.
    """
    print("\n" + "=" * 65)
    print("[*] EVALUATING TEMPORAL EYE VIDEO SCAN CLIPS (dataset_videos/)")
    print("=" * 65)
    
    csv_path = os.path.join(config.BASE_DIR, "dataset_videos", "videos_train.csv")
    if not os.path.exists(csv_path):
        print(f"Error: Video metadata CSV not found at {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    processor = VideoProcessor()
    
    test_records = []
    for cid in range(5):
        sub = df[df['diagnosis'] == cid].tail(samples_per_grade)
        test_records.extend(sub.to_dict('records'))
        
    print(f"Evaluating {len(test_records)} video clips across 5 clinical severity grades...")
    y_true = []
    y_pred_video = []
    
    for idx, item in enumerate(test_records, 1):
        v_path = item['video_path']
        cap = cv2.VideoCapture(v_path)
        frames = []
        while cap.isOpened():
            ret, f = cap.read()
            if not ret: break
            frames.append(f)
        cap.release()
        
        res = processor.predict_video_sequence(frames)
        y_true.append(item['diagnosis'])
        y_pred_video.append(res['video_class_id'] if res else 0)
        if idx % 5 == 0 or idx == len(test_records):
            print(f"  Processed {idx}/{len(test_records)} video clips...")
            
    y_true = np.array(y_true)
    y_pred_video = np.array(y_pred_video)
    
    v_acc = np.mean(y_true == y_pred_video) * 100
    v_adj = np.mean(np.abs(y_true - y_pred_video) <= 1) * 100
    v_bin = np.mean((y_true > 0) == (y_pred_video > 0)) * 100
    v_qwk = cohen_kappa_score(y_true, y_pred_video, weights='quadratic')
    
    print("\n" + "=" * 65)
    print("[*] VIDEO BENCHMARK EVALUATION RESULTS")
    print("=" * 65)
    print(f"[+] Video Binary Screening Accuracy (Healthy vs DR)   : {v_bin:.2f}%  (Target >= 90%)")
    print(f"[+] Video Clinical Adjacent-Grade Diagnostic Agreement: {v_adj:.2f}%")
    print(f"[+] Video Exact 5-Class Multi-Grade Accuracy          : {v_acc:.2f}%")
    print(f"[+] Video Quadratic Weighted Kappa (QWK Score)        : {v_qwk:.4f}")
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Diabetic Retinopathy Diagnostic System")
    parser.add_argument("--samples", type=int, default=8, help="Number of samples per class to evaluate (default: 8, total 40)")
    parser.add_argument("--all", action="store_true", help="Evaluate entire dataset")
    parser.add_argument("--video", action="store_true", help="Run evaluation on clinical video sequences")
    args = parser.parse_args()

    samples_per_class = 0 if args.all else args.samples
    run_evaluation(num_samples_per_class=samples_per_class)
    
    if args.video:
        run_video_evaluation(samples_per_grade=5)
