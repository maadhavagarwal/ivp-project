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
            
        res = processor.predict_single_frame(img, generate_heatmap=False, use_tta=True)
        
        y_true.append(row['gt_class'])
        y_pred.append(res['class_id'])
        y_bio.append(res['biomarker_grade'])
        y_raw_dl.append(res['raw_class_id'])
        
        if (idx + 1) % 10 == 0 or (idx + 1) == len(eval_df):
            print(f"  Processed {idx + 1}/{len(eval_df)} images...")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 1. Accuracy
    acc = np.mean(y_true == y_pred) * 100
    
    # 2. Quadratic Weighted Kappa (Medical DR Standard)
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    
    # 3. Classification Report
    target_names = [f"Grade {i}: {config.CLASS_NAMES[i]}" for i in range(5)]
    labels_present = sorted(list(set(y_true).union(set(y_pred))))
    labels_names = [target_names[i] for i in labels_present]
    
    report_str = classification_report(y_true, y_pred, labels=labels_present, target_names=labels_names, zero_division=0)
    
    print("\n" + "=" * 65)
    print("[*] BENCHMARK EVALUATION RESULTS")
    print("=" * 65)
    print(f"[+] Overall Multi-Class Accuracy: {acc:.2f}%")
    print(f"[+] Quadratic Weighted Kappa (QWK): {qwk:.4f}")
    print("\nDetailed Clinical Grade Classification Report:")
    print(report_str)
    
    # 4. Save Confusion Matrix
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
        f.write(f"Accuracy: {acc:.2f}%\n")
        f.write(f"QWK Score: {qwk:.4f}\n\n")
        f.write(report_str)
    print(f"[+] Full summary saved to: {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Diabetic Retinopathy Diagnostic System")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples per class to evaluate (default: 10, total 50)")
    parser.add_argument("--all", action="store_true", help="Evaluate entire dataset")
    args = parser.parse_args()

    samples_per_class = 0 if args.all else args.samples
    run_evaluation(num_samples_per_class=samples_per_class)
