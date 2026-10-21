"""
Diabetic Retinopathy Video Training Pipeline
Trains RetinalVideoCNN_LSTM directly on the 125-video dataset (dataset_videos/)
across all 5 DR severity grades using Focal Loss and temporal attention.
Saves model checkpoint to models/dr_video_model.pth.
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, cohen_kappa_score

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from src import config
from src.video_model import RetinalVideoCNN_LSTM
from src.train import FocalLossWithLabelSmoothing

VIDEO_CSV_PATH = os.path.join(config.BASE_DIR, "dataset_videos", "videos_train.csv")
VIDEO_MODEL_SAVE_PATH = os.path.join(config.MODEL_DIR, "dr_video_model.pth")

class RetinalVideoDataset(Dataset):
    """
    Loads 16-frame video clips from dataset_videos/ for temporal deep learning.
    """
    def __init__(self, video_paths, labels, num_frames=16, is_train=True):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.is_train = is_train
        
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def __len__(self):
        return len(self.video_paths)

    def load_video_frames(self, path):
        cap = cv2.VideoCapture(path)
        raw_frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            raw_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        
        if len(raw_frames) == 0:
            raw_frames = [np.zeros((224, 224, 3), dtype=np.uint8)] * self.num_frames
            
        # Sample exactly self.num_frames uniformly
        indices = np.linspace(0, len(raw_frames) - 1, self.num_frames, dtype=int)
        sampled = [raw_frames[i] for i in indices]
        return sampled

    def __getitem__(self, idx):
        v_path = self.video_paths[idx]
        label = self.labels[idx]
        
        frames = self.load_video_frames(v_path)
        
        # Temporal coherent data augmentation
        do_hflip = self.is_train and (np.random.rand() > 0.5)
        do_vflip = self.is_train and (np.random.rand() > 0.5)
        
        proc_tensors = []
        for f in frames:
            if f.shape[:2] != (224, 224):
                f = cv2.resize(f, (224, 224))
            if do_hflip:
                f = np.fliplr(f)
            if do_vflip:
                f = np.flipud(f)
                
            norm = (f.astype(np.float32) / 255.0 - self.mean) / self.std
            tensor_f = torch.tensor(norm, dtype=torch.float32).permute(2, 0, 1)  # (3, 224, 224)
            proc_tensors.append(tensor_f)
            
        # Stack to (T, C, H, W) = (16, 3, 224, 224)
        video_tensor = torch.stack(proc_tensors, dim=0)
        return video_tensor, label, v_path

def train_video_model(epochs=6, batch_size=4, lr=5e-4):
    print("=" * 65)
    print("🎬 STARTING VIDEO NEURAL NETWORK TRAINING (CNN + Bi-LSTM)")
    print("=" * 65)
    
    if not os.path.exists(VIDEO_CSV_PATH):
        raise FileNotFoundError(f"Video metadata CSV not found at {VIDEO_CSV_PATH}. Run create_video_dataset.py first!")
        
    df = pd.read_csv(VIDEO_CSV_PATH)
    print(f"Loaded {len(df)} total video clips from {VIDEO_CSV_PATH}:")
    print(df['class_name'].value_counts())
    
    video_paths = df['video_path'].tolist()
    labels = df['diagnosis'].tolist()
    
    # 80/20 Stratified Split
    train_paths, val_paths, train_y, val_y = train_test_split(
        video_paths, labels, test_size=0.20, random_state=42, stratify=labels
    )
    print(f"\nVideo Dataset Split: {len(train_paths)} Training Videos | {len(val_paths)} Validation Videos")
    
    train_ds = RetinalVideoDataset(train_paths, train_y, num_frames=16, is_train=True)
    val_ds = RetinalVideoDataset(val_paths, val_y, num_frames=16, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing video training on compute device: {device}")
    
    model = RetinalVideoCNN_LSTM(num_classes=5, hidden_dim=128, lstm_layers=2, pretrained_backbone=True).to(device)
    
    # Balanced class loss weights
    class_counts = np.bincount(train_y, minlength=5)
    weights = 1.0 / (class_counts + 1e-5)
    weights = weights / np.sum(weights)
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    
    criterion = FocalLossWithLabelSmoothing(alpha=class_weights_tensor, gamma=2.0, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    best_val_acc = 0.0
    best_val_qwk = -1.0
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for step, (videos, targets, _) in enumerate(train_loader, 1):
            videos, targets = videos.to(device), targets.to(device)
            optimizer.zero_grad()
            
            logits, _ = model(videos)
            loss = criterion(logits, targets)
            loss.backward()
            
            # Gradient clipping for recurrent LSTM stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            
            running_loss += loss.item() * videos.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
            if step % 5 == 0 or step == len(train_loader):
                print(f"  Epoch [{epoch}/{epochs}] Step [{step}/{len(train_loader)}] Step Loss: {loss.item():.4f}")
                
        scheduler.step()
        train_acc = (correct / total) * 100
        
        # Validation evaluation
        model.eval()
        val_loss = 0.0
        val_preds_all = []
        val_targets_all = []
        
        with torch.no_grad():
            for videos, targets, _ in val_loader:
                videos, targets = videos.to(device), targets.to(device)
                logits, _ = model(videos)
                loss = criterion(logits, targets)
                val_loss += loss.item() * videos.size(0)
                preds = torch.argmax(logits, dim=1)
                val_preds_all.extend(preds.cpu().numpy())
                val_targets_all.extend(targets.cpu().numpy())
                
        val_acc = accuracy_score(val_targets_all, val_preds_all) * 100
        val_qwk = cohen_kappa_score(val_targets_all, val_preds_all, weights='quadratic')
        
        print(f"==> Epoch {epoch}/{epochs} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Val QWK: {val_qwk:.4f}")
        
        if val_acc >= best_val_acc or val_qwk > best_val_qwk:
            best_val_acc = max(best_val_acc, val_acc)
            best_val_qwk = max(best_val_qwk, val_qwk)
            torch.save(model.state_dict(), VIDEO_MODEL_SAVE_PATH)
            print(f"    [*] Saved Best Video Model Checkpoint -> {VIDEO_MODEL_SAVE_PATH}")
            
    print("\n" + "=" * 65)
    print("🎬 VIDEO NEURAL NETWORK TRAINING FINISHED!")
    print(f"[+] Best Validation Accuracy: {best_val_acc:.2f}% | Best QWK Score: {best_val_qwk:.4f}")
    print(f"[+] Video-trained weights saved to: {VIDEO_MODEL_SAVE_PATH}")
    print("=" * 65)
    return VIDEO_MODEL_SAVE_PATH

if __name__ == "__main__":
    train_video_model(epochs=5, batch_size=4, lr=5e-4)
