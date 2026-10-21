import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, cohen_kappa_score
import joblib
import cv2

from src import config
from src.ivp_preprocessing import full_ivp_pipeline
from src.feature_extraction import extract_all_ivp_features
from src.model import get_model, load_dr_model

CACHE_DIR = os.path.join(config.BASE_DIR, "preprocessed_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

class FocalLossWithLabelSmoothing(nn.Module):
    """
    Focal Loss with Label Smoothing for Imbalanced Ordinal Medical Classification.
    Focuses on hard-to-classify lesions and prevents overconfidence on adjacent grades.
    """
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1):
        super(FocalLossWithLabelSmoothing, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        log_preds = torch.log_softmax(inputs, dim=1)
        
        with torch.no_grad():
            smooth_targets = torch.full_like(inputs, self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        ce_loss = -torch.sum(smooth_targets * log_preds, dim=1)
        pt = torch.exp(-torch.nn.functional.cross_entropy(inputs, targets, reduction='none'))
        focal_weight = (1.0 - pt) ** self.gamma
        
        if self.alpha is not None:
            loss = self.alpha[targets] * focal_weight * ce_loss
        else:
            loss = focal_weight * ce_loss
            
        return loss.mean()

class DRDataset(Dataset):
    """
    Custom Dataset for Diabetic Retinopathy Fundus Images with Caching & Data Augmentation.
    """
    def __init__(self, image_paths, labels, is_train=True):
        self.image_paths = image_paths
        self.labels = labels
        self.is_train = is_train

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Fast Disk Cache for IVP Preprocessing
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        cache_file = os.path.join(CACHE_DIR, f"{base_name}.npy")
        
        if os.path.exists(cache_file):
            try:
                img_rgb = np.load(cache_file)
            except Exception:
                res = full_ivp_pipeline(img_path)
                img_rgb = res['final_processed']
        else:
            try:
                res = full_ivp_pipeline(img_path)
                img_rgb = res['final_processed']
                np.save(cache_file, img_rgb)
            except Exception:
                img_rgb = np.zeros((config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 3), dtype=np.uint8)

        # Convert HWC numpy (0..255) to PIL Image for augmentations
        pil_img = Image.fromarray(img_rgb)
        
        if self.is_train:
            # Data Augmentations
            if np.random.rand() > 0.5:
                pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
            if np.random.rand() > 0.5:
                pil_img = pil_img.transpose(Image.FLIP_TOP_BOTTOM)
            if np.random.rand() > 0.5:
                angle = np.random.choice([90, 180, 270])
                pil_img = pil_img.rotate(angle)

        img_np = np.array(pil_img)

        # Convert HWC numpy (0..255) to CHW tensor (0..1)
        tensor_img = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor_img = (tensor_img - mean) / std

        return tensor_img, label, img_path

def collect_dataset_file_paths():
    """
    Collect all image file paths and target labels from class folders.
    """
    image_paths = []
    labels = []
    
    for cls_idx, folder_name in config.CLASS_FOLDERS.items():
        folder_dir = os.path.join(config.DATASET_DIR, folder_name)
        if os.path.exists(folder_dir):
            files = glob.glob(os.path.join(folder_dir, "*.png")) + glob.glob(os.path.join(folder_dir, "*.jpg"))
            for f in files:
                image_paths.append(f)
                labels.append(cls_idx)
                
    return image_paths, labels

def train_classical_baseline(train_paths, train_labels, val_paths, val_labels):
    """
    Train classical Machine Learning baseline (Random Forest) on IVP features.
    """
    print("\n--- Training Phase 2: Classical Machine Learning Baseline (Random Forest) ---")
    print(f"Extracting IVP features for {len(train_paths)} training & {len(val_paths)} validation images...")
    
    def extract_dataset_features(paths):
        X = []
        for i, p in enumerate(paths):
            if (i + 1) % 100 == 0 or i == len(paths) - 1:
                print(f"  Processed {i + 1}/{len(paths)} feature extractions...")
            try:
                img = cv2.imread(p)
                feats, _ = extract_all_ivp_features(img)
                X.append(list(feats.values()))
            except Exception:
                X.append([0.0] * 12)
        return np.array(X)

    X_train = extract_dataset_features(train_paths)
    X_val = extract_dataset_features(val_paths)
    
    rf = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")
    rf.fit(X_train, train_labels)
    
    val_preds = rf.predict(X_val)
    acc = accuracy_score(val_labels, val_preds)
    qwk = cohen_kappa_score(val_labels, val_preds, weights="quadratic")
    
    print(f"\nClassical RF Model Validation Accuracy: {acc * 100:.2f}% | QWK Score: {qwk:.4f}")
    print(classification_report(val_labels, val_preds, target_names=list(config.CLASS_NAMES.values()), zero_division=0))
    
    joblib.dump(rf, config.CLASSICAL_MODEL_PATH)
    print(f"Saved classical RF model to: {config.CLASSICAL_MODEL_PATH}")
    return rf

def train_deep_learning_model(epochs=8, sample_limit=800):
    """
    Train Deep Learning PyTorch Model (Phase 3) with extended epochs & learning rate scheduling.
    """
    print("\n=======================================================")
    print("   Starting Extended Diabetic Retinopathy Training")
    print("=======================================================")
    
    image_paths, labels = collect_dataset_file_paths()
    print(f"Found total {len(image_paths)} images across class folders:")
    for cid, folder in config.CLASS_FOLDERS.items():
        count = sum(1 for l in labels if l == cid)
        print(f" - Grade {cid} ({config.CLASS_NAMES[cid]}): {count} images")
    
    if len(image_paths) == 0:
        raise RuntimeError("No image files found in dataset directories!")

    # Balanced class stratification across all 5 classes
    balanced_paths = []
    balanced_labels = []
    max_per_class = max(1, sample_limit // config.NUM_CLASSES)
    print(f"\nPerforming balanced class sampling (up to {max_per_class} images per class)...")
    
    for cid in range(config.NUM_CLASSES):
        cls_paths = [p for p, l in zip(image_paths, labels) if l == cid]
        if len(cls_paths) > max_per_class:
            selected = np.random.choice(cls_paths, max_per_class, replace=False).tolist()
        else:
            selected = cls_paths
        balanced_paths.extend(selected)
        balanced_labels.extend([cid] * len(selected))
        print(f" - Class {cid} ({config.CLASS_NAMES[cid]}): {len(selected)} samples")

    train_paths, val_paths, train_y, val_y = train_test_split(
        balanced_paths, balanced_labels, test_size=0.2, random_state=42, stratify=balanced_labels
    )
    print(f"\nDataset split: {len(train_paths)} Train | {len(val_paths)} Validation")
    
    # 1. Train Classical Baseline
    train_classical_baseline(train_paths, train_y, val_paths, val_y)
    
    # 2. Train Deep Learning PyTorch Model
    print("\n--- Training Phase 3: Deep Learning Neural Network (Focal Loss + Cosine Schedule) ---")
    train_ds = DRDataset(train_paths, train_y, is_train=True)
    val_ds = DRDataset(val_paths, val_y, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    device = torch.device(config.DEVICE)
    model = get_model(num_classes=config.NUM_CLASSES, pretrained=True, backbone="mobilenet_v3").to(device)
    
    # Compute class weights for loss function
    class_counts = np.bincount(train_y, minlength=config.NUM_CLASSES)
    weights = 1.0 / (class_counts + 1e-5)
    weights = weights / np.sum(weights)
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    
    criterion = FocalLossWithLabelSmoothing(alpha=class_weights_tensor, gamma=2.0, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_acc = 0.0
    best_qwk = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for step, (imgs, lbls, _) in enumerate(train_loader, 1):
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)

            if step % 5 == 0 or step == len(train_loader):
                print(f"  Epoch [{epoch}/{epochs}] Step [{step}/{len(train_loader)}] Step Loss: {loss.item():.4f}")
            
        scheduler.step()
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds_all = []
        val_targets_all = []
        
        with torch.no_grad():
            for imgs, lbls, _ in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                val_loss += loss.item() * imgs.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == lbls).sum().item()
                val_total += lbls.size(0)
                val_preds_all.extend(preds.cpu().numpy())
                val_targets_all.extend(lbls.cpu().numpy())
                
        val_acc = val_correct / val_total
        val_qwk = cohen_kappa_score(val_targets_all, val_preds_all, weights="quadratic")
        
        print(f"==> Epoch {epoch}/{epochs} Finished | Train Acc: {epoch_acc*100:.2f}% | Val Loss: {val_loss/val_total:.4f} | Val Acc: {val_acc*100:.2f}% | Val QWK: {val_qwk:.4f}")
        
        if val_acc >= best_val_acc or val_qwk > best_qwk:
            best_val_acc = max(best_val_acc, val_acc)
            best_qwk = max(best_qwk, val_qwk)
            torch.save(model.state_dict(), config.MODEL_PATH)
            print(f"    [Checkpoint Saved] New Best Validation Accuracy: {best_val_acc*100:.2f}% | QWK: {best_qwk:.4f}")
            
    print(f"\n=======================================================")
    print(f"  Extended Training Completed!")
    print(f"  Best Validation Accuracy: {best_val_acc * 100:.2f}% | Best QWK: {best_qwk:.4f}")
    print(f"  Saved PyTorch model weights to: {config.MODEL_PATH}")
    print(f"=======================================================")

if __name__ == "__main__":
    train_deep_learning_model(epochs=6, sample_limit=800)
