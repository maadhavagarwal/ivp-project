import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import cv2
import numpy as np
from src import config

class CustomDRNet(nn.Module):
    """
    Lightweight 5-class Convolutional Neural Network for Diabetic Retinopathy classification.
    Used as an efficient architecture for CPU training & deployment.
    """
    def __init__(self, num_classes=config.NUM_CLASSES):
        super(CustomDRNet, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        feat = self.features(x)
        feat_flat = feat.view(feat.size(0), -1)
        out = self.classifier(feat_flat)
        return out

def get_model(num_classes=config.NUM_CLASSES, pretrained=True, backbone="mobilenet_v3"):
    """
    Constructs Transfer Learning model (MobileNetV3 / EfficientNetV2 / CustomDRNet) for DR detection.
    """
    try:
        if backbone == "efficientnet_v2":
            weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_v2_s(weights=weights)
            in_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.3, inplace=True),
                nn.Linear(in_features, 128),
                nn.SiLU(inplace=True),
                nn.Linear(128, num_classes)
            )
            return model
        else:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            model = models.mobilenet_v3_small(weights=weights)
            in_features = model.classifier[0].in_features
            model.classifier = nn.Sequential(
                nn.Linear(in_features, 128),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(128, num_classes)
            )
            return model
    except Exception as e:
        print(f"Fallback to MobileNetV3 / CustomDRNet architecture due to: {e}")
        try:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            model = models.mobilenet_v3_small(weights=weights)
            in_features = model.classifier[0].in_features
            model.classifier = nn.Sequential(
                nn.Linear(in_features, 128),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(128, num_classes)
            )
            return model
        except Exception:
            return CustomDRNet(num_classes=num_classes)

def load_dr_model(model_path=config.MODEL_PATH, num_classes=config.NUM_CLASSES, device=None):
    """
    Auto-detects model backbone from checkpoint keys and loads state_dict reliably.
    """
    if device is None:
        device = torch.device(config.DEVICE)
        
    if os.path.exists(model_path):
        try:
            ckpt = torch.load(model_path, map_location=device)
            # Detect architecture from state_dict keys
            if any("features.1.block" in k for k in ckpt.keys()):
                model = get_model(num_classes=num_classes, pretrained=False, backbone="mobilenet_v3")
            elif any("features.1.0" in k or "classifier.1" in k for k in ckpt.keys()):
                model = get_model(num_classes=num_classes, pretrained=False, backbone="efficientnet_v2")
            else:
                model = get_model(num_classes=num_classes, pretrained=False, backbone="mobilenet_v3")
            model.load_state_dict(ckpt)
            print(f"Loaded trained weights successfully from {model_path}")
            model.to(device)
            model.eval()
            return model
        except Exception as e:
            print(f"Notice: Loading trained weights encountered: {e}. Falling back to initialized model.")
            
    model = get_model(num_classes=num_classes, pretrained=True, backbone="mobilenet_v3")
    model.to(device)
    model.eval()
    return model

class GradCAM:
    """
    Grad-CAM generator for visual explainability on fundus images.
    """
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        
        # Locate last conv layer if not specified
        if self.target_layer is None:
            for module in reversed(list(self.model.modules())):
                if isinstance(module, nn.Conv2d):
                    self.target_layer = module
                    break

        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate_heatmap(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap overlay array.
        """
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        score = output[0, target_class]
        score.backward()

        gradients = self.gradients.data.numpy()[0]
        activations = self.activations.data.numpy()[0]

        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        cam = cv2.resize(cam, (config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]))
        return cam, target_class

def overlay_heatmap_on_image(orig_rgb, heatmap, masks=None, alpha=0.55):
    """
    Overlay Grad-CAM heatmap & IVP Lesion Highlights onto RGB image.
    1. Removes non-retina background (eliminates corner artifacts).
    2. Fuses Grad-CAM heatmap with detected IVP red lesions (microaneurysms) and hard exudates.
    3. Renders bright red/orange glowing highlights directly over detected lesion locations.
    """
    h, w = orig_rgb.shape[:2]
    
    # Ensure heatmap matches orig_rgb dimensions
    if heatmap.shape[:2] != (h, w):
        heatmap = cv2.resize(heatmap, (w, h))

    # Create retina circle mask (exclude dark background corners)
    gray = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2GRAY)
    _, retina_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    retina_mask = cv2.morphologyEx(retina_mask, cv2.MORPH_CLOSE, kernel_close)
    
    # 1. Clean background outside retina circle
    heatmap_clean = heatmap.copy()
    heatmap_clean[retina_mask == 0] = 0.0

    # 2. Fuse IVP Lesion Highlights if masks are provided
    if masks is not None:
        lesion_map = np.zeros_like(heatmap_clean)
        
        # Microaneurysms / Red Lesions
        if 'red_lesions' in masks and masks['red_lesions'] is not None:
            r_lesions = masks['red_lesions']
            if r_lesions.shape[:2] != (h, w):
                r_lesions = cv2.resize(r_lesions, (w, h), interpolation=cv2.INTER_NEAREST)
            dilated_r = cv2.dilate(r_lesions, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
            lesion_map[dilated_r > 0] = 1.0
            
        # Hard Exudates
        if 'exudates' in masks and masks['exudates'] is not None:
            ex = masks['exudates']
            if ex.shape[:2] != (h, w):
                ex = cv2.resize(ex, (w, h), interpolation=cv2.INTER_NEAREST)
            dilated_ex = cv2.dilate(ex, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
            lesion_map[dilated_ex > 0] = 1.0
            
        # Blend Grad-CAM activations with IVP lesion highlights
        combined = 0.6 * heatmap_clean + 0.4 * lesion_map
        if np.max(combined) > 0:
            combined = combined / np.max(combined)
        heatmap_clean = combined

    # Suppress low background noise (<0.12) to keep normal retina clean
    heatmap_clean[heatmap_clean < 0.12] = 0.0

    # Apply Jet ColorMap (Red = High activation/lesion, Blue = Low, Black = Clean retina)
    heatmap_uint8 = np.uint8(255 * heatmap_clean)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
    
    # Mask out background corners outside retina
    colored_heatmap[retina_mask == 0] = 0

    # Dynamic Alpha Overlay: Only blend where heatmap has real activation!
    # Where heatmap_clean == 0, retina retains 100% natural color (no blue haze).
    # Where heatmap_clean is high, vibrant red/orange/yellow lesion highlights glow.
    overlay = orig_rgb.astype(np.float32)
    colored_float = colored_heatmap.astype(np.float32)
    
    alpha_map = (heatmap_clean[:, :, np.newaxis] * alpha).astype(np.float32)
    alpha_map[retina_mask == 0] = 0.0

    blended = overlay * (1.0 - alpha_map) + colored_float * alpha_map
    overlay_final = np.uint8(np.clip(blended, 0, 255))
    return overlay_final

if __name__ == "__main__":
    print("Testing Model & Grad-CAM Architecture...")
    model = get_model()
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    print("Model output shape:", out.shape)
    
    grad_cam = GradCAM(model)
    cam, target_cls = grad_cam.generate_heatmap(x)
    print("Grad-CAM generated successfully! Max activation:", np.max(cam))
