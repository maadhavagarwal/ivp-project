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

def get_model(num_classes=config.NUM_CLASSES, pretrained=True):
    """
    Constructs Transfer Learning model (MobileNetV3 / CustomDRNet) for DR detection.
    """
    try:
        if pretrained:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
            model = models.mobilenet_v3_small(weights=weights)
        else:
            model = models.mobilenet_v3_small(weights=None)
            
        in_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(128, num_classes)
        )
        return model
    except Exception as e:
        print(f"Fallback to CustomDRNet architecture due to: {e}")
        return CustomDRNet(num_classes=num_classes)

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

def overlay_heatmap_on_image(orig_rgb, heatmap, alpha=0.5):
    """
    Overlay Grad-CAM heatmap onto RGB image.
    """
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(orig_rgb, 1 - alpha, colored_heatmap, alpha, 0)
    return overlay

if __name__ == "__main__":
    print("Testing Model & Grad-CAM Architecture...")
    model = get_model()
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    print("Model output shape:", out.shape)
    
    grad_cam = GradCAM(model)
    cam, target_cls = grad_cam.generate_heatmap(x)
    print("Grad-CAM generated successfully! Max activation:", np.max(cam))
