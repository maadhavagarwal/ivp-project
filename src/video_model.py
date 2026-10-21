"""
Spatial-Temporal Video Neural Network Architecture (CNN + Bi-LSTM + Temporal Attention)
Processes 5D video tensors: (Batch, Frames, Channels, Height, Width)
1. Spatial Feature Extractor: MobileNetV3 backbone extracts frame-level embeddings
2. Temporal Sequence Modeler: Bidirectional LSTM learns inter-frame lesion dynamics
3. Temporal Attention Pooling: Dynamically weights frames by lesion visibility
4. Multi-class Head: Classifies video sequence into 5 Diabetic Retinopathy stages
"""

import torch
import torch.nn as nn
import torchvision.models as models

class TemporalAttention(nn.Module):
    """
    Learns attention weights across video frames to prioritize frames with visible lesions
    over blurred or transit frames.
    """
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, lstm_outputs):
        # lstm_outputs: (B, T, hidden_dim)
        scores = self.attn(lstm_outputs)  # (B, T, 1)
        weights = torch.softmax(scores, dim=1)  # (B, T, 1)
        context = torch.sum(lstm_outputs * weights, dim=1)  # (B, hidden_dim)
        return context, weights

class RetinalVideoCNN_LSTM(nn.Module):
    """
    Unified Video Neural Network for Continuous Retinal Fundus Video Analysis.
    Input: (Batch, Num_Frames, 3, 224, 224)
    Output: (Batch, 5) Logits for DR Grades 0 to 4
    """
    def __init__(self, num_classes=5, hidden_dim=128, lstm_layers=2, pretrained_backbone=True):
        super(RetinalVideoCNN_LSTM, self).__init__()
        
        # 1. Spatial Backbone (MobileNetV3-Small)
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained_backbone else None)
        self.spatial_backbone = mobilenet.features
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # MobileNetV3-Small features output 576 channels
        spatial_feat_dim = 576
        
        # 2. Projection & Batch Normalization
        self.proj = nn.Sequential(
            nn.Linear(spatial_feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        # 3. Temporal Sequence Modeler (Bidirectional LSTM)
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if lstm_layers > 1 else 0.0
        )
        
        # Bidirectional outputs 2 * hidden_dim
        bi_hidden = hidden_dim * 2
        
        # 4. Temporal Attention Pooling
        self.temporal_attn = TemporalAttention(bi_hidden)
        
        # 5. Diagnostic Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(bi_hidden, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        x: (B, T, C, H, W)
        """
        b, t, c, h, w = x.shape
        
        # Collapse batch and temporal dimensions for spatial CNN pass
        x_reshaped = x.view(b * t, c, h, w)
        feat_map = self.spatial_backbone(x_reshaped)  # (B*T, 576, 7, 7)
        feat_pooled = self.spatial_pool(feat_map).squeeze(-1).squeeze(-1)  # (B*T, 576)
        
        # Linear projection
        proj_feat = self.proj(feat_pooled)  # (B*T, 256)
        
        # Reshape back to temporal sequence: (B, T, 256)
        seq_feat = proj_feat.view(b, t, 256)
        
        # Temporal LSTM pass: (B, T, 2 * hidden_dim)
        lstm_out, _ = self.lstm(seq_feat)
        
        # Temporal Attention Pooling across frames: (B, 2 * hidden_dim)
        context, attn_weights = self.temporal_attn(lstm_out)
        
        # Multi-class prediction logits
        logits = self.classifier(context)  # (B, 5)
        
        return logits, attn_weights

if __name__ == "__main__":
    print("Testing Video CNN-LSTM Architecture...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RetinalVideoCNN_LSTM(num_classes=5).to(device)
    
    dummy_video = torch.randn(2, 16, 3, 224, 224).to(device)
    logits, attn = model(dummy_video)
    
    print("Input Video Batch Shape:", dummy_video.shape)
    print("Output Logits Shape:", logits.shape)
    print("Attention Weights Shape:", attn.shape)
    print("Total Parameters:", sum(p.numel() for p in model.parameters()))
