import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pose_estimation import CustomHRNet, get_pose_features
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class IntegratedModel(nn.Module):
    def __init__(self, num_classes=1, num_frames=16, hidden_size=64):
        super().__init__()
        self.num_frames = num_frames
        
        # Lightweight Pose Estimator extracting 17 keypoint heatmaps
        self.cnn = CustomHRNet(num_keypoints=17)
        
        self.lstm = nn.LSTM(
            input_size=34 + 6 + 4, # 34 coords + 6 dists + 4 angles
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        self.motion_bn = nn.Identity() # Preserve absolute scale of handcrafted features
        
        # Dedicated High-Capacity Motion MLP
        self.motion_mlp = nn.Sequential(
            nn.Linear(8, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        # Unified Classifier: Final decision head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2 + 16, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
        
        # Initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def extract_lstm_features(self, x):
        """Extract LSTM temporal features from video tensor without motion features"""
        B, C, T, H, W = x.shape
        # CNN-LSTM path Extract sequence pose features
        x_cnn = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        heatmaps = self.cnn(x_cnn)
        pose_coords = get_pose_features(heatmaps) # (B*T, 34)
        x_lstm = pose_coords.view(B, T, -1)
        
        lstm_out, _ = self.lstm(x_lstm)
        temporal_feats = lstm_out[:, -1, :] # (B, hidden_size*2)
        
        return temporal_feats

    def forward(self, x, motion_feats):
        B, C, T, H, W = x.shape
        # CNN-LSTM path Extract sequence pose features
        x_cnn = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        heatmaps = self.cnn(x_cnn)
        pose_coords = get_pose_features(heatmaps) # (B*T, 34)
        x_lstm = pose_coords.view(B, T, -1)
        
        lstm_out, _ = self.lstm(x_lstm)
        temporal_feats = lstm_out[:, -1, :] # (B, hidden_size*2)
        
        # Motion MLP path
        m_feats = self.motion_mlp(motion_feats) # (B, 16)
        
        # Concat: Motion (anchor) + Temporal Pose (refinement)
        combined = torch.cat([temporal_feats, m_feats], dim=1)
        out = self.classifier(combined)
        return out
