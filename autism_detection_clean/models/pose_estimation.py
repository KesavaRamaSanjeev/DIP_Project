import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CustomHRNet(nn.Module):
    """
    A simplified version of HRNet for lightweight pose estimation.
    This architecture maintains high-resolution representations throughout the process.
    """
    def __init__(self, num_keypoints=17):
        super(CustomHRNet, self).__init__()
        
        # Initial stem (Modified for 2-channel input: Diff + Flow)
        self.stem = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Parallel branches (simplified)
        self.branch1 = BasicBlock(32, 32)
        self.branch2 = nn.Sequential(
            nn.AvgPool2d(2),
            BasicBlock(32, 64)
        )
        
        # Upsample branch2 and merge
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Final head to produce heatmaps
        self.final_head = nn.Conv2d(32 + 64, num_keypoints, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        
        feat1 = self.branch1(x)
        feat2 = self.branch2(x)
        
        # Merge branches (multi-resolution fusion)
        feat2_up = self.upsample(feat2)
        if feat2_up.size()[2:] != feat1.size()[2:]:
            feat2_up = F.interpolate(feat2, size=feat1.size()[2:], mode='bilinear', align_corners=True)
            
        out = torch.cat([feat1, feat2_up], dim=1)
        heatmaps = self.final_head(out)
        
        return heatmaps

def get_pose_features(heatmaps):
    """
    Extracts differentiable (x, y) coordinates from heatmaps using Soft-Argmax.
    Returns: (B, num_keypoints * 2) normalized coordinates [0, 1].
    """
    B, K, H, W = heatmaps.shape
    probs = F.softmax(heatmaps.view(B, K, -1), dim=-1).view(B, K, H, W)
    
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, 1, H, device=heatmaps.device),
        torch.linspace(0, 1, W, device=heatmaps.device),
        indexing='ij'
    )
    
    x_coords = torch.sum(probs * grid_x, dim=(2, 3)) # (B, K)
    y_coords = torch.sum(probs * grid_y, dim=(2, 3)) # (B, K)
    
    # Kinematic Features: Calculate Euclidean distances between key points
    # This provides scale and rotation invariance
    # Points 0-4: Head, 5-10: Upper Body, 11-16: Lower Body
    joints = torch.stack([x_coords, y_coords], dim=2) # (B, K, 2)
    
    # Feature 1: Raw normalized coordinates (B, K*2)
    raw_coords = joints.view(B, -1)
    
    # Feature 2: Pairwise distances (6 features)
    dist_pairs = [(0, 9), (0, 10), (9, 10), (5, 6), (9, 5), (10, 6)]
    dists = [torch.norm(joints[:, i] - joints[:, j], p=2, dim=1, keepdim=True) for i, j in dist_pairs]
    kinematics = torch.cat(dists, dim=1) 
    
    # Feature 3: Joint Angles (Scale/Rotation Invariant)
    def compute_angle(a, b, c):
        # Angle at B between vectors BA and BC
        ba = a - b
        bc = c - b
        cosine_angle = torch.sum(ba * bc, dim=1) / (torch.norm(ba, dim=1) * torch.norm(bc, dim=1) + 1e-6)
        angle = torch.acos(torch.clamp(cosine_angle, -1.0, 1.0))
        return angle.unsqueeze(1)
        
    # Example angles: Elbows (Shoulder-Elbow-Wrist)
    # 5:L_Shoulder, 7:L_Elbow, 9:L_Wrist | 6:R_Shoulder, 8:R_Elbow, 10:R_Wrist
    l_elbow_angle = compute_angle(joints[:, 5], joints[:, 7], joints[:, 9])
    r_elbow_angle = compute_angle(joints[:, 6], joints[:, 8], joints[:, 10])
    l_shoulder_angle = compute_angle(joints[:, 7], joints[:, 5], joints[:, 11]) # Shoulder-Shoulder-Hip approx
    r_shoulder_angle = compute_angle(joints[:, 8], joints[:, 6], joints[:, 12])
    
    angles = torch.cat([l_elbow_angle, r_elbow_angle, l_shoulder_angle, r_shoulder_angle], dim=1) # (B, 4)
    
    # Total pose features: 34 (coords) + 6 (dists) + 4 (angles) = 44 features
    return torch.cat([raw_coords, kinematics, angles], dim=1)
