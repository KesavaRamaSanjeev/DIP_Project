#!/usr/bin/env python
"""
Simple script to test whether a video shows autism behaviors or not
Usage: python predict.py <path_to_video> [--model <model_path>]
"""

import cv2
import numpy as np
import torch
import sys
import os
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models.classifier import IntegratedModel
from models.pose_estimation import get_pose_features
from scripts.autism_features import extract_autism_features

def load_model(model_path='checkpoints/best_model.pth', device='cpu'):
    """Load the trained autism detection model"""
    model = IntegratedModel(num_classes=1, num_frames=16, hidden_size=64)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✓ Model loaded from {model_path}")
    else:
        print(f"✗ Model not found at {model_path}")
        print(f"  Available models:")
        if os.path.exists('checkpoints'):
            for f in os.listdir('checkpoints'):
                if f.endswith('.pth'):
                    print(f"    - checkpoints/{f}")
        return None
    
    model.eval()
    return model.to(device)

def compute_optical_flow(frames):
    """
    Compute optical flow from frames
    Args:
        frames: (C, T, H, W) numpy array in range [0, 1]
    Returns: 
        2-channel tensor (flow_x, flow_y) for each frame - shape (2, T, H, W)
    """
    C, T, H, W = frames.shape
    flow_channels = np.zeros((2, T, H, W), dtype=np.float32)
    
    # Convert frames to grayscale for flow computation
    gray_frames = []
    for t in range(T):
        # frames[t] is (C, H, W) -> need to convert to (H, W, C) then to grayscale
        frame = (frames[:, t] * 255).astype(np.uint8)  # (C, H, W)
        # Transpose to (H, W, C) for cv2
        frame_hwc = frame.transpose(1, 2, 0)  # (H, W, 3)
        # Convert RGB to grayscale
        gray = cv2.cvtColor(frame_hwc, cv2.COLOR_RGB2GRAY)
        gray_frames.append(gray)
    
    # Compute optical flow between consecutive frames
    for t in range(1, T):
        flow = cv2.calcOpticalFlowFarneback(
            gray_frames[t-1], gray_frames[t], 
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        flow_channels[0, t] = flow[..., 0]  # x component
        flow_channels[1, t] = flow[..., 1]  # y component
    
    return torch.from_numpy(flow_channels).float()

def extract_motion_features(flow_tensor):
    """
    Extract 8-dimensional motion features from optical flow
    Args:
        flow_tensor: (2, T, H, W) - optical flow x and y components
    Returns:
        motion_feats: (1, 8) - motion feature vector
    """
    flow_x = flow_tensor[0]  # (T, H, W)
    flow_y = flow_tensor[1]  # (T, H, W)
    
    # Compute motion magnitude
    flow_mag = torch.sqrt(flow_x**2 + flow_y**2)  # (T, H, W)
    
    # Extract 8 features
    features = []
    features.append(flow_mag.mean())  # 1: Average motion magnitude
    features.append(flow_mag.std())   # 2: Std of motion magnitude
    features.append(flow_mag.max())   # 3: Max motion magnitude
    features.append(flow_x.abs().mean())  # 4: Average |flow_x|
    features.append(flow_y.abs().mean())  # 5: Average |flow_y|
    features.append(torch.sum(flow_mag > 0.1).float() / flow_mag.numel())  # 6: Proportion of motion
    
    # Temporal features
    flow_mag_mean_per_frame = flow_mag.mean(dim=(1, 2))  # (T,)
    features.append(flow_mag_mean_per_frame.std())  # 7: Temporal variance
    features.append((flow_mag_mean_per_frame[1:] - flow_mag_mean_per_frame[:-1]).abs().mean())  # 8: Temporal smoothness
    
    # Stack features and return as (1, 8)
    motion_feats = torch.stack(features).unsqueeze(0)  # (1, 8)
    return motion_feats

def extract_frames(video_path, num_frames=16, target_fps=30):
    """Extract frames from video and compute optical flow"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"✗ Cannot open video: {video_path}")
        return None, None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  Video info: {total_frames} frames @ {fps:.1f} fps")
    
    frames = []
    frame_count = 0
    
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        frame_count += 1
    
    cap.release()
    
    if len(frames) < num_frames:
        print(f"  ⚠ Only {len(frames)} frames extracted (need {num_frames})")
        return None, None
    
    frames = np.array(frames)  # (T, H, W, 3)
    frames = frames.transpose(3, 0, 1, 2)  # (3, T, H, W)
    frames = frames / 255.0  # Normalize to [0, 1]
    
    # Compute optical flow (2-channel output)
    print(f"  Computing optical flow...")
    flow = compute_optical_flow(frames)  # (2, T, H, W)
    
    return flow, fps

def predict(video_path, model_path='checkpoints/best_model.pth', threshold=0.5):
    """
    Predict whether video shows autism behaviors
    
    Args:
        video_path: Path to video file
        model_path: Path to trained model
        threshold: Classification threshold (0.5 = 50% confidence)
    
    Returns:
        prediction: 'AUTISM' or 'NORMAL'
        confidence: Confidence score (0-100%)
    """
    
    print(f"\n📹 Analyzing video: {video_path}")
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device.upper()}")
    
    # Load model
    model = load_model(model_path, device)
    if model is None:
        return None, None
    
    # Extract frames and compute optical flow
    print(f"  Extracting frames...")
    video_tensor, fps = extract_frames(video_path, num_frames=16)
    
    if video_tensor is None:
        return None, None
    
    # Predict
    print(f"  Running model...")
    with torch.no_grad():
        # Reshape to (B=1, C=2, T=16, H=224, W=224)
        video_input = video_tensor.unsqueeze(0).to(device)
        
        # Extract motion features from optical flow
        motion_feats = extract_motion_features(video_tensor).to(device)  # (1, 8)
        
        # Get raw logit output
        logit = model(video_input, motion_feats)  # (1, 1)
        logit_val = logit.item()
        
        # Apply sigmoid to get probability
        score = torch.sigmoid(logit).item()
        
        print(f"  Raw logit: {logit_val:.6f}, Sigmoid: {score:.6f}")
        
        # Interpret: score > 0.5 = autism, score <= 0.5 = normal
        # For confidence, use absolute distance from 0.5
        confidence = abs(score - 0.5) * 2 * 100  # Convert to 0-100%
    
    # Interpret result
    if score > 0.5:
        prediction = "🔴 AUTISM BEHAVIOR DETECTED"
    else:
        prediction = "🟢 NORMAL (No autism behavior)"
    
    return prediction, confidence

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("""
Usage: python predict.py <video_path> [--model <model_path>]

Examples:
  python predict.py video.mp4
  python predict.py /path/to/video.avi --model checkpoints/best_model.pth
  python predict.py test.mp4 --model checkpoints/fold_1.pth

Models available:
  - checkpoints/best_model.pth (Best overall)
  - checkpoints/fold_1.pth through fold_5.pth (K-fold models)
  - checkpoints/ensemble_model.joblib (Ensemble)
        """)
        sys.exit(1)
    
    video_path = sys.argv[1]
    model_path = 'checkpoints/best_model.pth'
    
    if '--model' in sys.argv:
        idx = sys.argv.index('--model')
        if idx + 1 < len(sys.argv):
            model_path = sys.argv[idx + 1]
    
    prediction, confidence = predict(video_path, model_path)
    
    if prediction:
        print(f"\n{'='*50}")
        print(f"RESULT: {prediction}")
        print(f"Confidence: {confidence:.1f}%")
        print(f"{'='*50}\n")
