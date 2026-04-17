#!/usr/bin/env python
"""
Predict autism from video using trained ensemble model
Usage: python predict_ensemble.py <video_path>
"""

import cv2
import numpy as np
import torch
import sys
import os
from pathlib import Path
import joblib
import warnings

warnings.filterwarnings('ignore')

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent))

from models.classifier import IntegratedModel
from models.pose_estimation import get_pose_features

def load_ensemble_model(model_path='checkpoints/ensemble_model.joblib'):
    """Load the trained ensemble (SVM + RF) model"""
    if not os.path.exists(model_path):
        print(f"✗ Ensemble model not found at {model_path}")
        return None
    
    model = joblib.load(model_path)
    print(f"✓ Loaded ensemble model from {model_path}")
    return model

def load_feature_extractor(model_path='checkpoints/best_model.pth', device='cpu'):
    """Load the neural network for feature extraction"""
    model = IntegratedModel(num_classes=1, num_frames=16, hidden_size=64)
    
    if os.path.exists(model_path):
        try:
            ckpt = torch.load(model_path, map_location=device)
            model.load_state_dict(ckpt)
            print(f"✓ Loaded feature extractor from {model_path}")
        except:
            print(f"⚠ Could not load weights from {model_path}, using random initialization")
    
    model.eval()
    return model.to(device)

def extract_video_features(video_path, extractor_model, device, num_frames=16):
    """Extract combined features from video (LSTM + Motion)"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"✗ Cannot open video: {video_path}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  Video: {total_frames} frames @ {fps:.1f} fps")
    
    # Extract frames
    frames = []
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    
    if len(frames) < num_frames:
        print(f"  ⚠ Only {len(frames)} frames extracted (need {num_frames})")
        while len(frames) < num_frames:
            frames.append(frames[-1])
    
    frames = np.array(frames)  # (T, H, W, 3)
    frames = frames.transpose(3, 0, 1, 2) / 255.0  # (3, T, H, W)
    
    # Compute optical flow for 2-channel input to model
    print(f"  Computing optical flow...")
    flow_channels = np.zeros((2, num_frames, 224, 224), dtype=np.float32)
    gray_frames = []
    for t in range(num_frames):
        gray = cv2.cvtColor((frames[:, t].transpose(1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray_frames.append(gray)
    
    for t in range(1, num_frames):
        flow = cv2.calcOpticalFlowFarneback(
            gray_frames[t-1], gray_frames[t],
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        flow_channels[0, t] = flow[..., 0]
        flow_channels[1, t] = flow[..., 1]
    
    flow_tensor = torch.from_numpy(flow_channels).float().unsqueeze(0).to(device)  # (1, 2, T, H, W)
    
    # Extract LSTM features using optical flow input
    print(f"  Extracting LSTM features...")
    with torch.no_grad():
        B, C, T, H, W = flow_tensor.shape
        
        # CNN part with optical flow
        x_cnn = flow_tensor.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        heatmaps = extractor_model.cnn(x_cnn)
        pose_coords = get_pose_features(heatmaps)  # (B*T, 34)
        x_lstm = pose_coords.view(B, T, -1)  # (B, T, 34)
        
        # LSTM part
        lstm_out, _ = extractor_model.lstm(x_lstm)
        lstm_feats = lstm_out[:, -1, :].cpu().numpy().flatten()  # (128,)
    
    # Extract motion features (8 dimensions)
    motion_feats = extract_motion_features_from_flow(flow_channels)  # (8,)
    
    # Combine: LSTM (128) + Motion (8) = 136
    combined_features = np.concatenate([lstm_feats, motion_feats])
    
    return combined_features

def extract_motion_features_from_flow(flow):
    """
    Extract 8-dimensional motion features from optical flow
    Args:
        flow: (2, T, H, W) optical flow
    Returns:
        features: (8,) motion feature vector
    """
    flow_x = flow[0]  # (T, H, W)
    flow_y = flow[1]  # (T, H, W)
    
    flow_mag = np.sqrt(flow_x**2 + flow_y**2)  # (T, H, W)
    
    features = []
    features.append(np.mean(flow_mag))  # 1: Average motion magnitude
    features.append(np.std(flow_mag))   # 2: Std of motion magnitude
    features.append(np.max(flow_mag))   # 3: Max motion magnitude
    features.append(np.mean(np.abs(flow_x)))  # 4: Average |flow_x|
    features.append(np.mean(np.abs(flow_y)))  # 5: Average |flow_y|
    features.append(np.sum(flow_mag > 0.1) / flow_mag.size)  # 6: Proportion of motion
    
    flow_mean_per_frame = flow_mag.mean(axis=(1, 2))  # (T,)
    features.append(np.std(flow_mean_per_frame))  # 7: Temporal variance
    if len(flow_mean_per_frame) > 1:
        features.append(np.mean(np.abs(np.diff(flow_mean_per_frame))))  # 8: Temporal smoothness
    else:
        features.append(0.0)
    
    return np.array(features)

def predict(video_path, ensemble_model, feature_extractor, device='cpu'):
    """
    Predict autism/normal from video
    
    Returns:
        prediction: 'AUTISM' or 'NORMAL'
        confidence: 0-100%
        probability: raw probability from model
    """
    
    print(f"\n📹 Analyzing video: {video_path}")
    print(f"  Device: {device.upper()}")
    
    # Extract features
    print(f"  Extracting LSTM features...")
    features = extract_video_features(video_path, feature_extractor, device, num_frames=16)
    
    if features is None:
        return None, None, None
    
    # Make prediction with ensemble
    print(f"  Running ensemble model...")
    try:
        # Reshape features to 2D: (1 sample, 136 features)
        features_2d = features.reshape(1, -1)
        
        # Ensemble predicts class labels
        y_pred = ensemble_model.predict(features_2d)  # (1,)
        y_proba = ensemble_model.predict_proba(features_2d)  # (1, 2) - probabilities for each class
        
        autism_prob = y_proba[0, 1]  # Probability of autism (class 1)
        normal_prob = y_proba[0, 0]   # Probability of normal (class 0)
        
        if y_pred[0] == 1:
            prediction = "🔴 AUTISM BEHAVIOR DETECTED"
        else:
            prediction = "🟢 NORMAL (No autism behavior)"
        
        confidence = max(autism_prob, normal_prob) * 100
        
    except Exception as e:
        print(f"  ✗ Prediction failed: {e}")
        return None, None, None
    
    return prediction, confidence, autism_prob

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("""
Usage: python predict_ensemble.py <video_path>

Examples:
  python predict_ensemble.py video.mp4
  python predict_ensemble.py /path/to/video.avi
  python predict_ensemble.py cleanest/cleanest/autism/Armflapping/video.mp4
        """)
        sys.exit(1)
    
    video_path = sys.argv[1]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load models
    ensemble = load_ensemble_model()
    if ensemble is None:
        sys.exit(1)
    
    extractor = load_feature_extractor(device=device)
    
    # Predict
    prediction, confidence, prob = predict(video_path, ensemble, extractor, device)
    
    if prediction:
        print(f"\n{'='*50}")
        print(f"RESULT: {prediction}")
        print(f"Confidence: {confidence:.1f}%")
        print(f"Autism Probability: {prob*100:.1f}%")
        print(f"{'='*50}\n")
