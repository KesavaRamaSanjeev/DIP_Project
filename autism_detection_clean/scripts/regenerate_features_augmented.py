"""
Regenerate cached features with Autism-Specific Behavioral Features
===================================================================
Features: LSTM (128) + Original Motion (8) + Autism Behavioral (13) = 149 total
"""

import sys
import os

# Move to parent directory
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

import numpy as np
import glob
import torch

from scripts.preprocess import extract_frames
from models.pose_estimation import get_pose_features
from models.classifier import IntegratedModel
from scripts.autism_features import extract_autism_features


def regenerate_augmented_features():
    """Extract and cache features with autism behavioral metrics"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print("Loading model...")
    model = IntegratedModel(num_classes=1, num_frames=16, hidden_size=64).to(device)
    model.eval()
    
    # Get clean dataset files
    autism_files = sorted(glob.glob('dataset/autism/*.mp4'))
    normal_files = sorted(glob.glob('dataset/normal/*.mp4'))
    
    autism_files = [f for f in autism_files if f.endswith('.mp4')]
    normal_files = [f for f in normal_files if f.endswith('.mp4')]
    
    all_files = autism_files + normal_files
    labels = np.array([1] * len(autism_files) + [0] * len(normal_files))
    
    print(f"\nDataset Summary:")
    print(f"  Autism: {len(autism_files)}")
    print(f"  Normal: {len(normal_files)}")
    print(f"  Total: {len(all_files)}")
    
    # Extract features
    X_lstm_list = []
    X_motion_list = []
    X_autism_list = []
    
    print(f"\nExtracting features...")
    for i, video_path in enumerate(all_files):
        print(f"  [{i+1}/{len(all_files)}] {os.path.basename(video_path)}...", end=' ')
        try:
            # Extract frames and get video tensor
            video_tensor, motion_feats = extract_frames(video_path, num_frames=16, training=False)
            
            # Process through CNN-LSTM to get pose features
            video_tensor_gpu = video_tensor.unsqueeze(0).to(device)
            B, C, T, H, W = video_tensor_gpu.shape
            
            with torch.no_grad():
                # CNN features extraction
                x_cnn = video_tensor_gpu.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
                heatmaps = model.cnn(x_cnn)
                pose_coords = get_pose_features(heatmaps)  # (B*T, 34)
                pose_coords = pose_coords.view(B, T, -1).cpu().numpy()  # (B, T, 34)
                
                x_lstm = torch.FloatTensor(pose_coords).to(device)
                
                # LSTM features
                lstm_out, _ = model.lstm(x_lstm)
                temporal_feats = lstm_out[:, -1, :].cpu().numpy()  # (1, 128)
            
            X_lstm_list.append(temporal_feats[0])
            X_motion_list.append(motion_feats.numpy())
            
            # Extract autism-specific behavioral features
            video_np = video_tensor.numpy()  # (C, T, H, W) -> convert to (T, H, W, C)
            video_np = np.transpose(video_np, (1, 2, 3, 0))  # (T, H, W, C)
            motion_np = motion_feats.numpy()  # (T-1, H, W, 2)
            
            autism_feats = extract_autism_features(video_np, motion_np, pose_coords[0])
            X_autism_list.append(autism_feats)
            print("✓")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            # Add zeros if extraction fails
            X_lstm_list.append(np.zeros(128))
            X_motion_list.append(np.zeros(8))
            X_autism_list.append(np.zeros(13))
    
    # Stack into arrays
    X_lstm = np.array(X_lstm_list)  # (117, 128)
    X_motion = np.array(X_motion_list)  # (117, 8)
    X_autism = np.array(X_autism_list)  # (117, 13)
    X_combined = np.concatenate([X_lstm, X_motion, X_autism], axis=1)  # (117, 149)
    
    print(f"\nFeature shapes:")
    print(f"  X_lstm: {X_lstm.shape}")
    print(f"  X_motion: {X_motion.shape}")
    print(f"  X_autism: {X_autism.shape}")
    print(f"  X_combined: {X_combined.shape}")
    print(f"  Y_labels: {labels.shape}")
    
    # Save cached features (keep old ones for reference)
    print(f"\nSaving cached features...")
    np.save('X_lstm.npy', X_lstm)
    np.save('X_motion.npy', X_motion)
    np.save('X_autism.npy', X_autism)
    np.save('X_combined.npy', X_combined)
    np.save('Y_labels.npy', labels)
    
    print("✓ All features cached successfully!")
    print(f"\nFeature breakdown:")
    print(f"  - LSTM temporal: 128 dimensions")
    print(f"  - Optical flow motion: 8 dimensions")
    print(f"  - Autism behavioral: 13 dimensions")
    print(f"  - Total combined: 149 dimensions")
    
    return X_lstm, X_motion, X_autism, X_combined, labels


if __name__ == "__main__":
    regenerate_augmented_features()
