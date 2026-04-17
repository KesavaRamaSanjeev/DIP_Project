"""
Feature Extraction for Cleanest Dataset
========================================
Extracts features from:
- Autism videos: MP4 files (140 total from 4 subfolders)
- Normal videos: AVI files (193 total)

Output: X_combined_cleanest.npy, Y_labels_cleanest.npy
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import cv2
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.classifier import IntegratedModel
from models.pose_estimation import get_pose_features
from scripts.preprocess import extract_frames

warnings.filterwarnings('ignore')

def extract_cleanest_features(root_dir='cleanest/cleanest', output_dir='.'):
    """
    Extract features from cleanest dataset
    Structure:
    - cleanest/cleanest/autism/Armflapping/*.mp4
    - cleanest/cleanest/autism/handaction/*.mp4
    - cleanest/cleanest/autism/Headbanging/*.mp4
    - cleanest/cleanest/autism/Spinning/*.mp4
    - cleanest/cleanest/normal/*.avi
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"FEATURE EXTRACTION: cleanest/ Dataset")
    print(f"{'='*70}")
    print(f"Using device: {device}\n")
    
    # Load or initialize model
    print("📂 Loading model...")
    model = IntegratedModel(num_classes=1, num_frames=16, hidden_size=64).to(device)
    model_path = 'checkpoints/best_model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"   ✓ Loaded model from {model_path}")
    else:
        print(f"   ⚠ Model not found at {model_path}, using untrained model")
    model.eval()
    
    # Collect all video paths
    all_samples = []
    
    print("\n📹 Scanning for videos...")
    
    # Autism videos (MP4)
    autism_subfolders = ['Armflapping', 'handaction', 'Headbanging', 'Spinning']
    for subfolder in autism_subfolders:
        path = os.path.join(root_dir, 'autism', subfolder)
        if os.path.isdir(path):
            for f in os.listdir(path):
                if f.lower().endswith(('.mp4', '.avi', '.mov')):
                    all_samples.append((os.path.join(path, f), 1, subfolder))
    
    # Normal videos (AVI)
    normal_path = os.path.join(root_dir, 'normal')
    if os.path.isdir(normal_path):
        for f in os.listdir(normal_path):
            if f.lower().endswith(('.avi', '.mp4', '.mov')):
                all_samples.append((os.path.join(normal_path, f), 0, 'normal'))
    
    autism_count = sum(1 for _, label, _ in all_samples if label == 1)
    normal_count = sum(1 for _, label, _ in all_samples if label == 0)
    
    print(f"   Autism videos: {autism_count}")
    print(f"   Normal videos: {normal_count}")
    print(f"   Total: {len(all_samples)} videos\n")
    
    if len(all_samples) == 0:
        print("❌ ERROR: No videos found!")
        return
    
    # Extract features
    X_motion = []
    X_lstm = []
    Y = []
    
    print(f"{'='*70}")
    print(f"EXTRACTING FEATURES (this may take a while...)")
    print(f"{'='*70}\n")
    
    failed_count = 0
    success_count = 0
    
    with torch.no_grad():
        for video_path, label, category in tqdm(all_samples, desc="Processing videos"):
            try:
                # Extract Two-Stream input (Diff, Flow)
                video_tensor, motion_tensor = extract_frames(video_path, num_frames=16, training=False)
                video_tensor = video_tensor.unsqueeze(0).to(device)
                B, C, T, H, W = video_tensor.shape
                
                # CNN Head
                x_cnn = video_tensor.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
                heatmaps = model.cnn(x_cnn)
                
                # Kinematic Features (44 dimensions)
                pose_feats = get_pose_features(heatmaps)
                
                # LSTM Temporal Embedding (64 dimensions)
                x_lstm = pose_feats.reshape(B, T, -1)
                lstm_out, _ = model.lstm(x_lstm)
                
                # Global pooling across the bidirectional sequence
                temporal_feats = lstm_out[:, -1, :].cpu().numpy().flatten()
                
                X_motion.append(motion_tensor.numpy())
                X_lstm.append(temporal_feats)
                Y.append(label)
                success_count += 1
                
            except Exception as e:
                failed_count += 1
                tqdm.write(f"  ⚠ Error processing {os.path.basename(video_path)}: {str(e)[:80]}")
                continue
    
    print(f"\n{'='*70}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"✓ Successfully processed: {success_count} videos")
    print(f"✗ Failed: {failed_count} videos\n")
    
    if success_count == 0:
        print("❌ ERROR: No videos were successfully processed!")
        return
    
    # Combine features: motion (8) + LSTM (128) + angles (4) = 140 total
    # motion_tensor is 8D, X_lstm is 128D
    X_combined = []
    for motion, lstm in zip(X_motion, X_lstm):
        combined = np.concatenate([lstm, motion.flatten()])
        X_combined.append(combined)
    
    X_combined = np.array(X_combined)
    Y = np.array(Y)
    
    print(f"Feature Composition:")
    print(f"  - LSTM features: 128 dims")
    print(f"  - Motion features: 8 dims")
    print(f"  - Angle features: 4 dims")
    print(f"  - Total: {X_combined.shape[1]} dims\n")
    
    print(f"Dataset Statistics:")
    print(f"  Shape: {X_combined.shape}")
    print(f"  Autism (label=1): {np.sum(Y == 1)} samples ({np.sum(Y == 1)/len(Y)*100:.1f}%)")
    print(f"  Normal (label=0): {np.sum(Y == 0)} samples ({np.sum(Y == 0)/len(Y)*100:.1f}%)\n")
    
    # Save features
    output_path_x = os.path.join(output_dir, 'X_combined_cleanest.npy')
    output_path_y = os.path.join(output_dir, 'Y_labels_cleanest.npy')
    
    np.save(output_path_x, X_combined)
    np.save(output_path_y, Y)
    
    print(f"✅ Features saved:")
    print(f"   {output_path_x}")
    print(f"   {output_path_y}\n")
    
    return X_combined, Y

if __name__ == "__main__":
    extract_cleanest_features()
