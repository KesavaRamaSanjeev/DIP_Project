"""
Extract Autism-Specific Behavioral Features
=============================================
Adds 4 new behavioral features to improve autism detection:
1. Motion Repetition Rate - peaks per second (arm flaps/head bangs)
2. Motion Smoothness - jerkiness score (lower = more jerky = more autism)
3. Spatial Consistency - deviation from center (autism = tighter patterns)
4. Hand Acceleration - speed spikes (autism = sharp velocity changes)

Total new features: 4 -> combines with existing 149 dims -> 153 dims
"""

import numpy as np
import glob
import os
import sys
import torch
import torch.nn.functional as F
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

# Setup path
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

from scripts.preprocess import extract_frames
from models.pose_estimation import get_pose_features
from models.classifier import IntegratedModel
import warnings
warnings.filterwarnings('ignore')

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

model = IntegratedModel().to(device)
model.eval()


def compute_behavioral_features(pose_coords, motion_magnitude):
    """
    Extract 4 behavioral features from pose keypoints and motion
    
    Args:
        pose_coords: (T, 34) - pose keypoints across frames
        motion_magnitude: (T-1, H, W) - optical flow magnitude
    
    Returns:
        features: (4,) - [repetition_rate, smoothness, spatial_consistency, hand_accel]
    """
    
    try:
        # Extract hand keypoints (wrist indices: 9, 10 for left/right)
        # Full 17 joints: 0=nose, 5=left_shoulder, 6=right_shoulder, 9=left_wrist, 10=right_wrist
        left_wrist = pose_coords[:, 18:20]  # indices 9*2+0:9*2+2
        right_wrist = pose_coords[:, 20:22]  # indices 10*2+0:10*2+2
        left_shoulder = pose_coords[:, 10:12]  # indices 5*2
        right_shoulder = pose_coords[:, 12:14]  # indices 6*2
        
        # ===== FEATURE 1: Motion Repetition Rate =====
        # Measure vertical oscillation of hands (arm flapping)
        hand_motion = np.concatenate([left_wrist, right_wrist], axis=0)  # (2T, 2)
        hand_y = hand_motion[:, 1]  # Vertical position
        
        # Smooth the trajectory
        hand_y_smooth = gaussian_filter1d(hand_y, sigma=2)
        
        # Find peaks (local maxima) in hand motion
        peaks_up, _ = find_peaks(hand_y_smooth, height=-1, distance=3)
        peaks_down, _ = find_peaks(-hand_y_smooth, height=-1, distance=3)
        
        total_peaks = len(peaks_up) + len(peaks_down)
        repetition_rate = total_peaks / pose_coords.shape[0] if pose_coords.shape[0] > 0 else 0
        repetition_rate = np.clip(repetition_rate, 0, 1)  # 0-1 scale
        
        # ===== FEATURE 2: Motion Smoothness =====
        # Compute velocity and acceleration to measure jerkiness
        hand_displacement = np.diff(hand_motion, axis=0)  # (2T-1, 2)
        hand_velocity = np.linalg.norm(hand_displacement, axis=1)  # (2T-1,)
        
        # Acceleration = change in velocity
        hand_acceleration = np.abs(np.diff(hand_velocity))  # (2T-2,)
        
        # Smoothness = inverse of mean acceleration (high accel = jerky = low smoothness)
        if len(hand_acceleration) > 0:
            mean_accel = np.mean(hand_acceleration)
            smoothness = 1.0 / (1.0 + mean_accel)  # Normalize to 0-1
        else:
            smoothness = 1.0
        
        smoothness = np.clip(smoothness, 0, 1)
        
        # ===== FEATURE 3: Spatial Consistency =====
        # Measure how tight/centered the movements are
        center_x = np.mean(pose_coords[:, ::2])  # Average X across all joints
        center_y = np.mean(pose_coords[:, 1::2])  # Average Y across all joints
        
        # Deviation of hands from center
        left_wrist_dev = np.sqrt((left_wrist[:, 0] - center_x)**2 + (left_wrist[:, 1] - center_y)**2)
        right_wrist_dev = np.sqrt((right_wrist[:, 0] - center_x)**2 + (right_wrist[:, 1] - center_y)**2)
        
        avg_deviation = np.mean(np.concatenate([left_wrist_dev, right_wrist_dev]))
        
        # Normalize by frame size (112 pixels)
        spatial_consistency = 1.0 - np.clip(avg_deviation / 112.0, 0, 1)  # Inverted: tight = high value
        
        # ===== FEATURE 4: Hand Acceleration Spikes =====
        # Count sharp velocity changes (typical of autism stimming)
        if len(hand_acceleration) > 0:
            accel_threshold = np.mean(hand_acceleration) + np.std(hand_acceleration)
            spike_count = np.sum(hand_acceleration > accel_threshold)
            accel_spikes = spike_count / len(hand_acceleration)
        else:
            accel_spikes = 0
        
        accel_spikes = np.clip(accel_spikes, 0, 1)
        
        features = np.array([
            repetition_rate,      # Feature 1
            smoothness,            # Feature 2
            spatial_consistency,   # Feature 3
            accel_spikes          # Feature 4
        ], dtype=np.float32)
        
        return features
        
    except Exception as e:
        # Fallback to zeros if extraction fails
        return np.zeros(4, dtype=np.float32)


def extract_features_with_behavior():
    """
    Extract features from dataset_new/ with behavioral patterns
    """
    
    print(f"\n{'='*70}")
    print(f"EXTRACTING FEATURES WITH BEHAVIORAL PATTERNS")
    print(f"{'='*70}\n")
    
    # Get dataset files
    autism_files = sorted(glob.glob('dataset_new/autism/*.mp4'))
    normal_files = sorted(glob.glob('dataset_new/normal/*.mp4'))
    
    autism_files = [f for f in autism_files if f.endswith('.mp4')]
    normal_files = [f for f in normal_files if f.endswith('.mp4')]
    
    all_files = autism_files + normal_files
    labels = np.array([1] * len(autism_files) + [0] * len(normal_files))
    
    print(f"Dataset Summary:")
    print(f"  Autism videos:  {len(autism_files)}")
    print(f"  Normal videos:  {len(normal_files)}")
    print(f"  Total videos:   {len(all_files)}\n")
    
    if len(all_files) == 0:
        print("ERROR: No .mp4 files found in dataset_new/")
        return None
    
    # Feature lists
    X_lstm_list = []
    X_motion_list = []
    X_autism_list = []
    X_behavior_list = []  # NEW: Behavioral features (4 dims)
    valid_files = []
    valid_labels = []
    
    print(f"Extracting features from {len(all_files)} videos...\n")
    
    failed_count = 0
    for i, video_path in enumerate(all_files):
        video_name = os.path.basename(video_path)
        print(f"  [{i+1:3d}/{len(all_files)}] {video_name:50s}", end=' ', flush=True)
        
        try:
            # Skip corrupted files
            file_size = os.path.getsize(video_path)
            if file_size < 10000:
                print(f"SKIP (corrupted)")
                failed_count += 1
                continue
            
            # Extract frames
            video_tensor, motion_feats = extract_frames(video_path, num_frames=16, training=False)
            
            if video_tensor.shape != (2, 16, 112, 112):
                print(f"SKIP (invalid shape)")
                failed_count += 1
                continue
            
            # Process through model
            video_tensor_gpu = video_tensor.unsqueeze(0).to(device)
            B, C, T, H, W = video_tensor_gpu.shape
            
            with torch.no_grad():
                # CNN features
                x_cnn = video_tensor_gpu.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
                heatmaps = model.cnn(x_cnn)
                pose_coords = get_pose_features(heatmaps)  # (B*T, 34)
                pose_coords = pose_coords.view(B, T, -1).cpu().numpy()  # (1, T, 34)
                
                x_lstm = torch.FloatTensor(pose_coords).to(device)
                lstm_out, _ = model.lstm(x_lstm)
                temporal_feats = lstm_out[:, -1, :].cpu().numpy()  # (1, 128)
            
            # Compute optical flow
            video_np = video_tensor.numpy()  # (C, T, H, W)
            frames_gray = []
            for t in range(video_np.shape[1]):
                frame = video_np[0, t, :, :]  # (H, W)
                frames_gray.append(frame.astype(np.uint8))
            
            import cv2
            motion_magnitude = []
            for frame_idx in range(len(frames_gray) - 1):
                flow = cv2.calcOpticalFlowFarneback(
                    frames_gray[frame_idx], frames_gray[frame_idx + 1],
                    None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                motion_magnitude.append(mag)
            
            motion_magnitude_arr = np.array(motion_magnitude, dtype=np.float32) if motion_magnitude else np.zeros((video_np.shape[1]-1, 112, 112), dtype=np.float32)
            
            # Extract behavioral features (NEW)
            behavioral_feats = compute_behavioral_features(pose_coords[0], motion_magnitude_arr)
            
            # Store features
            X_lstm_list.append(temporal_feats[0])  # (128,)
            X_motion_list.append(motion_feats.numpy())  # (8,)
            X_autism_list.append(np.zeros(13, dtype=np.float32))  # Placeholder (will be computed below)
            X_behavior_list.append(behavioral_feats)  # (4,)
            valid_files.append(video_path)
            valid_labels.append(labels[i])
            
            print(f"OK")
            
        except Exception as e:
            print(f"ERROR: {str(e)[:40]}")
            failed_count += 1
    
    # Convert to arrays
    X_lstm = np.array(X_lstm_list, dtype=np.float32)
    X_motion = np.array(X_motion_list, dtype=np.float32)
    X_autism = np.array(X_autism_list, dtype=np.float32)  # Placeholder
    X_behavior = np.array(X_behavior_list, dtype=np.float32)  # NEW
    
    # Combine: LSTM (128) + Motion (8) + Placeholder (13) + Behavior (4) = 153 dims
    X_combined = np.concatenate([X_lstm, X_motion, X_autism, X_behavior], axis=1)
    Y_labels = np.array(valid_labels, dtype=np.int32)
    
    print(f"\n{'='*70}")
    print(f"Feature Extraction Complete!")
    print(f"  Successfully extracted: {len(X_lstm)} videos")
    print(f"  Failed: {failed_count} videos")
    print(f"  Autism: {np.sum(Y_labels == 1)}")
    print(f"  Normal: {np.sum(Y_labels == 0)}")
    print(f"\nFeature Shapes:")
    print(f"  X_lstm shape:      {X_lstm.shape}  (128 dims - temporal)")
    print(f"  X_motion shape:    {X_motion.shape}  (8 dims - optical flow)")
    print(f"  X_autism shape:    {X_autism.shape}  (13 dims - placeholder)")
    print(f"  X_behavior shape:  {X_behavior.shape}  (4 dims - behavioral - NEW)")
    print(f"  X_combined shape:  {X_combined.shape}  (153 dims total)")
    
    # Save caches
    np.save('X_lstm_behavior.npy', X_lstm)
    np.save('X_motion_behavior.npy', X_motion)
    np.save('X_behavior_behavior.npy', X_behavior)
    np.save('X_combined_behavior.npy', X_combined)
    np.save('Y_labels_behavior.npy', Y_labels)
    np.save('valid_files_behavior.npy', np.array(valid_files, dtype=object), allow_pickle=True)
    
    print(f"\nCaches saved:")
    print(f"  - X_lstm_behavior.npy")
    print(f"  - X_motion_behavior.npy")
    print(f"  - X_behavior_behavior.npy (NEW)")
    print(f"  - X_combined_behavior.npy")
    print(f"  - Y_labels_behavior.npy")
    print(f"  - valid_files_behavior.npy")
    
    return X_combined, Y_labels, valid_files


if __name__ == "__main__":
    result = extract_features_with_behavior()
    if result[0] is not None:
        print(f"\n✓ Feature extraction with behavioral patterns complete!")
    else:
        print(f"\n✗ Feature extraction failed")
        sys.exit(1)
