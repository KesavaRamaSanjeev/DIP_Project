"""
Phase 1: Enhanced Feature Extraction
=====================================
Adds 3 novel features to existing 136 features:
1. Bilateral Symmetry Asymmetry Index (1 feature)
2. Local Motion Entropy (4 features)  
3. Jerk Analysis (5 features)

Output: X_phase1.npy, Y_phase1.npy (146-dimensional features)
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import cv2
import warnings
from scipy.stats import entropy as scipy_entropy
from scipy.signal import find_peaks

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.classifier import IntegratedModel
from models.pose_estimation import get_pose_features
from scripts.preprocess import extract_frames

warnings.filterwarnings('ignore')

# ============================================
# PHASE 1 NOVEL FEATURES
# ============================================

def compute_bilateral_symmetry(keypoints):
    """
    NOVELTY 1: Bilateral Symmetry Asymmetry Index (1 feature)
    Measures asymmetry between left and right body sides
    
    Left keypoints: 5, 7, 9, 11 (left shoulder, elbow, wrist, hip)
    Right keypoints: 6, 8, 10, 12 (right shoulder, elbow, wrist, hip)
    
    High score = asymmetric movement (autism indicator)
    Low score = symmetric movement (normal)
    """
    try:
        left_motion = np.linalg.norm(
            keypoints[1:, [5, 7, 9, 11]] - keypoints[:-1, [5, 7, 9, 11]], 
            axis=1
        ).mean()
        right_motion = np.linalg.norm(
            keypoints[1:, [6, 8, 10, 12]] - keypoints[:-1, [6, 8, 10, 12]], 
            axis=1
        ).mean()
        asymmetry_index = abs(left_motion - right_motion) / (left_motion + right_motion + 1e-6)
        return np.array([asymmetry_index])
    except Exception as e:
        return np.array([0.0])


def compute_motion_entropy(optical_flow_seq):
    """
    NOVELTY 2: Local Motion Entropy (4 features)
    Measures randomness vs predictability of motion
    
    Low entropy = repetitive movement (autism indicator)
    High entropy = random movement (normal)
    """
    try:
        flow_mag = np.linalg.norm(optical_flow_seq, axis=-1)
        
        # Shannon entropy
        hist, _ = np.histogram(flow_mag, bins=20)
        prob_dist = hist / (hist.sum() + 1e-10)
        shannon_entropy = scipy_entropy(prob_dist + 1e-10)
        
        # Approximate entropy
        def approx_entropy(serie, m=2, r=0.1):
            def _maxdist(x_i, x_j):
                return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
            
            def _phi(m):
                x = [[serie[j] for j in range(i, i + m - 1 + 1)] 
                     for i in range(len(serie) - m + 1)]
                C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / len(x) 
                     for x_i in x]
                return (len(serie) - m + 1) ** (-1) * sum(np.log(C + 1e-10))
            
            return abs(_phi(m + 1) - _phi(m))
        
        approx_ent = approx_entropy(flow_mag)
        entropy_ratio = approx_ent / (shannon_entropy + 1e-6)
        predictability = 1 - (shannon_entropy / np.log(20 + 1e-6))
        
        return np.array([shannon_entropy, approx_ent, entropy_ratio, predictability])
    except Exception as e:
        return np.array([0.0, 0.0, 0.0, 0.0])


def compute_jerk_analysis(keypoints):
    """
    NOVELTY 3: Jerk Analysis - Motion Smoothness Index (5 features)
    Calculates the 3rd derivative of position (jerkiness)
    
    High jerk = sudden, abrupt movements (autism indicator)
    Low jerk = smooth transitions (normal)
    """
    try:
        velocity = np.diff(keypoints, axis=0)
        acceleration = np.diff(velocity, axis=0)
        jerk = np.diff(acceleration, axis=0)
        jerk_mag = np.linalg.norm(jerk, axis=1)
        
        mean_jerk = jerk_mag.mean()
        max_jerk = jerk_mag.max()
        jerk_var = jerk_mag.var()
        jerk_p75 = np.percentile(jerk_mag, 75)
        smooth_ratio = np.sum(jerk_mag < np.median(jerk_mag)) / len(jerk_mag)
        
        return np.array([mean_jerk, max_jerk, jerk_var, jerk_p75, smooth_ratio])
    except Exception as e:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0])


def extract_phase1_features(root_dir='cleanest/cleanest', output_dir='.'):
    """
    Extract Phase 1 features (136 + 10 = 146 dimensions)
    
    Structure:
    - cleanest/cleanest/autism/*/videos
    - cleanest/cleanest/normal/videos
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"PHASE 1 FEATURE EXTRACTION: Enhanced Features")
    print(f"{'='*70}")
    print(f"Using device: {device}")
    print(f"Features added:")
    print(f"  1. Bilateral Symmetry (1 feature)")
    print(f"  2. Motion Entropy (4 features)")
    print(f"  3. Jerk Analysis (5 features)")
    print(f"  = Total: +10 new features (136 → 146)\n")
    
    # Load or initialize model
    print("📂 Loading model...")
    model = IntegratedModel(num_classes=1, num_frames=16, hidden_size=64).to(device)
    model_path = 'checkpoints/best_model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"   ✓ Loaded model from {model_path}\n")
    else:
        print(f"   ⚠ Model not found at {model_path}, using untrained model\n")
    model.eval()
    
    # Collect all video paths
    all_samples = []
    
    print("📹 Scanning for videos...")
    
    # Autism videos
    autism_subfolders = ['Armflapping', 'handaction', 'Headbanging', 'Spinning']
    for subfolder in autism_subfolders:
        path = os.path.join(root_dir, 'autism', subfolder)
        if os.path.isdir(path):
            for f in os.listdir(path):
                if f.lower().endswith(('.mp4', '.avi', '.mov')):
                    all_samples.append((os.path.join(path, f), 1, subfolder))
    
    # Normal videos
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
    X_symmetry = []
    X_entropy = []
    X_jerk = []
    Y = []
    
    print(f"{'='*70}")
    print(f"EXTRACTING FEATURES + NOVELTIES")
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
                
                # Keypoints for novel features (extract from heatmaps)
                # Heatmaps shape: (B*T, 17, 7, 7) - 17 keypoints
                # Convert heatmaps to keypoint coordinates
                keypoints = []
                for t in range(T):
                    frame_heatmaps = heatmaps[t].cpu().numpy()  # (17, 7, 7)
                    frame_kpts = []
                    for kp_idx in range(17):
                        hm = frame_heatmaps[kp_idx]
                        y, x = np.unravel_index(np.argmax(hm), hm.shape)
                        frame_kpts.append([x, y])
                    keypoints.append(frame_kpts)
                
                keypoints = np.array(keypoints)  # (T, 17, 2)
                
                # LSTM Temporal Embedding (64 dimensions)
                x_lstm = pose_feats.reshape(B, T, -1)
                lstm_out, _ = model.lstm(x_lstm)
                temporal_feats = lstm_out[:, -1, :].cpu().numpy().flatten()
                
                # Extract novel features
                symmetry = compute_bilateral_symmetry(keypoints)
                entropy = compute_motion_entropy(motion_tensor)
                jerk = compute_jerk_analysis(keypoints)
                
                X_motion.append(motion_tensor.numpy())
                X_lstm.append(temporal_feats)
                X_symmetry.append(symmetry)
                X_entropy.append(entropy)
                X_jerk.append(jerk)
                Y.append(label)
                success_count += 1
                
            except Exception as e:
                failed_count += 1
                tqdm.write(f"  ⚠ Error: {str(e)[:60]}")
                continue
    
    print(f"\n{'='*70}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"✓ Successfully processed: {success_count} videos")
    print(f"✗ Failed: {failed_count} videos\n")
    
    if success_count == 0:
        print("❌ ERROR: No videos were successfully processed!")
        return
    
    # Combine all features
    X_combined = []
    for lstm, motion, symmetry, entropy, jerk in zip(
        X_lstm, X_motion, X_symmetry, X_entropy, X_jerk
    ):
        combined = np.concatenate([
            lstm,           # 128
            motion,         # 8
            symmetry,       # 1 (PHASE 1)
            entropy,        # 4 (PHASE 1)
            jerk            # 5 (PHASE 1)
        ])
        X_combined.append(combined)
    
    X_combined = np.array(X_combined)
    Y = np.array(Y)
    
    print(f"Feature Composition:")
    print(f"  - LSTM features: 128 dims")
    print(f"  - Motion features: 8 dims")
    print(f"  - Bilateral Symmetry: 1 dim (NEW)")
    print(f"  - Motion Entropy: 4 dims (NEW)")
    print(f"  - Jerk Analysis: 5 dims (NEW)")
    print(f"  - Total: {X_combined.shape[1]} dims\n")
    
    print(f"Dataset Statistics:")
    print(f"  Shape: {X_combined.shape}")
    print(f"  Autism (label=1): {np.sum(Y == 1)} samples ({np.sum(Y == 1)/len(Y)*100:.1f}%)")
    print(f"  Normal (label=0): {np.sum(Y == 0)} samples ({np.sum(Y == 0)/len(Y)*100:.1f}%)\n")
    
    # Save features
    output_path_x = os.path.join(output_dir, 'X_phase1.npy')
    output_path_y = os.path.join(output_dir, 'Y_phase1.npy')
    
    np.save(output_path_x, X_combined)
    np.save(output_path_y, Y)
    
    print(f"✅ Features saved:")
    print(f"   {output_path_x}")
    print(f"   {output_path_y}\n")
    
    return X_combined, Y


if __name__ == "__main__":
    extract_phase1_features()
