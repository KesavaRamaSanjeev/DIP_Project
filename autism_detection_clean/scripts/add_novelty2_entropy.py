"""
Phase 1 - Novelty 2: Add Motion Entropy features
Load Novelty 1 features (137) + compute 4 entropy features = 141 total
"""

import numpy as np
import os
import cv2
from scipy.stats import entropy as scipy_entropy
from scipy.stats import approximate_entropy

def compute_motion_entropy(video_path):
    """
    NOVELTY 2: Local Motion Entropy
    Captures complexity of motion patterns
    Returns: 4 features
    - Shannon Entropy
    - Approximate Entropy
    - Entropy Ratio
    - Predictability Index
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    
    if len(frames) < 3:
        return np.array([0.0, 0.0, 0.0, 0.0])
    
    frames = np.array(frames)
    
    # Compute optical flow
    flow_mags = []
    for i in range(1, len(frames)):
        flow = cv2.calcOpticalFlowFarneback(frames[i-1], frames[i], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.linalg.norm(flow, axis=-1)
        flow_mags.append(mag.flatten())
    
    if len(flow_mags) < 2:
        return np.array([0.0, 0.0, 0.0, 0.0])
    
    flow_mags = np.concatenate(flow_mags)
    
    # Normalize optical flow magnitude
    flow_normalized = (flow_mags - flow_mags.min()) / (flow_mags.max() - flow_mags.min() + 1e-6)
    
    # Feature 1: Shannon Entropy
    hist, _ = np.histogram(flow_normalized, bins=20)
    prob_dist = hist / (hist.sum() + 1e-10)
    shannon_entropy = scipy_entropy(prob_dist + 1e-10)
    
    # Feature 2: Approximate Entropy
    try:
        approx_ent = approximate_entropy(flow_normalized, order=2, metric=0.2)
    except:
        approx_ent = 0.0
    
    # Feature 3: Entropy Ratio (Shannon / Approximate)
    entropy_ratio = shannon_entropy / (approx_ent + 1e-6)
    
    # Feature 4: Predictability Index (inverse of variability)
    motion_variance = np.var(flow_normalized)
    predictability = 1.0 / (1.0 + motion_variance)
    
    return np.array([shannon_entropy, approx_ent, entropy_ratio, predictability])

def add_novelty2_features():
    """
    Add NOVELTY 2 (Motion Entropy) to Novelty 1 features
    Input: X_novelty1.npy (333, 137)
    Output: X_novelty2.npy (333, 141)
    """
    print("=" * 70)
    print("ADDING NOVELTY 2: MOTION ENTROPY")
    print("=" * 70)
    print("Previous: 137 features (baseline + novelty1)")
    print("Adding: 4 entropy features")
    print("Total: 141 features")
    print("=" * 70)
    
    # Load Novelty 1 features
    try:
        X_novelty1 = np.load('X_novelty1.npy')
        Y_novelty = np.load('Y_novelty1.npy')
        print(f"\n✓ Loaded Novelty 1 features: {X_novelty1.shape}")
        print(f"✓ Loaded labels: {Y_novelty.shape}")
    except FileNotFoundError:
        print("ERROR: X_novelty1.npy not found!")
        print("Run: python scripts/add_novelty1_symmetry.py first")
        return
    
    autism_dir = 'dataset/autism'
    normal_dir = 'dataset/normal'
    
    # Get correct paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    autism_dir = os.path.join(root_dir, 'dataset', 'autism')
    normal_dir = os.path.join(root_dir, 'dataset', 'normal')
    
    X_novelty2 = []
    
    # Autism videos
    autism_files = sorted([f for f in os.listdir(autism_dir) if f.endswith(('.mp4', '.npy'))])
    print(f"\nProcessing {len(autism_files)} autism videos...")
    
    for idx, filename in enumerate(autism_files):
        if filename.endswith('.npy'):
            video_path = os.path.join(autism_dir, filename.replace('.npy', '.mp4'))
        else:
            video_path = os.path.join(autism_dir, filename)
        
        if os.path.exists(video_path):
            entropy_feat = compute_motion_entropy(video_path)
            X_novelty2.append(entropy_feat)
        
        if (idx + 1) % 10 == 0:
            print(f"  ✓ Processed {idx + 1}/{len(autism_files)}")
    
    # Normal videos
    normal_files = sorted([f for f in os.listdir(normal_dir) if f.endswith(('.mp4', '.npy'))])
    print(f"\nProcessing {len(normal_files)} normal videos...")
    
    for idx, filename in enumerate(normal_files):
        if filename.endswith('.npy'):
            video_path = os.path.join(normal_dir, filename.replace('.npy', '.mp4'))
        else:
            video_path = os.path.join(normal_dir, filename)
        
        if os.path.exists(video_path):
            entropy_feat = compute_motion_entropy(video_path)
            X_novelty2.append(entropy_feat)
        
        if (idx + 1) % 10 == 0:
            print(f"  ✓ Processed {idx + 1}/{len(normal_files)}")
    
    X_novelty2 = np.array(X_novelty2)
    
    if len(X_novelty2) == len(X_novelty1):
        X_combined = np.hstack([X_novelty1, X_novelty2])
        print(f"\n✓ Combined features:")
        print(f"  Novelty 1: {X_novelty1.shape}")
        print(f"  Novelty 2: {X_novelty2.shape}")
        print(f"  Combined:  {X_combined.shape}")
        
        np.save('X_novelty2.npy', X_combined)
        np.save('Y_novelty2.npy', Y_novelty)
        
        print(f"\n✓ Saved:")
        print(f"  X_novelty2.npy ({X_combined.shape})")
        print(f"  Y_novelty2.npy ({Y_novelty.shape})")
        print("=" * 70)
    else:
        print(f"ERROR: Feature count mismatch!")

if __name__ == '__main__':
    add_novelty2_features()
