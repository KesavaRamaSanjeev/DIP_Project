"""
Phase 1 - Novelty 3: Add Jerk Analysis features
Load Novelty 2 features (141) + compute 5 jerk features = 146 total
"""

import numpy as np
import os
import cv2

def compute_jerk_analysis(video_path):
    """
    NOVELTY 3: Jerk Analysis (smoothness of motion)
    Jerk = time derivative of acceleration
    Returns: 5 features
    - Mean Jerk
    - Max Jerk
    - Jerk Variance
    - 75th Percentile Jerk
    - Smooth Motion Ratio
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    
    if len(frames) < 4:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    
    frames = np.array(frames)
    
    # Compute optical flow for velocity approximation
    velocity = []
    for i in range(1, len(frames)):
        flow = cv2.calcOpticalFlowFarneback(frames[i-1], frames[i], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        vel_mag = np.linalg.norm(flow, axis=-1).mean()
        velocity.append(vel_mag)
    
    velocity = np.array(velocity)
    
    if len(velocity) < 3:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Acceleration = second derivative
    acceleration = np.diff(velocity, n=1)
    
    # Jerk = third derivative (derivative of acceleration)
    if len(acceleration) < 2:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    
    jerk = np.diff(acceleration, n=1)
    jerk_mag = np.abs(jerk)
    
    # Feature 1: Mean Jerk
    mean_jerk = np.mean(jerk_mag)
    
    # Feature 2: Max Jerk
    max_jerk = np.max(jerk_mag)
    
    # Feature 3: Jerk Variance
    jerk_var = np.var(jerk_mag)
    
    # Feature 4: 75th Percentile Jerk
    percentile_75 = np.percentile(jerk_mag, 75)
    
    # Feature 5: Smooth Motion Ratio (low jerk indicates smooth motion)
    # Ratio of smooth frames (jerk < median) to total
    median_jerk = np.median(jerk_mag)
    smooth_ratio = np.sum(jerk_mag < median_jerk) / len(jerk_mag)
    
    return np.array([mean_jerk, max_jerk, jerk_var, percentile_75, smooth_ratio])

def add_novelty3_features():
    """
    Add NOVELTY 3 (Jerk Analysis) to Novelty 2 features
    Input: X_novelty2.npy (333, 141)
    Output: X_novelty3.npy (333, 146)
    """
    print("=" * 70)
    print("ADDING NOVELTY 3: JERK ANALYSIS")
    print("=" * 70)
    print("Previous: 141 features (baseline + novelty1 + novelty2)")
    print("Adding: 5 jerk analysis features")
    print("Total: 146 features")
    print("=" * 70)
    
    try:
        X_novelty2 = np.load('X_novelty2.npy')
        Y_novelty = np.load('Y_novelty2.npy')
        print(f"\n✓ Loaded Novelty 2 features: {X_novelty2.shape}")
        print(f"✓ Loaded labels: {Y_novelty.shape}")
    except FileNotFoundError:
        print("ERROR: X_novelty2.npy not found!")
        print("Run: python scripts/add_novelty2_entropy.py first")
        return
    
    autism_dir = 'dataset/autism'
    normal_dir = 'dataset/normal'
    
    # Get correct paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    autism_dir = os.path.join(root_dir, 'dataset', 'autism')
    normal_dir = os.path.join(root_dir, 'dataset', 'normal')
    
    X_novelty3 = []
    
    # Autism videos
    autism_files = sorted([f for f in os.listdir(autism_dir) if f.endswith(('.mp4', '.npy'))])
    print(f"\nProcessing {len(autism_files)} autism videos...")
    
    for idx, filename in enumerate(autism_files):
        if filename.endswith('.npy'):
            video_path = os.path.join(autism_dir, filename.replace('.npy', '.mp4'))
        else:
            video_path = os.path.join(autism_dir, filename)
        
        if os.path.exists(video_path):
            jerk_feat = compute_jerk_analysis(video_path)
            X_novelty3.append(jerk_feat)
        
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
            jerk_feat = compute_jerk_analysis(video_path)
            X_novelty3.append(jerk_feat)
        
        if (idx + 1) % 10 == 0:
            print(f"  ✓ Processed {idx + 1}/{len(normal_files)}")
    
    X_novelty3 = np.array(X_novelty3)
    
    if len(X_novelty3) == len(X_novelty2):
        X_combined = np.hstack([X_novelty2, X_novelty3])
        print(f"\n✓ Combined features:")
        print(f"  Novelty 2: {X_novelty2.shape}")
        print(f"  Novelty 3: {X_novelty3.shape}")
        print(f"  Combined:  {X_combined.shape}")
        
        np.save('X_novelty3.npy', X_combined)
        np.save('Y_novelty3.npy', Y_novelty)
        
        print(f"\n✓ Saved:")
        print(f"  X_novelty3.npy ({X_combined.shape})")
        print(f"  Y_novelty3.npy ({Y_novelty.shape})")
        print("=" * 70)
    else:
        print(f"ERROR: Feature count mismatch!")

if __name__ == '__main__':
    add_novelty3_features()
