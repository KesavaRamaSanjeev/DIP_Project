"""
Phase 1 - Novelty 1 ONLY: Add Bilateral Symmetry to baseline features
Load existing 136 baseline features + compute 1 symmetry feature = 137 total
"""

import numpy as np
import os
from pathlib import Path
import cv2
from scipy.spatial.distance import euclidean
import sys

def compute_bilateral_symmetry_simple(video_path):
    """
    NOVELTY 1: Bilateral Symmetry Asymmetry Index
    Measure left-right asymmetry in movement patterns
    Returns: 1 feature
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    
    if len(frames) < 2:
        return np.array([0.0])
    
    frames = np.array(frames)
    
    # Compute bilateral asymmetry from frame differences
    # Left vs Right regions analysis
    h, w = frames[0].shape
    left_region = frames[:, :, :w//2]
    right_region = frames[:, :, w//2:]
    
    left_motion = np.mean(np.abs(np.diff(left_region, axis=0)))
    right_motion = np.mean(np.abs(np.diff(right_region, axis=0)))
    
    # Asymmetry index
    asymmetry = abs(left_motion - right_motion) / (left_motion + right_motion + 1e-6)
    
    return np.array([asymmetry])

def add_novelty1_features():
    """
    Add NOVELTY 1 (Bilateral Symmetry) to existing baseline features
    Baseline: X_combined_cleanest.npy (333, 136)
    Output: X_novelty1.npy (333, 137)
    """
    print("=" * 70)
    print("ADDING NOVELTY 1: BILATERAL SYMMETRY")
    print("=" * 70)
    print("Baseline: 136 features")
    print("Adding: 1 symmetry feature")
    print("Total: 137 features")
    print("=" * 70)
    
    # Load baseline features
    try:
        X_baseline = np.load('X_combined_cleanest.npy')
        Y_baseline = np.load('Y_labels_cleanest.npy')
        print(f"\n✓ Loaded baseline features: {X_baseline.shape}")
        print(f"✓ Loaded labels: {Y_baseline.shape}")
    except FileNotFoundError:
        print("ERROR: Baseline features not found!")
        print("Run: python scripts/extract_features_cleanest.py")
        return
    
    # Get correct paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    autism_dir = os.path.join(root_dir, 'dataset', 'autism')
    normal_dir = os.path.join(root_dir, 'dataset', 'normal')
    
    # Compute symmetry features
    X_novelty1 = []
    video_files = []
    
    # Autism videos
    autism_files = sorted([f for f in os.listdir(autism_dir) if f.endswith(('.mp4', '.npy'))])
    print(f"\nProcessing {len(autism_files)} autism videos...")
    
    for idx, filename in enumerate(autism_files):
        # Handle both .mp4 and .npy files
        if filename.endswith('.npy'):
            video_path = os.path.join(autism_dir, filename.replace('.npy', '.mp4'))
        else:
            video_path = os.path.join(autism_dir, filename)
        
        if os.path.exists(video_path):
            sym_feat = compute_bilateral_symmetry_simple(video_path)
            X_novelty1.append(sym_feat)
            video_files.append(filename)
        
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
            sym_feat = compute_bilateral_symmetry_simple(video_path)
            X_novelty1.append(sym_feat)
            video_files.append(filename)
        
        if (idx + 1) % 10 == 0:
            print(f"  ✓ Processed {idx + 1}/{len(normal_files)}")
    
    X_novelty1 = np.array(X_novelty1)
    
    # Combine baseline + novelty1
    if len(X_novelty1) == len(X_baseline):
        X_combined = np.hstack([X_baseline, X_novelty1])
        print(f"\n✓ Combined features:")
        print(f"  Baseline: {X_baseline.shape}")
        print(f"  Novelty1: {X_novelty1.shape}")
        print(f"  Combined: {X_combined.shape}")
        
        # Save
        np.save('X_novelty1.npy', X_combined)
        np.save('Y_novelty1.npy', Y_baseline)
        
        print(f"\n✓ Saved:")
        print(f"  X_novelty1.npy ({X_combined.shape})")
        print(f"  Y_novelty1.npy ({Y_baseline.shape})")
        print("=" * 70)
    else:
        print(f"ERROR: Feature count mismatch!")
        print(f"  Baseline: {len(X_baseline)}, Novelty1: {len(X_novelty1)}")

if __name__ == '__main__':
    add_novelty1_features()
