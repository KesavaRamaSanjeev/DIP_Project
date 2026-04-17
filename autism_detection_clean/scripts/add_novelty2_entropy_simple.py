"""
Phase 1 - Novelty 2 SIMPLIFIED: Add Motion Entropy features
Adds 4 entropy features from existing motion data
"""

import numpy as np
from scipy.stats import entropy as scipy_entropy

def add_novelty2_features_simplified():
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
    except FileNotFoundError:
        print("ERROR: X_novelty1.npy not found!")
        print("Run: python scripts/add_novelty1_symmetry_simple.py first")
        return
    
    # Extract motion features (last 8 dimensions of baseline before novelty1)
    X_entropy = []
    
    print(f"\nComputing motion entropy features...")
    
    for i, features in enumerate(X_novelty1):
        # Motion features are dims 128-135 (8 dims)
        motion_features = features[128:136]
        
        # Normalize motion features
        motion_norm = (motion_features - motion_features.min()) / (motion_features.max() - motion_features.min() + 1e-6)
        
        # Feature 1: Shannon Entropy
        hist, _ = np.histogram(motion_norm, bins=8)
        prob_dist = hist / (hist.sum() + 1e-10)
        shannon_entropy = scipy_entropy(prob_dist + 1e-10)
        
        # Feature 2: Approximate Entropy (simplified)
        motion_std = np.std(motion_features)
        approx_ent = np.abs(np.diff(motion_features)).mean()
        
        # Feature 3: Entropy Ratio
        entropy_ratio = shannon_entropy / (approx_ent + 1e-6)
        
        # Feature 4: Predictability Index
        motion_variance = np.var(motion_features)
        predictability = 1.0 / (1.0 + motion_variance)
        
        X_entropy.append([shannon_entropy, approx_ent, entropy_ratio, predictability])
        
        if (i + 1) % 50 == 0:
            print(f"  ✓ Processed {i + 1}/333")
    
    X_entropy = np.array(X_entropy)
    
    # Combine
    X_combined = np.hstack([X_novelty1, X_entropy])
    
    print(f"\n✓ Combined features:")
    print(f"  Novelty 1: {X_novelty1.shape}")
    print(f"  Novelty 2: {X_entropy.shape}")
    print(f"  Combined:  {X_combined.shape}")
    
    np.save('X_novelty2.npy', X_combined)
    np.save('Y_novelty2.npy', Y_novelty)
    
    print(f"\n✓ Saved:")
    print(f"  X_novelty2.npy ({X_combined.shape})")
    print(f"  Y_novelty2.npy ({Y_novelty.shape})")
    print("=" * 70)

if __name__ == '__main__':
    add_novelty2_features_simplified()
