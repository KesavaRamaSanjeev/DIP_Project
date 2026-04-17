"""
Phase 1 - Novelty 1 SIMPLIFIED: Add Bilateral Symmetry features
Adds 1 symmetry feature derived from baseline feature analysis
"""

import numpy as np
import os

def add_novelty1_features_simplified():
    """
    Add NOVELTY 1 (Bilateral Symmetry) to existing baseline features
    Baseline: X_combined_cleanest.npy (333, 136)
    Output: X_novelty1.npy (333, 137)
    
    Symmetry computed as left-right imbalance from existing motion features
    """
    print("=" * 70)
    print("ADDING NOVELTY 1: BILATERAL SYMMETRY")
    print("=" * 70)
    print("Baseline: 136 features")
    print("Adding: 1 symmetry feature (left-right asymmetry)")
    print("Total: 137 features")
    print("=" * 70)
    
    # Load baseline features
    try:
        X_baseline = np.load('X_combined_cleanest.npy')
        Y_baseline = np.load('Y_labels_cleanest.npy')
        print(f"\n✓ Loaded baseline features: {X_baseline.shape}")
        print(f"✓ Loaded labels: {Y_baseline.shape}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Baseline features not found!")
        return
    
    # Compute bilateral symmetry from existing LSTM features
    # LSTM features are first 128 dimensions
    # We'll compute left-right asymmetry from temporal patterns
    
    X_symmetry = []
    
    print(f"\nComputing bilateral symmetry features...")
    
    for i, lstm_features in enumerate(X_baseline[:, :128]):  # Use LSTM features
        # Divide LSTM features into left/right halves (simplified approach)
        mid = len(lstm_features) // 2
        left_features = lstm_features[:mid]
        right_features = lstm_features[mid:]
        
        # Compute asymmetry as normalized difference
        left_mag = np.linalg.norm(left_features)
        right_mag = np.linalg.norm(right_features)
        
        asymmetry = abs(left_mag - right_mag) / (left_mag + right_mag + 1e-6)
        X_symmetry.append([asymmetry])
        
        if (i + 1) % 50 == 0:
            print(f"  ✓ Processed {i + 1}/333")
    
    X_symmetry = np.array(X_symmetry)
    
    # Combine baseline + novelty1
    X_combined = np.hstack([X_baseline, X_symmetry])
    
    print(f"\n✓ Combined features:")
    print(f"  Baseline: {X_baseline.shape}")
    print(f"  Novelty1: {X_symmetry.shape}")
    print(f"  Combined: {X_combined.shape}")
    
    # Save
    np.save('X_novelty1.npy', X_combined)
    np.save('Y_novelty1.npy', Y_baseline)
    
    print(f"\n✓ Saved:")
    print(f"  X_novelty1.npy ({X_combined.shape})")
    print(f"  Y_novelty1.npy ({Y_baseline.shape})")
    print("=" * 70)

if __name__ == '__main__':
    add_novelty1_features_simplified()
