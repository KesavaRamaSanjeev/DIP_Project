"""
Phase 1 - Novelty 3 SIMPLIFIED: Add Jerk Analysis features  
Adds 5 jerk features from motion acceleration patterns
"""

import numpy as np

def add_novelty3_features_simplified():
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
    except FileNotFoundError:
        print("ERROR: X_novelty2.npy not found!")
        print("Run: python scripts/add_novelty2_entropy_simple.py first")
        return
    
    X_jerk = []
    
    print(f"\nComputing jerk analysis features...")
    
    for i, features in enumerate(X_novelty2):
        # Extract LSTM temporal features (first 128)
        lstm_features = features[:128]
        
        # Compute velocity (first derivative)
        velocity = np.diff(lstm_features, n=1)
        
        if len(velocity) < 2:
            X_jerk.append([0.0, 0.0, 0.0, 0.0, 0.5])
            continue
        
        # Compute acceleration (second derivative)
        acceleration = np.diff(velocity, n=1)
        
        # Compute jerk (third derivative / rate of change of acceleration)
        if len(acceleration) < 2:
            X_jerk.append([0.0, 0.0, 0.0, 0.0, 0.5])
            continue
        
        jerk = np.diff(acceleration, n=1)
        jerk_mag = np.abs(jerk)
        
        # Feature 1: Mean Jerk
        mean_jerk = np.mean(jerk_mag) if len(jerk_mag) > 0 else 0.0
        
        # Feature 2: Max Jerk
        max_jerk = np.max(jerk_mag) if len(jerk_mag) > 0 else 0.0
        
        # Feature 3: Jerk Variance
        jerk_var = np.var(jerk_mag) if len(jerk_mag) > 0 else 0.0
        
        # Feature 4: 75th Percentile Jerk
        percentile_75 = np.percentile(jerk_mag, 75) if len(jerk_mag) > 0 else 0.0
        
        # Feature 5: Smooth Motion Ratio
        if len(jerk_mag) > 0:
            median_jerk = np.median(jerk_mag)
            smooth_ratio = np.sum(jerk_mag < median_jerk) / len(jerk_mag) if median_jerk > 0 else 0.5
        else:
            smooth_ratio = 0.5
        
        X_jerk.append([mean_jerk, max_jerk, jerk_var, percentile_75, smooth_ratio])
        
        if (i + 1) % 50 == 0:
            print(f"  ✓ Processed {i + 1}/333")
    
    X_jerk = np.array(X_jerk)
    
    # Combine
    X_combined = np.hstack([X_novelty2, X_jerk])
    
    print(f"\n✓ Combined features:")
    print(f"  Novelty 2: {X_novelty2.shape}")
    print(f"  Novelty 3: {X_jerk.shape}")
    print(f"  Combined:  {X_combined.shape}")
    
    np.save('X_novelty3.npy', X_combined)
    np.save('Y_novelty3.npy', Y_novelty)
    
    print(f"\n✓ Saved:")
    print(f"  X_novelty3.npy ({X_combined.shape})")
    print(f"  Y_novelty3.npy ({Y_novelty.shape})")
    print("=" * 70)

if __name__ == '__main__':
    add_novelty3_features_simplified()
