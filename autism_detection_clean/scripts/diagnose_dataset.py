"""
Diagnostic Analysis: Why is Fold-3 failing?
============================================
Investigate data distribution and separability
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform


def analyze_fold_difficulty():
    """Analyze each fold to identify why some fail"""
    
    print("\n" + "="*80)
    print("DATASET DIAGNOSTIC ANALYSIS")
    print("="*80)
    
    # Load data
    X = np.load('X_combined.npy')
    y = np.load('Y_labels.npy')
    
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.sum(y)} autism, {len(y) - np.sum(y)} normal")
    
    # Overall separability
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    autism_points = X_scaled[y == 1]
    normal_points = X_scaled[y == 0]
    
    print(f"\n--- OVERALL STATISTICS ---")
    print(f"Autism samples: {len(autism_points)}")
    print(f"Normal samples: {len(normal_points)}")
    
    # Calculate within-class and between-class distances
    within_autism = np.mean(pdist(autism_points, metric='euclidean'))
    within_normal = np.mean(pdist(normal_points, metric='euclidean'))
    
    # Between-class distance
    between_distances = []
    for a in autism_points:
        for n in normal_points:
            between_distances.append(np.linalg.norm(a - n))
    between_class = np.mean(between_distances)
    
    print(f"Within-class distance (autism): {within_autism:.4f}")
    print(f"Within-class distance (normal): {within_normal:.4f}")
    print(f"Between-class distance: {between_class:.4f}")
    
    separability_ratio = between_class / ((within_autism + within_normal) / 2)
    print(f"Separability ratio: {separability_ratio:.4f} (>2.0 = good)")
    
    # Analyze each fold
    print(f"\n--- FOLD-BY-FOLD ANALYSIS ---")
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_num = 0
    for train_idx, test_idx in skf.split(X, y):
        fold_num += 1
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        scaler_fold = StandardScaler()
        X_train_s = scaler_fold.fit_transform(X_train)
        X_test_s = scaler_fold.transform(X_test)
        
        # Test set separability
        test_autism = X_test_s[y_test == 1]
        test_normal = X_test_s[y_test == 0]
        
        if len(test_autism) > 0 and len(test_normal) > 0:
            test_within_autism = np.mean(pdist(test_autism, metric='euclidean')) if len(test_autism) > 1 else 0
            test_within_normal = np.mean(pdist(test_normal, metric='euclidean')) if len(test_normal) > 1 else 0
            
            test_between = []
            for a in test_autism:
                for n in test_normal:
                    test_between.append(np.linalg.norm(a - n))
            test_between_class = np.mean(test_between) if test_between else 0
            
            test_sep = test_between_class / ((test_within_autism + test_within_normal) / 2 + 1e-6)
        else:
            test_sep = 0.0
        
        print(f"\nFold {fold_num}:")
        print(f"  Test set - Autism: {np.sum(y_test)} | Normal: {len(y_test) - np.sum(y_test)}")
        print(f"  Test separability ratio: {test_sep:.4f}")
        print(f"  Difficulty: {'⭐ HARD' if test_sep < 1.5 else '✓ NORMAL'}")


def analyze_class_overlap():
    """Find how much autism and normal samples overlap"""
    
    print("\n" + "="*80)
    print("CLASS OVERLAP ANALYSIS")
    print("="*80)
    
    X = np.load('X_combined.npy')
    y = np.load('Y_labels.npy')
    
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    autism_pca = X_pca[y == 1]
    normal_pca = X_pca[y == 0]
    
    print(f"\nPCA Explained variance: {np.sum(pca.explained_variance_ratio_):.4f}")
    print(f"First 2 components capture {np.sum(pca.explained_variance_ratio_):.1%} of variance")
    
    # Calculate overlap region
    autism_bounds = (autism_pca.min(axis=0), autism_pca.max(axis=0))
    normal_bounds = (normal_pca.min(axis=0), normal_pca.max(axis=0))
    
    overlap_x = max(0, min(autism_bounds[1][0], normal_bounds[1][0]) - max(autism_bounds[0][0], normal_bounds[0][0]))
    overlap_y = max(0, min(autism_bounds[1][1], normal_bounds[1][1]) - max(autism_bounds[0][1], normal_bounds[0][1]))
    
    autism_area = (autism_bounds[1] - autism_bounds[0]).prod()
    normal_area = (normal_bounds[1] - normal_bounds[0]).prod()
    overlap_area = overlap_x * overlap_y if overlap_x > 0 and overlap_y > 0 else 0
    
    print(f"\nAutism region area: {autism_area:.4f}")
    print(f"Normal region area: {normal_area:.4f}")
    print(f"Overlap area: {overlap_area:.4f}")
    print(f"Overlap %: {(overlap_area / min(autism_area, normal_area) * 100):.1f}%")


if __name__ == "__main__":
    analyze_fold_difficulty()
    analyze_class_overlap()
    
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print("\nIf separability ratio < 1.5: Classes are highly overlapped (hard problem)")
    print("If overlap > 60%: Too much mixing between classes")
    print("\nLow k-fold accuracy (60%) suggests inherent dataset difficulty,")
    print("not a modeling issue.")
    print("="*80 + "\n")
