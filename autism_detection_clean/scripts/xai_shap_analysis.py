"""
XAI: SHAP Feature Importance Analysis
======================================
Analyzes which of the 136 features are most important for autism detection.

IMPORTANT: This script does NOT modify the baseline model.
It analyzes feature importance from the trained ensemble.
"""

import os
import sys
import numpy as np
import json
import warnings
from datetime import datetime
import shap
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
warnings.filterwarnings('ignore')

# Feature names mapping
FEATURE_NAMES = {
    'lstm_forward': {
        'left_hand': list(range(0, 4)),
        'left_elbow': list(range(4, 8)),
        'right_hand': list(range(8, 12)),
        'right_elbow': list(range(12, 16)),
        'left_knee': list(range(16, 20)),
        'right_knee': list(range(20, 24)),
        'head': list(range(24, 28)),
        'torso': list(range(28, 32)),
        'global_context': list(range(32, 64))
    },
    'lstm_backward': {
        'left_hand': list(range(64, 68)),
        'left_elbow': list(range(68, 72)),
        'right_hand': list(range(72, 76)),
        'right_elbow': list(range(76, 80)),
        'left_knee': list(range(80, 84)),
        'right_knee': list(range(84, 88)),
        'head': list(range(88, 92)),
        'torso': list(range(92, 96)),
        'global_context': list(range(96, 128))
    },
    'motion_features': {
        128: 'Left_Hand_Jerk',
        129: 'Right_Hand_Jerk',
        130: 'Wrist_Asymmetry',
        131: 'Head_Stability',
        132: 'Repetition_Index',
        133: 'Posture_Rigidity',
        134: 'Limb_Synchronization',
        135: 'Motion_Speed_Variability'
    }
}


def get_feature_name(dim):
    """Get human-readable name for dimension"""
    if dim in FEATURE_NAMES['motion_features']:
        return FEATURE_NAMES['motion_features'][dim]
    
    if dim < 64:
        for body_part, dims in FEATURE_NAMES['lstm_forward'].items():
            if dim in dims:
                idx = dims.index(dim)
                types = ['X_temporal', 'Y_temporal', 'velocity', 'acceleration']
                return f'{body_part}_forward_{types[idx % 4]}'
    else:
        dim_offset = dim - 64
        for body_part, dims in FEATURE_NAMES['lstm_backward'].items():
            if dim_offset in dims:
                idx = dims.index(dim_offset)
                types = ['X_temporal', 'Y_temporal', 'velocity', 'acceleration']
                return f'{body_part}_backward_{types[idx % 4]}'
    
    return f'Dim_{dim}'


def train_rf_for_shap(X_train, y_train):
    """Train Random Forest for SHAP analysis"""
    print("[*] Training Random Forest for SHAP analysis...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, max_depth=20)
    rf.fit(X_train, y_train)
    print(f"    [OK] RF trained. Training accuracy: {rf.score(X_train, y_train):.4f}")
    return rf


def calculate_shap_values(rf_model, X_data, X_background=None):
    """Calculate SHAP values using TreeExplainer"""
    print("[*] Calculating SHAP values...")
    
    # Use TreeExplainer for Random Forest
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_data)
    
    # For binary classification, SHAP returns [shap_class_0, shap_class_1]
    # We're interested in class 1 (Autism) - take second array
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Class 1 (Autism)
    elif len(shap_values.shape) == 3:
        # Sometimes returns (samples, features, classes) - extract class 1
        shap_values = shap_values[:, :, 1]
    
    print(f"    [OK] SHAP values computed. Shape: {shap_values.shape}")
    return shap_values, explainer


def compute_feature_importance(shap_values):
    """Compute mean absolute SHAP values per feature"""
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    rankings = np.argsort(mean_abs_shap)[::-1]  # Descending order
    
    # Convert to native Python int to avoid numpy array issues
    rankings = [int(r) for r in rankings]
    
    feature_importance = {
        'mean_abs_shap': mean_abs_shap,
        'rankings': rankings
    }
    return feature_importance


def shap_analysis_cleanest():
    """
    SHAP Feature Importance Analysis on Cleanest Dataset
    Does NOT use the final trained model (to avoid retraining)
    Instead: Trains a fresh RF on test data, computes SHAP
    """
    
    print(f"\n{'='*70}")
    print(f"SHAP FEATURE IMPORTANCE ANALYSIS - Cleanest Dataset")
    print(f"{'='*70}\n")
    
    # Load data
    print("[*] Loading features...")
    feature_file = 'X_combined_cleanest.npy'
    label_file = 'Y_labels_cleanest.npy'
    
    if not os.path.exists(feature_file):
        print(f"[ERROR] {feature_file} not found!")
        return
    
    X_combined = np.load(feature_file)
    Y_labels = np.load(label_file)
    
    print(f"  [OK] Loaded {feature_file}: {X_combined.shape}")
    print(f"  [OK] Loaded {label_file}: {Y_labels.shape}\n")
    
    # Use first fold for SHAP analysis
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_num = 0
    
    all_fold_results = []
    
    for train_idx, test_idx in skf.split(X_combined, Y_labels):
        fold_num += 1
        print(f"[FOLD {fold_num}]")
        print(f"{'─'*70}")
        
        X_train, X_test = X_combined[train_idx], X_combined[test_idx]
        y_train, y_test = Y_labels[train_idx], Y_labels[test_idx]
        
        print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
        
        # Train RF
        rf_model = train_rf_for_shap(X_train, y_train)
        
        # Calculate SHAP on TEST set
        shap_values, explainer = calculate_shap_values(rf_model, X_test)
        
        # Compute importance
        feature_imp = compute_feature_importance(shap_values)
        mean_abs_shap = feature_imp['mean_abs_shap']
        rankings = feature_imp['rankings']
        
        # Top 15 features
        top_15_dims = rankings[:15]
        
        print(f"\n  Top 15 Most Important Features:")
        print(f"  {'Rank':<6} {'Dim':<6} {'SHAP':<12} {'Feature Name':<40} {'Body Part'}")
        print(f"  {'-'*110}")
        
        fold_result = {
            'fold': fold_num,
            'top_15_features': [],
            'all_feature_importance': mean_abs_shap.tolist(),
            'model_test_accuracy': rf_model.score(X_test, y_test)
        }
        
        for rank, dim in enumerate(top_15_dims, 1):
            shap_importance = mean_abs_shap[dim]
            feature_name = get_feature_name(dim)
            
            # Infer body part
            if dim in FEATURE_NAMES['motion_features']:
                body_part = 'MOTION_ANALYSIS'
            elif dim < 64:
                for bp, dims in FEATURE_NAMES['lstm_forward'].items():
                    if dim in dims:
                        body_part = f'{bp.upper()}_FWD'
                        break
            else:
                for bp, dims in FEATURE_NAMES['lstm_backward'].items():
                    if (dim - 64) in dims:
                        body_part = f'{bp.upper()}_BWD'
                        break
            
            print(f"  {rank:<6} {dim:<6} {shap_importance:<12.6f} {feature_name:<40} {body_part}")
            
            fold_result['top_15_features'].append({
                'rank': rank,
                'dimension': int(dim),
                'shap_importance': float(shap_importance),
                'feature_name': feature_name,
                'body_part': body_part
            })
        
        all_fold_results.append(fold_result)
        print(f"\n  Model Test Accuracy: {rf_model.score(X_test, y_test):.4f}\n")
        
        # Only analyze first fold for speed
        break
    
    # Save results
    output_file = 'shap_feature_importance.json'
    with open(output_file, 'w') as f:
        json.dump(all_fold_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"[OK] Results saved to: {output_file}")
    print(f"{'='*70}\n")
    
    return all_fold_results


if __name__ == '__main__':
    try:
        results = shap_analysis_cleanest()
        print("\n[SUCCESS] SHAP analysis completed!")
        print("\nNext steps:")
        print("  1. Review SHAP results in shap_feature_importance.json")
        print("  2. Look up important features in FEATURE_MAPPING.md")
        print("  3. Run: python scripts/xai_clinical_heatmap.py")
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
