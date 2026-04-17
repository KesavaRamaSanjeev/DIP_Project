"""
K-Fold Cross-Validation with Autism-Specific Features
=======================================================
Uses: LSTM (128) + Motion (8) + Autism Behavioral (13) = 149 features
Dataset: 117 clean samples, 5-fold stratified validation
"""

import sys
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except:
    HAS_XGB = False


def load_augmented_features():
    """Load augmented feature set with autism-specific metrics"""
    print("Loading augmented features...")
    try:
        X_combined = np.load('X_combined.npy')
        y = np.load('Y_labels.npy')
        print(f"✓ Augmented features loaded: {X_combined.shape}")
    except:
        print("⚠ X_combined.npy not found, using standard features")
        X_lstm = np.load('X_lstm.npy')
        X_motion = np.load('X_motion.npy')
        X_autism = np.load('X_autism.npy')
        X_combined = np.concatenate([X_lstm, X_motion, X_autism], axis=1)
        y = np.load('Y_labels.npy')
    
    return X_combined, y


def main():
    print("\n" + "="*80)
    print("AUTISM DETECTION: AUGMENTED FEATURES K-FOLD VALIDATION")
    print("="*80)
    print("Features: LSTM (128) + Motion (8) + Autism Behavioral (13) = 149 total")
    print("Dataset: 117 clean samples, 5-fold stratified cross-validation")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load features
    X, y = load_augmented_features()
    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    # K-fold setup
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    all_results = {
        'rf': [],
        'xgb': [],
        'svm': [],
        'ensemble': []
    }
    
    fold_num = 0
    print("\n" + "="*80)
    print("TRAINING K-FOLDS")
    print("="*80)
    
    for train_idx, test_idx in skf.split(X, y):
        fold_num += 1
        print(f"\n--- FOLD {fold_num}/{n_splits} ---")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        print(f"Train: {len(y_train)} | Test: {len(y_test)}")
        print(f"Classes - Train: {np.sum(y_train)} autism, {len(y_train) - np.sum(y_train)} normal")
        
        # Normalize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Random Forest
        print("  Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=150, max_depth=15, min_samples_split=2, 
                                     random_state=42, n_jobs=-1)
        rf.fit(X_train_scaled, y_train)
        rf_pred = rf.predict(X_test_scaled)
        rf_acc = accuracy_score(y_test, rf_pred)
        all_results['rf'].append(rf_acc)
        print(f"    ✓ RF: {rf_acc:.4f}")
        
        # XGBoost
        if HAS_XGB:
            print("  Training XGBoost...")
            xgb = XGBClassifier(n_estimators=150, max_depth=8, learning_rate=0.1, 
                                random_state=42, verbosity=0)
            xgb.fit(X_train_scaled, y_train)
            xgb_pred = xgb.predict(X_test_scaled)
            xgb_acc = accuracy_score(y_test, xgb_pred)
            all_results['xgb'].append(xgb_acc)
            print(f"    ✓ XGB: {xgb_acc:.4f}")
        else:
            all_results['xgb'].append(0.0)
        
        # SVM
        print("  Training SVM (RBF)...")
        svm = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
        svm.fit(X_train_scaled, y_train)
        svm_pred = svm.predict(X_test_scaled)
        svm_acc = accuracy_score(y_test, svm_pred)
        all_results['svm'].append(svm_acc)
        print(f"    ✓ SVM: {svm_acc:.4f}")
        
        # Ensemble
        if HAS_XGB:
            ensemble_pred = (rf_pred.astype(float) + xgb_pred.astype(float) + svm_pred.astype(float)) / 3
        else:
            ensemble_pred = (rf_pred.astype(float) + svm_pred.astype(float)) / 2
        
        ensemble_pred = (ensemble_pred > 0.5).astype(int)
        ensemble_acc = accuracy_score(y_test, ensemble_pred)
        all_results['ensemble'].append(ensemble_acc)
        print(f"    ✓ Ensemble: {ensemble_acc:.4f}")
    
    # Print summary
    print("\n" + "="*80)
    print("FINAL RESULTS: 5-FOLD STRATIFIED CROSS-VALIDATION")
    print("="*80)
    
    summary = {}
    for model_name in ['rf', 'xgb', 'svm', 'ensemble']:
        accs = [a for a in all_results[model_name] if a > 0]
        if accs:
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            summary[model_name] = (mean_acc, std_acc)
            
            names = {'rf': 'Random Forest', 'xgb': 'XGBoost', 'svm': 'SVM (RBF)', 'ensemble': 'Ensemble'}
            print(f"\n{names[model_name]}:")
            print(f"  Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
            print(f"  Folds: {' | '.join([f'{x:.4f}' for x in accs])}")
    
    # Best model
    print("\n" + "="*80)
    best_model = max(summary.items(), key=lambda x: x[1][0])
    best_name = {'rf': 'Random Forest', 'xgb': 'XGBoost', 'svm': 'SVM', 'ensemble': 'Ensemble'}[best_model[0]]
    
    print(f"★ {best_name}: {best_model[1][0]:.4f} ± {best_model[1][1]:.4f} ★")
    
    if best_model[1][0] >= 0.81:
        print("✅ TARGET ACHIEVED: Accuracy ≥ 81%")
    else:
        print(f"⚠️  Current: {best_model[1][0]:.4f} (Target: 0.81)")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
