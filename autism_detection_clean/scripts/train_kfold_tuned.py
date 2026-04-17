"""
K-Fold with Data Augmentation + Hyperparameter Tuning
=====================================================
Augment the small dataset + optimize SVM hyperparameters
"""

import sys
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


def augment_features(X_train, y_train, augmentation_factor=3):
    """
    Augment training data with small perturbations
    This helps small datasets generalize better
    """
    X_augmented = [X_train]
    y_augmented = [y_train]
    
    for _ in range(augmentation_factor - 1):
        # Add Gaussian noise (small)
        noise = np.random.normal(0, 0.05 * np.std(X_train, axis=0), X_train.shape)
        X_augmented.append(X_train + noise)
        y_augmented.append(y_train)
    
    X_aug = np.vstack(X_augmented)
    y_aug = np.concatenate(y_augmented)
    
    return X_aug, y_aug


def main():
    print("\n" + "="*80)
    print("AUTISM DETECTION: DATA AUGMENTATION + HYPERPARAMETER TUNING")
    print("="*80)
    print("Strategy: Augment small dataset + optimize SVM with GridSearch")
    print("="*80)
    
    device = 'cuda'
    print(f"Device: {device}\n")
    
    # Load features
    print("Loading augmented features...")
    X = np.load('X_combined.npy')
    y = np.load('Y_labels.npy')
    print(f"✓ Features loaded: {X.shape}")
    
    # K-fold setup
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    all_results = {
        'rf': [],
        'svm_tuned': [],
        'ensemble': []
    }
    
    fold_num = 0
    print("\n" + "="*80)
    print("TRAINING K-FOLDS WITH AUGMENTATION")
    print("="*80)
    
    for train_idx, test_idx in skf.split(X, y):
        fold_num += 1
        print(f"\n--- FOLD {fold_num}/{n_splits} ---")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Augment training data
        X_train_aug, y_train_aug = augment_features(X_train, y_train, augmentation_factor=2)
        
        print(f"Original train: {len(y_train)} -> Augmented: {len(y_train_aug)} | Test: {len(y_test)}")
        
        # Normalize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_aug)
        X_test_scaled = scaler.transform(X_test)
        
        # ===== Random Forest =====
        print("  Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=200, max_depth=16, 
                                     min_samples_split=2, random_state=42, n_jobs=-1)
        rf.fit(X_train_scaled, y_train_aug)
        rf_pred = rf.predict(X_test_scaled)
        rf_acc = accuracy_score(y_test, rf_pred)
        all_results['rf'].append(rf_acc)
        print(f"    ✓ RF: {rf_acc:.4f}")
        
        # ===== SVM with GridSearch Tuning =====
        print("  Tuning SVM hyperparameters...")
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf', 'poly']
        }
        
        svm = SVC(random_state=42)
        grid_search = GridSearchCV(svm, param_grid, cv=3, n_jobs=-1, verbose=0)
        grid_search.fit(X_train_scaled, y_train_aug)
        
        best_svm = grid_search.best_estimator_
        svm_pred = best_svm.predict(X_test_scaled)
        svm_acc = accuracy_score(y_test, svm_pred)
        all_results['svm_tuned'].append(svm_acc)
        print(f"    ✓ SVM tuned: {svm_acc:.4f} (best params: C={grid_search.best_params_['C']}, kernel={grid_search.best_params_['kernel']})")
        
        # ===== Ensemble =====
        print("  Computing ensemble...")
        ensemble_pred = (rf_pred.astype(float) + svm_pred.astype(float)) / 2
        ensemble_pred = (ensemble_pred > 0.5).astype(int)
        ensemble_acc = accuracy_score(y_test, ensemble_pred)
        all_results['ensemble'].append(ensemble_acc)
        print(f"    ✓ Ensemble: {ensemble_acc:.4f}")
    
    # Results
    print("\n" + "="*80)
    print("FINAL RESULTS: 5-FOLD WITH AUGMENTATION & TUNING")
    print("="*80)
    
    summary = {}
    for model_name in ['rf', 'svm_tuned', 'ensemble']:
        accs = all_results[model_name]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        summary[model_name] = (mean_acc, std_acc)
        
        names = {'rf': 'Random Forest', 'svm_tuned': 'SVM Tuned', 'ensemble': 'Ensemble (RF+SVM)'}
        print(f"\n{names[model_name]}:")
        print(f"  Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"  Folds: {' | '.join([f'{x:.4f}' for x in accs])}")
    
    # Best
    print("\n" + "="*80)
    best_model = max(summary.items(), key=lambda x: x[1][0])
    best_name = {'rf': 'Random Forest', 'svm_tuned': 'SVM Tuned', 'ensemble': 'Ensemble'}[best_model[0]]
    
    print(f"★ {best_name}: {best_model[1][0]:.4f} ± {best_model[1][1]:.4f} ★")
    
    if best_model[1][0] >= 0.81:
        print("✅ TARGET ACHIEVED: Accuracy ≥ 81%")
    else:
        gap = 0.81 - best_model[1][0]
        print(f"⚠️  Current: {best_model[1][0]:.4f} | Gap to 81%: {gap:.4f}")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
