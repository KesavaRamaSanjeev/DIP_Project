"""
Train with NOVELTY 1 + 2 (141 features)
Baseline: 136 features → 90.08%
Novelty 1: +1 Symmetry → 137
Novelty 2: +4 Entropy → 141
"""

import numpy as np
import json
import os
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

def augment_data(X, Y, augmentation_factor=2):
    """Gaussian augmentation"""
    if augmentation_factor <= 1:
        return X, Y
    X_augmented = [X]
    Y_augmented = [Y]
    for _ in range(augmentation_factor - 1):
        noise = np.random.normal(0, 0.02, X.shape)
        X_augmented.append(X + noise)
        Y_augmented.append(Y)
    return np.vstack(X_augmented), np.hstack(Y_augmented)

def train_novelty2():
    print("\n" + "=" * 80)
    print("TRAINING WITH NOVELTY 1 + 2: SYMMETRY + MOTION ENTROPY")
    print("=" * 80)
    print("Baseline: 136 features → 90.08% accuracy")
    print("Novelty 1: +1 symmetry → 137")
    print("Novelty 2: +4 entropy → 141 total")
    print("=" * 80)
    
    if not os.path.exists('X_novelty2.npy'):
        print("ERROR: X_novelty2.npy not found!")
        print("Run: python scripts/add_novelty2_entropy.py")
        return
    
    X = np.load('X_novelty2.npy')
    Y = np.load('Y_novelty2.npy')
    
    print(f"\n✓ Loaded: X shape {X.shape}, Y shape {Y.shape}")
    print(f"  Autism: {(Y == 1).sum()}, Normal: {(Y == 0).sum()}")
    
    X_aug, Y_aug = augment_data(X, Y, augmentation_factor=2)
    print(f"✓ Augmented: X shape {X_aug.shape}")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {
        'novelties': 'Novelty 1 (Symmetry) + Novelty 2 (Motion Entropy)',
        'features': 141,
        'baseline_features': 136,
        'new_features': 5,
        'baseline_accuracy': 90.08,
        'novelty1_accuracy': None,  # Will be filled from prev results if available
        'folds': {}
    }
    
    fold_accs = []
    
    print(f"\n{'='*80}")
    print("5-FOLD CROSS-VALIDATION")
    print(f"{'='*80}")
    
    fold_num = 1
    for train_idx, test_idx in skf.split(X_aug, Y_aug):
        print(f"\nFOLD {fold_num}:")
        
        X_train, X_test = X_aug[train_idx], X_aug[test_idx]
        Y_train, Y_test = Y_aug[train_idx], Y_aug[test_idx]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        svm = GridSearchCV(SVC(probability=True), 
                          {'C': [1, 10, 100], 'kernel': ['rbf', 'linear'], 'gamma': ['scale', 'auto']},
                          cv=3, n_jobs=-1, verbose=0)
        svm.fit(X_train_s, Y_train)
        svm_pred = svm.predict_proba(X_test_s)
        
        rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X_train_s, Y_train)
        rf_pred = rf.predict_proba(X_test_s)
        
        ensemble_pred = ((svm_pred + rf_pred) / 2)[:, 1] >= 0.5
        ensemble_pred = ensemble_pred.astype(int)
        
        acc = accuracy_score(Y_test, ensemble_pred)
        prec = precision_score(Y_test, ensemble_pred, zero_division=0)
        rec = recall_score(Y_test, ensemble_pred, zero_division=0)
        f1 = f1_score(Y_test, ensemble_pred, zero_division=0)
        
        fold_accs.append(acc)
        
        print(f"  Accuracy:  {acc:.4f} ({(acc-0.9008)*100:+.2f}% vs baseline)")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1:        {f1:.4f}")
        
        results['folds'][f'fold_{fold_num}'] = {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1)
        }
        fold_num += 1
    
    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    
    results['summary'] = {
        'mean_accuracy': float(mean_acc),
        'std_accuracy': float(std_acc),
        'fold_accuracies': [float(x) for x in fold_accs],
        'improvement_vs_baseline': float((mean_acc - 0.9008) * 100)
    }
    
    print(f"\n{'='*80}")
    print("SUMMARY - NOVELTY 1 + 2")
    print(f"{'='*80}")
    print(f"Mean Accuracy:  {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Baseline:       90.08%")
    print(f"Improvement:    {(mean_acc - 0.9008)*100:+.2f}%")
    print(f"Per-fold:       {[f'{x*100:.2f}%' for x in fold_accs]}")
    print(f"{'='*80}\n")
    
    with open('kfold_results_novelty2.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved: kfold_results_novelty2.json\n")

if __name__ == '__main__':
    train_novelty2()
