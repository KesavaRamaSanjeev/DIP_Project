"""
STEP 1: ABLATION STUDY
======================
Test how important each novelty feature is by removing them one-by-one.
This proves each feature contributes to the 90.99% accuracy.

Output: ablation_results.json with detailed breakdown
"""

import os
import sys
import numpy as np
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

print("\n" + "="*70)
print("ABLATION STUDY: Importance of Each Novelty Feature")
print("="*70 + "\n")

# Load Phase 1 complete features
X_complete = np.load('X_novelty3.npy')  # All 146 features
Y_labels = np.load('Y_novelty3.npy')

print(f"[*] Loaded features: {X_complete.shape}")
print(f"    Total samples: {len(Y_labels)}")
print(f"    Autism cases: {np.sum(Y_labels == 1)}")
print(f"    Normal cases: {np.sum(Y_labels == 0)}\n")

# Feature indices
# Baseline: 0-135 (136 dims)
# Novelty 1 (Symmetry): 136 (1 dim)
# Novelty 2 (Entropy): 137-140 (4 dims)
# Novelty 3 (Jerk): 141-145 (5 dims)

ablation_configs = {
    'Full Model (All 146)': np.arange(146),
    'Without Symmetry (145)': np.concatenate([np.arange(136), np.arange(137, 146)]),
    'Without Entropy (142)': np.concatenate([np.arange(137), np.arange(141, 146)]),
    'Without Jerk (141)': np.arange(141),
    'Baseline Only (136)': np.arange(136),
    'Only Symmetry (1)': np.array([136]),
    'Only Entropy (4)': np.arange(137, 141),
    'Only Jerk (5)': np.arange(141, 146),
}

results = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for config_name, feature_indices in ablation_configs.items():
    print(f"Testing: {config_name}")
    
    X_subset = X_complete[:, feature_indices]
    fold_accuracies = []
    fold_metrics = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_subset, Y_labels)):
        X_train, X_test = X_subset[train_idx], X_subset[test_idx]
        y_train, y_test = Y_labels[train_idx], Y_labels[test_idx]
        
        # Augment training data (same as original)
        noise = np.random.normal(0, X_train.std(axis=0) * 0.05, X_train.shape)
        X_train_aug = np.vstack([X_train, X_train + noise])
        y_train_aug = np.hstack([y_train, y_train])
        
        # Standardize
        scaler = StandardScaler()
        X_train_aug = scaler.fit_transform(X_train_aug)
        X_test = scaler.transform(X_test)
        
        # Train ensemble
        svm = SVC(C=100, kernel='rbf', gamma='scale', probability=True, random_state=42)
        rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        
        svm.fit(X_train_aug, y_train_aug)
        rf.fit(X_train_aug, y_train_aug)
        
        # Soft voting
        svm_proba = svm.predict_proba(X_test)[:, 1]
        rf_proba = rf.predict_proba(X_test)[:, 1]
        ensemble_proba = (svm_proba + rf_proba) / 2
        pred = (ensemble_proba > 0.5).astype(int)
        
        # Metrics
        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)
        
        fold_accuracies.append(acc)
        fold_metrics.append({'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1})
        
        print(f"  Fold {fold_idx+1}: {acc*100:.2f}%", end=" ")
    
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    
    results[config_name] = {
        'fold_accuracies': [float(x) for x in fold_accuracies],
        'mean_accuracy': float(mean_acc),
        'std_dev': float(std_acc),
        'n_features': len(feature_indices),
        'metrics_per_fold': fold_metrics
    }
    
    print(f"→ Mean: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%\n")

# Save results
with open('ablation_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*70)
print("ABLATION STUDY SUMMARY")
print("="*70)

baseline = results['Baseline Only (136)']['mean_accuracy']
full = results['Full Model (All 146)']['mean_accuracy']
improvement = (full - baseline) * 100

print(f"\nBaseline (136 features):     {baseline*100:.2f}%")
print(f"Full Model (146 features):   {full*100:.2f}%")
print(f"Improvement:                 +{improvement:.2f}%")

print(f"\nIndividual Feature Impact:")
print(f"  Symmetry alone:  {results['Only Symmetry (1)']['mean_accuracy']*100:.2f}%")
print(f"  Entropy alone:   {results['Only Entropy (4)']['mean_accuracy']*100:.2f}%")
print(f"  Jerk alone:      {results['Only Jerk (5)']['mean_accuracy']*100:.2f}%")

print(f"\nRemoval Impact (drop when removed):")
print(f"  Remove Symmetry: {(full - results['Without Symmetry (145)']['mean_accuracy'])*100:.2f}%")
print(f"  Remove Entropy:  {(full - results['Without Entropy (142)']['mean_accuracy'])*100:.2f}%")
print(f"  Remove Jerk:     {(full - results['Without Jerk (141)']['mean_accuracy'])*100:.2f}%")

print(f"\n✅ Results saved: ablation_results.json\n")
