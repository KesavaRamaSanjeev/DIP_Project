"""
STEP 4: STATISTICAL SIGNIFICANCE TESTING
=========================================
Test if improvement from 90.08% → 90.99% is statistically significant.
Uses McNemar's test and confidence intervals.

Output: statistical_tests.json
"""

import os
import sys
import numpy as np
import json
from scipy.stats import binom
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

print("\n" + "="*70)
print("STATISTICAL SIGNIFICANCE TESTING")
print("="*70 + "\n")

# Load features
X_complete = np.load('X_novelty3.npy')
X_baseline = X_complete[:, :136]  # First 136 features (baseline)
Y_labels = np.load('Y_novelty3.npy')

print("[*] Comparing Baseline (136 features) vs Full Model (146 features)\n")

baseline_predictions_all = []
full_predictions_all = []

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_complete, Y_labels)):
    print(f"Fold {fold_idx + 1}/5...", end=" ")
    
    # Baseline model
    X_train_base = X_baseline[train_idx]
    X_test_base = X_baseline[test_idx]
    
    # Full model
    X_train_full = X_complete[train_idx]
    X_test_full = X_complete[test_idx]
    
    y_train = Y_labels[train_idx]
    y_test = Y_labels[test_idx]
    
    # Augment
    noise_base = np.random.normal(0, X_train_base.std(axis=0) * 0.05, X_train_base.shape)
    X_train_base_aug = np.vstack([X_train_base, X_train_base + noise_base])
    
    noise_full = np.random.normal(0, X_train_full.std(axis=0) * 0.05, X_train_full.shape)
    X_train_full_aug = np.vstack([X_train_full, X_train_full + noise_full])
    
    y_train_aug = np.hstack([y_train, y_train])
    
    # Standardize baseline
    scaler_base = StandardScaler()
    X_train_base_aug = scaler_base.fit_transform(X_train_base_aug)
    X_test_base = scaler_base.transform(X_test_base)
    
    # Standardize full
    scaler_full = StandardScaler()
    X_train_full_aug = scaler_full.fit_transform(X_train_full_aug)
    X_test_full = scaler_full.transform(X_test_full)
    
    # Baseline ensemble
    svm_base = SVC(C=100, kernel='rbf', gamma='scale', probability=True, random_state=42)
    rf_base = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    svm_base.fit(X_train_base_aug, y_train_aug)
    rf_base.fit(X_train_base_aug, y_train_aug)
    baseline_proba = (svm_base.predict_proba(X_test_base)[:, 1] + rf_base.predict_proba(X_test_base)[:, 1]) / 2
    baseline_pred = (baseline_proba > 0.5).astype(int)
    
    # Full ensemble
    svm_full = SVC(C=100, kernel='rbf', gamma='scale', probability=True, random_state=42)
    rf_full = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    svm_full.fit(X_train_full_aug, y_train_aug)
    rf_full.fit(X_train_full_aug, y_train_aug)
    full_proba = (svm_full.predict_proba(X_test_full)[:, 1] + rf_full.predict_proba(X_test_full)[:, 1]) / 2
    full_pred = (full_proba > 0.5).astype(int)
    
    for i in range(len(y_test)):
        baseline_predictions_all.append({'actual': int(y_test[i]), 'pred': int(baseline_pred[i])})
        full_predictions_all.append({'actual': int(y_test[i]), 'pred': int(full_pred[i])})
    
    print("✓")

# Calculate accuracies
baseline_acc = sum(1 for p in baseline_predictions_all if p['actual'] == p['pred']) / len(baseline_predictions_all)
full_acc = sum(1 for p in full_predictions_all if p['actual'] == p['pred']) / len(full_predictions_all)
improvement = (full_acc - baseline_acc) * 100

print(f"\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"\nBaseline Accuracy (136 features): {baseline_acc*100:.2f}%")
print(f"Full Model Accuracy (146 features): {full_acc*100:.2f}%")
print(f"Absolute Improvement: +{improvement:.2f}%")
print(f"Relative Improvement: +{(improvement/baseline_acc)*100:.2f}%")

# McNemar's Test
# Count agreements and disagreements
n_total = len(baseline_predictions_all)
both_correct = sum(1 for b, f in zip(baseline_predictions_all, full_predictions_all) 
                   if b['actual'] == b['pred'] and f['actual'] == f['pred'])
both_wrong = sum(1 for b, f in zip(baseline_predictions_all, full_predictions_all) 
                 if (b['actual'] != b['pred']) and (f['actual'] != f['pred']))
baseline_correct_full_wrong = sum(1 for b, f in zip(baseline_predictions_all, full_predictions_all) 
                                  if (b['actual'] == b['pred']) and (f['actual'] != f['pred']))
baseline_wrong_full_correct = sum(1 for b, f in zip(baseline_predictions_all, full_predictions_all) 
                                  if (b['actual'] != b['pred']) and (f['actual'] == f['pred']))

print(f"\nMcNemar's Test (paired comparisons):")
print(f"  Both correct: {both_correct}")
print(f"  Both wrong: {both_wrong}")
print(f"  Baseline correct, Full wrong: {baseline_correct_full_wrong}")
print(f"  Baseline wrong, Full correct: {baseline_wrong_full_correct}")

n_b = baseline_correct_full_wrong
n_f = baseline_wrong_full_correct
n_total_disagreement = n_b + n_f

if n_total_disagreement > 0:
    # McNemar statistic
    chi2 = ((n_b - n_f) ** 2) / (n_b + n_f)
    
    # Exact binomial test (more conservative)
    p_value = 2 * min(binom.cdf(min(n_b, n_f), n_total_disagreement, 0.5), 
                      1 - binom.cdf(min(n_b, n_f) - 1, n_total_disagreement, 0.5))
    
    print(f"\n  McNemar chi² statistic: {chi2:.4f}")
    print(f"  P-value (2-tailed): {p_value:.4f}")
    
    if p_value < 0.05:
        print(f"  ✓ Difference is STATISTICALLY SIGNIFICANT at α=0.05")
    else:
        print(f"  ✗ Difference is NOT statistically significant at α=0.05")
else:
    print(f"\n  No disagreements found between models")
    p_value = None

# Confidence intervals on accuracies (Wilson score interval)
def wilson_ci(successes, n, confidence=0.95):
    from scipy.stats import norm
    z = norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / n
    
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2*n)) / denom
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4*n**2)) / denom
    
    return center - margin, center + margin

n_correct_base = sum(1 for p in baseline_predictions_all if p['actual'] == p['pred'])
n_correct_full = sum(1 for p in full_predictions_all if p['actual'] == p['pred'])

ci_base = wilson_ci(n_correct_base, n_total)
ci_full = wilson_ci(n_correct_full, n_total)

print(f"\n95% Confidence Intervals:")
print(f"  Baseline: [{ci_base[0]*100:.2f}%, {ci_base[1]*100:.2f}%]")
print(f"  Full Model: [{ci_full[0]*100:.2f}%, {ci_full[1]*100:.2f}%]")

# Save detailed results
results = {
    'accuracies': {
        'baseline_136': float(baseline_acc),
        'full_146': float(full_acc),
        'improvement_absolute': float(improvement),
        'improvement_relative': float((improvement/baseline_acc)*100)
    },
    'paired_analysis': {
        'both_correct': both_correct,
        'both_wrong': both_wrong,
        'baseline_correct_full_wrong': baseline_correct_full_wrong,
        'baseline_wrong_full_correct': baseline_wrong_full_correct
    },
    'mcnemar_test': {
        'chi_squared': float(chi2) if n_total_disagreement > 0 else None,
        'p_value': float(p_value) if p_value is not None else None,
        'significant_at_0.05': bool(p_value < 0.05) if p_value is not None else None
    },
    'confidence_intervals': {
        'baseline_ci': [float(ci_base[0]), float(ci_base[1])],
        'full_ci': [float(ci_full[0]), float(ci_full[1])]
    }
}

with open('statistical_tests.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved: statistical_tests.json\n")
