"""
STEP 3: ERROR ANALYSIS
======================
Analyze which cases your model gets wrong and identify patterns.
This helps understand model limitations and edge cases.

Output: error_analysis.json + detailed breakdown
"""

import os
import sys
import numpy as np
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

print("\n" + "="*70)
print("ERROR ANALYSIS")
print("="*70 + "\n")

# Load Phase 1 complete features
X_complete = np.load('X_novelty3.npy')
Y_labels = np.load('Y_novelty3.npy')

print(f"[*] Loaded features: {X_complete.shape}\n")

all_errors = []
all_predictions = []
all_confidences = []
fold_cm_list = []

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_complete, Y_labels)):
    print(f"Analyzing Fold {fold_idx + 1}/5...", end=" ")
    
    X_train, X_test = X_complete[train_idx], X_complete[test_idx]
    y_train, y_test = Y_labels[train_idx], Y_labels[test_idx]
    
    # Augment
    noise = np.random.normal(0, X_train.std(axis=0) * 0.05, X_train.shape)
    X_train_aug = np.vstack([X_train, X_train + noise])
    y_train_aug = np.hstack([y_train, y_train])
    
    # Standardize
    scaler = StandardScaler()
    X_train_aug = scaler.fit_transform(X_train_aug)
    X_test_scaled = scaler.transform(X_test)
    
    # Train ensemble
    svm = SVC(C=100, kernel='rbf', gamma='scale', probability=True, random_state=42)
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    
    svm.fit(X_train_aug, y_train_aug)
    rf.fit(X_train_aug, y_train_aug)
    
    # Soft voting
    svm_proba = svm.predict_proba(X_test_scaled)[:, 1]
    rf_proba = rf.predict_proba(X_test_scaled)[:, 1]
    ensemble_proba = (svm_proba + rf_proba) / 2
    pred = (ensemble_proba > 0.5).astype(int)
    
    # Find errors
    errors = y_test != pred
    error_indices = np.where(errors)[0]
    
    for error_idx in error_indices:
        actual = int(y_test[error_idx])
        predicted = int(pred[error_idx])
        confidence = float(ensemble_proba[error_idx])
        
        all_errors.append({
            'fold': fold_idx + 1,
            'actual': actual,
            'predicted': predicted,
            'confidence': confidence,
            'error_type': 'False Positive' if actual == 0 and predicted == 1 else 'False Negative'
        })
    
    # Store all predictions
    for i in range(len(y_test)):
        all_predictions.append({
            'fold': fold_idx + 1,
            'actual': int(y_test[i]),
            'predicted': int(pred[i]),
            'confidence': float(ensemble_proba[i]),
            'correct': int(y_test[i]) == int(pred[i])
        })
        all_confidences.append(ensemble_proba[i])
    
    # Confusion matrix
    cm = confusion_matrix(y_test, pred)
    fold_cm_list.append(cm.tolist())
    
    print(f"✓ ({len(error_indices)} errors out of {len(y_test)})")

print(f"\n" + "="*70)
print("ERROR SUMMARY")
print("="*70)

total_errors = len(all_errors)
false_positives = sum(1 for e in all_errors if e['error_type'] == 'False Positive')
false_negatives = sum(1 for e in all_errors if e['error_type'] == 'False Negative')
total_predictions = len(all_predictions)
accuracy = sum(1 for p in all_predictions if p['correct']) / total_predictions

print(f"\nTotal predictions: {total_predictions}")
print(f"Correct predictions: {sum(1 for p in all_predictions if p['correct'])}")
print(f"Total errors: {total_errors} ({total_errors/total_predictions*100:.2f}%)")
print(f"  - False Positives (Normal → Autism): {false_positives} ({false_positives/total_errors*100:.1f}% of errors)")
print(f"  - False Negatives (Autism → Normal): {false_negatives} ({false_negatives/total_errors*100:.1f}% of errors)")

# Error confidence analysis
error_confidences = np.array([e['confidence'] for e in all_errors])
if len(error_confidences) > 0:
    print(f"\nError confidence pattern:")
    print(f"  Mean confidence on errors: {error_confidences.mean():.4f}")
    print(f"  Std of error confidences: {error_confidences.std():.4f}")
    print(f"  Min error confidence: {error_confidences.min():.4f}")
    print(f"  Max error confidence: {error_confidences.max():.4f}")

# Confidence on correct vs incorrect
correct_confidences = [p['confidence'] for p in all_predictions if p['correct']]
incorrect_confidences = [p['confidence'] for p in all_predictions if not p['correct']]

print(f"\nConfidence distribution:")
print(f"  Mean confidence (correct): {np.mean(correct_confidences):.4f}")
print(f"  Mean confidence (incorrect): {np.mean(incorrect_confidences):.4f}")
print(f"  Separation: {np.mean(correct_confidences) - np.mean(incorrect_confidences):.4f}")

# Aggregate confusion matrix
cm_total = np.array(fold_cm_list).sum(axis=0)
print(f"\nAggregate Confusion Matrix (5-fold):")
print(f"  True Negatives:  {cm_total[0, 0]}")
print(f"  False Positives: {cm_total[0, 1]}")
print(f"  False Negatives: {cm_total[1, 0]}")
print(f"  True Positives:  {cm_total[1, 1]}")

sensitivity = cm_total[1, 1] / (cm_total[1, 1] + cm_total[1, 0]) if (cm_total[1, 1] + cm_total[1, 0]) > 0 else 0
specificity = cm_total[0, 0] / (cm_total[0, 0] + cm_total[0, 1]) if (cm_total[0, 0] + cm_total[0, 1]) > 0 else 0

print(f"\nClinical metrics:")
print(f"  Sensitivity (autism detection): {sensitivity*100:.2f}%")
print(f"  Specificity (normal detection): {specificity*100:.2f}%")

# Save detailed results
results = {
    'summary': {
        'total_predictions': total_predictions,
        'total_errors': total_errors,
        'accuracy': float(accuracy),
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'sensitivity': float(sensitivity),
        'specificity': float(specificity)
    },
    'all_errors': all_errors[:50],  # Store first 50 for examples
    'confidence_stats': {
        'mean_error_confidence': float(error_confidences.mean()) if len(error_confidences) > 0 else 0,
        'std_error_confidence': float(error_confidences.std()) if len(error_confidences) > 0 else 0,
        'mean_correct_confidence': float(np.mean(correct_confidences)),
        'mean_incorrect_confidence': float(np.mean(incorrect_confidences)),
        'confidence_separation': float(np.mean(correct_confidences) - np.mean(incorrect_confidences))
    },
    'confusion_matrices': {
        'fold_wise': fold_cm_list,
        'aggregate': cm_total.tolist()
    }
}

with open('error_analysis.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved: error_analysis.json\n")
