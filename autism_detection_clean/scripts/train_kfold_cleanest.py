"""
Workflow 2: K-Fold Training on Cleanest Dataset
================================================
Stratified 5-fold cross-validation with:
- SVM + Random Forest Ensemble (soft voting)
- GridSearch hyperparameter tuning for SVM
- Data augmentation (2x Gaussian noise)
- Full results reporting per fold
"""

import os
import sys
import numpy as np
import warnings
import json
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

warnings.filterwarnings('ignore')


def augment_data(X, Y, augmentation_factor=2):
    """Apply Gaussian noise augmentation to training data"""
    X_augmented = [X.copy()]
    Y_augmented = [Y.copy()]
    
    for _ in range(augmentation_factor - 1):
        # Gaussian noise: 5% of std per dimension
        noise = np.random.normal(0, X.std(axis=0) * 0.05, X.shape)
        X_aug = X + noise
        X_augmented.append(X_aug)
        Y_augmented.append(Y.copy())
    
    return np.vstack(X_augmented), np.hstack(Y_augmented)


def train_kfold_cleanest():
    """
    Stratified 5-fold validation on cleanest dataset using Workflow 2
    """
    
    print(f"\n{'='*70}")
    print(f"K-FOLD VALIDATION: Cleanest Dataset (Workflow 2)")
    print(f"{'='*70}\n")
    
    # Load features
    print("[*] Loading features...")
    feature_file = 'X_combined_cleanest.npy'
    label_file = 'Y_labels_cleanest.npy'
    
    if not os.path.exists(feature_file) or not os.path.exists(label_file):
        print(f"[ERROR] Feature files not found!")
        print(f"   Please run: python scripts/extract_features_cleanest.py")
        return
    
    X_combined = np.load(feature_file)
    Y_labels = np.load(label_file)
    
    print(f"  [OK] Loaded {feature_file}: {X_combined.shape}")
    print(f"  [OK] Loaded {label_file}: {Y_labels.shape}")
    
    # Dataset info
    n_autism = np.sum(Y_labels == 1)
    n_normal = np.sum(Y_labels == 0)
    print(f"\n  Dataset Statistics:")
    print(f"    Autism (label=1): {n_autism} samples ({n_autism/len(Y_labels)*100:.1f}%)")
    print(f"    Normal (label=0): {n_normal} samples ({n_normal/len(Y_labels)*100:.1f}%)\n")
    
    if len(X_combined) < 10:
        print(f"[ERROR] Dataset too small ({len(X_combined)} samples)")
        return
    
    # Stratified 5-fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_results = []
    all_accuracies = []
    all_precisions = []
    all_recalls = []
    all_f1s = []
    all_y_true = []
    all_y_pred = []
    
    print(f"{'='*70}")
    print(f"Running 5-Fold Cross-Validation with Workflow 2")
    print(f"{'='*70}\n")
    
    fold_num = 1
    start_time = time.time()
    
    for train_idx, test_idx in skf.split(X_combined, Y_labels):
        print(f"[FOLD {fold_num}/5]") 
        print(f"{'─'*70}")
        
        # Split
        X_train, X_test = X_combined[train_idx], X_combined[test_idx]
        y_train, y_test = Y_labels[train_idx], Y_labels[test_idx]
        
        print(f"  Train set: {len(X_train)} samples ({np.sum(y_train == 1)} autism, {np.sum(y_train == 0)} normal)")
        print(f"  Test set:  {len(X_test)} samples ({np.sum(y_test == 1)} autism, {np.sum(y_test == 0)} normal)")
        
        # Data augmentation (2x on training)
        X_train_aug, y_train_aug = augment_data(X_train, y_train, augmentation_factor=2)
        print(f"  After augmentation: {len(X_train_aug)} training samples")
        
        # ====== SVM with GridSearch ======
        print(f"\n  [*] SVM Hyperparameter Tuning...") 
        param_grid_svm = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['rbf', 'poly'],
            'gamma': ['scale', 'auto']
        }
        
        svm_grid = GridSearchCV(
            SVC(random_state=42, probability=True),
            param_grid_svm,
            cv=3,
            n_jobs=-1,
            verbose=0
        )
        svm_grid.fit(X_train_aug, y_train_aug)
        
        print(f"    Best params: C={svm_grid.best_params_['C']}, "
              f"kernel={svm_grid.best_params_['kernel']}, "
              f"gamma={svm_grid.best_params_['gamma']}")
        print(f"    Best CV score: {svm_grid.best_score_:.4f}")
        
        # ====== Random Forest ======
        print(f"  [*] Training Random Forest...") 
        rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, max_depth=20)
        rf.fit(X_train_aug, y_train_aug)
        rf_score = rf.score(X_train_aug, y_train_aug)
        print(f"    RF train accuracy: {rf_score:.4f}")
        
        # ====== Ensemble Voting ======
        print(f"  [*] Creating Ensemble (RF + SVM)...") 
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('svm', svm_grid.best_estimator_)
            ],
            voting='soft'
        )
        ensemble.fit(X_train_aug, y_train_aug)
        
        # ====== Test on unseen data ======
        print(f"\n  [RESULTS] Test Set:") 
        
        y_pred = ensemble.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        print(f"    Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall:    {recall:.4f}")
        print(f"    F1-Score:  {f1:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n    Confusion Matrix:")
        print(f"      TN={cm[0,0]}, FP={cm[0,1]}")
        print(f"      FN={cm[1,0]}, TP={cm[1,1]}")
        
        fold_results.append({
            'fold': fold_num,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'svm_best_params': svm_grid.best_params_,
            'svm_cv_score': svm_grid.best_score_,
            'n_train': len(X_train_aug),
            'n_test': len(X_test)
        })
        
        all_accuracies.append(accuracy)
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        
        print(f"\n")
        fold_num += 1
    
    # Summary
    elapsed = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"CROSS-VALIDATION SUMMARY (Cleanest Dataset - Workflow 2)")
    print(f"{'='*70}\n")
    
    mean_acc = np.mean(all_accuracies)
    std_acc = np.std(all_accuracies)
    mean_prec = np.mean(all_precisions)
    mean_rec = np.mean(all_recalls)
    mean_f1 = np.mean(all_f1s)
    
    print(f"[RESULTS] Overall Results:")
    print(f"  Accuracy:  {mean_acc:.4f} ± {std_acc:.4f} ({[f'{x:.2f}%' for x in np.array(all_accuracies)*100]})")
    print(f"  Precision: {mean_prec:.4f}")
    print(f"  Recall:    {mean_rec:.4f}")
    print(f"  F1-Score:  {mean_f1:.4f}\n")
    
    # Overall confusion matrix
    cm_overall = confusion_matrix(all_y_true, all_y_pred)
    print(f"Overall Confusion Matrix (all folds combined):")
    print(f"  TN={cm_overall[0,0]}, FP={cm_overall[0,1]}")
    print(f"  FN={cm_overall[1,0]}, TP={cm_overall[1,1]}\n")
    
    print(f"⏱  Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)\n")
    
    # Save results
    results = {
        'dataset': 'cleanest',
        'workflow': 'Workflow 2 (SVM + RF Ensemble)',
        'timestamp': datetime.now().isoformat(),
        'fold_results': fold_results,
        'summary': {
            'mean_accuracy': float(mean_acc),
            'std_accuracy': float(std_acc),
            'mean_precision': float(mean_prec),
            'mean_recall': float(mean_rec),
            'mean_f1': float(mean_f1),
            'total_samples': int(len(X_combined)),
            'autism_samples': int(n_autism),
            'normal_samples': int(n_normal),
            'feature_dimensions': int(X_combined.shape[1]),
            'elapsed_seconds': float(elapsed)
        },
        'per_fold_accuracies': [f'{x:.4f}' for x in all_accuracies]
    }
    
    # Save to JSON
    result_file = 'kfold_results_cleanest.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Results saved to: {result_file}")
    
    # Save to NPY (for compatibility)
    result_npy_file = 'kfold_results_cleanest.npy'
    np.save(result_npy_file, fold_results)
    print(f"[OK] Results also saved to: {result_npy_file}\n")
    
    print(f"{'='*70}")
    print(f"Training Complete!")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    train_kfold_cleanest()
