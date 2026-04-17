"""
K-Fold Validation with Data Augmentation for dataset_new/
==========================================================
Stratified 5-fold cross-validation with:
- 2x data augmentation per fold
- SVM hyperparameter tuning (GridSearch)
- Ensemble voting (RF + SVM)
- Honest per-fold test accuracy
"""

import os
import sys
import numpy as np
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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


def train_kfold_new():
    """
    Stratified 5-fold validation on dataset_new
    """
    
    print(f"\n{'='*70}")
    print(f"K-FOLD VALIDATION: dataset_new/")
    print(f"{'='*70}\n")
    
    # Load features
    print("📂 Loading features...")
    X_combined = np.load('X_combined_new.npy')
    Y_labels = np.load('Y_labels_new.npy')
    
    print(f"  Shape: {X_combined.shape}")
    print(f"  Labels: {np.sum(Y_labels == 1)} autism, {np.sum(Y_labels == 0)} normal\n")
    
    if len(X_combined) < 10:
        print(f"❌ ERROR: Dataset too small ({len(X_combined)} samples)")
        return
    
    # Stratified 5-fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_results = []
    all_accuracies = []
    all_precisions = []
    all_recalls = []
    all_f1s = []
    
    print(f"{'='*70}")
    print(f"Running 5-Fold Cross-Validation")
    print(f"{'='*70}\n")
    
    fold_num = 1
    start_time = time.time()
    
    for train_idx, test_idx in skf.split(X_combined, Y_labels):
        print(f"🔄 FOLD {fold_num}/5")
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
        print(f"\n  🔍 SVM Hyperparameter Tuning...")
        param_grid_svm = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['rbf', 'poly'],
            'gamma': ['scale', 'auto']
        }
        
        svm_grid = GridSearchCV(
            SVC(random_state=42),
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
        print(f"  🌳 Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X_train_aug, y_train_aug)
        rf_score = rf.score(X_train_aug, y_train_aug)
        print(f"    RF train accuracy: {rf_score:.4f}")
        
        # ====== Ensemble Voting ======
        print(f"  🤝 Creating Ensemble (RF + SVM)...")
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('svm', svm_grid.best_estimator_)
            ],
            voting='soft'
        )
        ensemble.fit(X_train_aug, y_train_aug)
        
        # ====== Test on unseen data ======
        print(f"\n  📊 Results on Test Set:")
        
        y_pred = ensemble.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"    Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall:    {recall:.4f}")
        print(f"    F1-Score:  {f1:.4f}")
        
        fold_results.append({
            'fold': fold_num,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        all_accuracies.append(accuracy)
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)
        
        print()
        fold_num += 1
    
    elapsed = time.time() - start_time
    
    # Summary Statistics
    print(f"{'='*70}")
    print(f"✅ VALIDATION COMPLETE (Time: {elapsed:.1f}s)")
    print(f"{'='*70}\n")
    
    mean_acc = np.mean(all_accuracies)
    std_acc = np.std(all_accuracies)
    
    mean_prec = np.mean(all_precisions)
    mean_recall = np.mean(all_recalls)
    mean_f1 = np.mean(all_f1s)
    
    print(f"📈 OVERALL METRICS (5 Folds):")
    print(f"  Accuracy:  {mean_acc:.4f} ± {std_acc:.4f}  [{mean_acc*100:.2f}% ± {std_acc*100:.2f}%]")
    print(f"  Precision: {mean_prec:.4f}")
    print(f"  Recall:    {mean_recall:.4f}")
    print(f"  F1-Score:  {mean_f1:.4f}")
    
    print(f"\n📊 Per-Fold Breakdown:")
    for result in fold_results:
        acc_pct = result['accuracy'] * 100
        print(f"  Fold {result['fold']}: {acc_pct:6.2f}%")
    
    print(f"\n🎯 FINAL RESULT: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    
    # Save results
    np.save('kfold_results_new.npy', np.array(fold_results, dtype=object), allow_pickle=True)
    
    print(f"\n💾 Results saved to kfold_results_new.npy")
    
    return fold_results


if __name__ == "__main__":
    try:
        results = train_kfold_new()
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print(f"Run this first: python scripts/process_dataset_new.py")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
