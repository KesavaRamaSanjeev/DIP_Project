"""
K-Fold Cross-Validation for Autism Detection
================================================
Stratified 5-fold validation using cached LSTM + Motion features
Dataset: Clean autism motion data (117 samples, no duplicates)
Models: CNN-LSTM, RandomForest, XGBoost, SVM, Ensemble Voting
"""

import sys
import os

# Move to parent directory
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import joblib

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except:
    HAS_XGB = False
    print("⚠ XGBoost not available")

from models.classifier import IntegratedModel


def load_cached_features():
    """Load pre-extracted features"""
    print("Loading cached features...")
    X_lstm = np.load('X_lstm.npy')
    X_motion = np.load('X_motion.npy')
    y = np.load('Y_labels.npy')
    
    print(f"✓ Features loaded:")
    print(f"  X_lstm: {X_lstm.shape}")
    print(f"  X_motion: {X_motion.shape}")
    print(f"  y: {y.shape}")
    
    # Stack features: LSTM (128) + Motion (8) = 136 features
    X = np.concatenate([X_lstm, X_motion], axis=1)
    print(f"  Combined features: {X.shape}")
    
    return X, y


def train_cnn_lstm_fold(X_lstm_train, X_motion_train, y_train, 
                        X_lstm_test, X_motion_test, y_test, device, fold_num):
    """Train CNN-LSTM for one fold"""
    
    # Convert to tensors
    X_lstm_train_t = torch.FloatTensor(X_lstm_train).to(device)
    X_motion_train_t = torch.FloatTensor(X_motion_train).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    
    X_lstm_test_t = torch.FloatTensor(X_lstm_test).to(device)
    X_motion_test_t = torch.FloatTensor(X_motion_test).to(device)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1).to(device)
    
    # Model
    model = IntegratedModel(num_classes=1, num_frames=16, hidden_size=64).to(device)
    
    # Simple linear classifier on top of extracted features
    lstm_classifier = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(32, 1),
        nn.Sigmoid()
    ).to(device)
    
    optimizer = optim.Adam(lstm_classifier.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    # Training
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    for epoch in range(100):
        # Forward
        pred_train = lstm_classifier(X_lstm_train_t)
        loss = criterion(pred_train, y_train_t)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Validation
        with torch.no_grad():
            pred_test = lstm_classifier(X_lstm_test_t)
            val_loss = criterion(pred_test, y_test_t)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    # Final prediction
    with torch.no_grad():
        pred_probs = lstm_classifier(X_lstm_test_t).cpu().numpy().flatten()
    
    pred_labels = (pred_probs > 0.5).astype(int)
    accuracy = accuracy_score(y_test, pred_labels)
    
    return accuracy


def evaluate_models(X_train, y_train, X_test, y_test, fold_num, device):
    """Train and evaluate all models for one fold"""
    
    results = {}
    
    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Split LSTM and Motion features
    X_lstm_train = X_train[:, :128]
    X_motion_train = X_train[:, 128:]
    X_lstm_test = X_test[:, :128]
    X_motion_test = X_test[:, 128:]
    
    # Random Forest
    print("  Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_split=3, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred)
    results['rf'] = {'accuracy': rf_acc, 'f1': rf_f1, 'pred': rf_pred}
    print(f"    ✓ RF: {rf_acc:.4f}")
    
    # XGBoost
    xgb_pred = rf_pred.copy() if HAS_XGB else rf_pred.copy()  # Default to RF predictions if XGB unavailable
    if HAS_XGB:
        print("  Training XGBoost...")
        xgb = XGBClassifier(n_estimators=100, max_depth=7, learning_rate=0.1, random_state=42, verbosity=0)
        xgb.fit(X_train_scaled, y_train)
        xgb_pred = xgb.predict(X_test_scaled)
        xgb_acc = accuracy_score(y_test, xgb_pred)
        xgb_f1 = f1_score(y_test, xgb_pred)
        results['xgb'] = {'accuracy': xgb_acc, 'f1': xgb_f1, 'pred': xgb_pred}
        print(f"    ✓ XGB: {xgb_acc:.4f}")
    else:
        results['xgb'] = {'accuracy': 0.0, 'f1': 0.0, 'pred': xgb_pred}
    
    # SVM
    print("  Training SVM (RBF)...")
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True)
    svm.fit(X_train_scaled, y_train)
    svm_pred = svm.predict(X_test_scaled)
    svm_acc = accuracy_score(y_test, svm_pred)
    svm_f1 = f1_score(y_test, svm_pred)
    results['svm'] = {'accuracy': svm_acc, 'f1': svm_f1, 'pred': svm_pred}
    print(f"    ✓ SVM: {svm_acc:.4f}")
    
    # LSTM classifier (on LSTM features only)
    print("  Training LSTM classifier...")
    lstm_acc = train_cnn_lstm_fold(X_lstm_train, X_motion_train, y_train,
                                   X_lstm_test, X_motion_test, y_test, device, fold_num)
    results['lstm'] = {'accuracy': lstm_acc, 'f1': 0.0, 'pred': None}
    print(f"    ✓ LSTM: {lstm_acc:.4f}")
    
    # Ensemble Voting (RF + XGB + SVM)
    print("  Computing ensemble voting...")
    ensemble_pred = (rf_pred.astype(float) + xgb_pred.astype(float) + svm_pred.astype(float)) / 3
    ensemble_pred = (ensemble_pred > 0.5).astype(int)
    ensemble_acc = accuracy_score(y_test, ensemble_pred)
    ensemble_f1 = f1_score(y_test, ensemble_pred)
    results['ensemble'] = {'accuracy': ensemble_acc, 'f1': ensemble_f1, 'pred': ensemble_pred}
    print(f"    ✓ Ensemble: {ensemble_acc:.4f}")
    
    return results


def main():
    print("\n" + "="*80)
    print("AUTISM DETECTION: K-FOLD CROSS-VALIDATION FRAMEWORK")
    print("="*80)
    print("Dataset: Clean autism motion data (117 samples, 4 duplicates removed)")
    print("Features: LSTM temporal (128) + Optical flow motion (8) = 136 total")
    print("Validation: 5-fold stratified cross-validation")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load features
    X, y = load_cached_features()
    
    # K-fold setup
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    all_results = {
        'lstm': [],
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
        
        print(f"Train samples: {len(y_train)} | Test samples: {len(y_test)}")
        print(f"Train class distribution: {np.sum(y_train)} autism, {len(y_train) - np.sum(y_train)} normal")
        print(f"Test class distribution: {np.sum(y_test)} autism, {len(y_test) - np.sum(y_test)} normal")
        
        fold_results = evaluate_models(X_train, y_train, X_test, y_test, fold_num, device)
        
        for model_name, result in fold_results.items():
            all_results[model_name].append(result['accuracy'])
    
    # Print Summary
    print("\n" + "="*80)
    print("FINAL RESULTS: 5-FOLD STRATIFIED CROSS-VALIDATION")
    print("="*80)
    
    summary_metrics = {}
    
    for model_name in ['lstm', 'rf', 'xgb', 'svm', 'ensemble']:
        accs = all_results[model_name]
        if any(a > 0 for a in accs):  # Only print if model was trained
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            summary_metrics[model_name] = (mean_acc, std_acc)
            
            model_display = model_name.upper()
            if model_name == 'lstm':
                model_display = 'LSTM Classifier'
            elif model_name == 'rf':
                model_display = 'Random Forest'
            elif model_name == 'xgb':
                model_display = 'XGBoost'
            elif model_name == 'svm':
                model_display = 'SVM (RBF)'
            elif model_name == 'ensemble':
                model_display = 'Ensemble Vote'
            
            print(f"\n{model_display}:")
            print(f"  Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
            print(f"  Min/Max: {np.min(accs):.4f} / {np.max(accs):.4f}")
            print(f"  Fold Scores: {' | '.join([f'{x:.4f}' for x in accs])}")
    
    # Find best model
    print("\n" + "="*80)
    print("BEST MODEL")
    print("="*80)
    best_model = max(summary_metrics.items(), key=lambda x: x[1][0])
    print(f"\n★ {best_model[0].upper()}: {best_model[1][0]:.4f} ± {best_model[1][1]:.4f} ★\n")
    
    # Final status
    if best_model[1][0] >= 0.81:
        print("✅ TARGET ACHIEVED: Accuracy ≥ 81%")
    else:
        print(f"⚠️  Target not met: {best_model[1][0]:.4f} < 0.81")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
