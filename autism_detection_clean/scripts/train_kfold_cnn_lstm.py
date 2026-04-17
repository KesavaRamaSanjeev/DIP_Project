"""
K-Fold Cross-Validation for CNN-LSTM on dataset_new
- 5-fold stratified cross-validation
- Train CNN-LSTM from scratch on each fold
- Report per-fold accuracy + mean ± std dev
- Compare with RF+SVM baseline (70.70%)
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
import random
from models.classifier import IntegratedModel
from scripts.dataset import get_dataloaders, AutoismDataset
from torch.utils.data import DataLoader

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class FocalLoss(nn.Module):
    """Focal Loss for small datasets"""
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        bce = nn.BCEWithLogitsLoss(reduction='none')
        loss = bce(logits, targets)
        p = torch.sigmoid(logits)
        p_t = torch.where(targets == 1, p, 1 - p)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = self.alpha * focal_weight * loss
        return focal_loss.mean()

def get_all_videos_and_labels(dataset_path='dataset_new'):
    """Get all video paths and labels"""
    autism_folder = os.path.join(dataset_path, 'autism')
    normal_folder = os.path.join(dataset_path, 'normal')
    
    video_paths = []
    labels = []
    
    # Autism videos (label 1)
    if os.path.exists(autism_folder):
        for f in os.listdir(autism_folder):
            if f.endswith('.mp4'):
                video_paths.append(os.path.join(autism_folder, f))
                labels.append(1)
    
    # Normal videos (label 0)
    if os.path.exists(normal_folder):
        for f in os.listdir(normal_folder):
            if f.endswith('.mp4'):
                video_paths.append(os.path.join(normal_folder, f))
                labels.append(0)
    
    return np.array(video_paths), np.array(labels)

def train_fold(fold_idx, train_indices, test_indices, all_videos, all_labels, device, num_frames=24):
    """Train CNN-LSTM on one fold"""
    print(f"\n{'='*70}")
    print(f"FOLD {fold_idx + 1}")
    print(f"{'='*70}")
    
    # Get train and test data for this fold
    train_videos = all_videos[train_indices]
    train_labels = all_labels[train_indices]
    test_videos = all_videos[test_indices]
    test_labels = all_labels[test_indices]
    
    print(f"Train samples: {len(train_videos)} (Autism: {sum(train_labels)}, Normal: {len(train_labels) - sum(train_labels)})")
    print(f"Test samples: {len(test_videos)} (Autism: {sum(test_labels)}, Normal: {len(test_labels) - sum(test_labels)})")
    
    # Create datasets and dataloaders
    train_dataset = AutoismDataset(train_videos, train_labels, num_frames=num_frames, augment=True)
    test_dataset = AutoismDataset(test_videos, test_labels, num_frames=num_frames, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    
    # Model
    model = IntegratedModel(num_classes=1, num_frames=num_frames, hidden_size=64).to(device)
    
    # Loss and Optimizer
    criterion = FocalLoss(alpha=1.0, gamma=2.0)
    optimizer = optim.SGD(model.parameters(), lr=5e-4, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5)
    
    # Training
    best_test_acc = 0.0
    patience = 25
    patience_counter = 0
    epochs_trained = 0
    
    print(f"\nTraining CNN-LSTM (max 100 epochs, batch=2, lr=5e-4)...")
    
    for epoch in range(100):
        # Train
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels_list = []
        
        for batch_idx, (videos, motion_feats, labels) in enumerate(train_loader):
            videos = videos.to(device)
            motion_feats = motion_feats.to(device)
            labels = labels.to(device).unsqueeze(1).float()
            
            optimizer.zero_grad()
            outputs = model(videos, motion_feats)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_preds.extend(preds.detach().cpu().numpy().flatten())
            train_labels_list.extend(labels.detach().cpu().numpy().flatten())
        
        train_acc = accuracy_score([int(l) for l in train_labels_list], [int(p) for p in train_preds])
        train_loss /= len(train_loader)
        
        # Test
        model.eval()
        test_preds = []
        test_labels_np = []
        
        with torch.no_grad():
            for videos, motion_feats, labels in test_loader:
                videos = videos.to(device)
                motion_feats = motion_feats.to(device)
                outputs = model(videos, motion_feats)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                test_preds.extend(preds.detach().cpu().numpy().flatten())
                test_labels_np.extend(labels.detach().cpu().numpy().flatten())
        
        test_acc = accuracy_score([int(l) for l in test_labels_np], [int(p) for p in test_preds])
        
        # Early stopping
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            patience_counter = 0
            torch.save(model.state_dict(), f'checkpoints/fold_{fold_idx+1}_best.pth')
        else:
            patience_counter += 1
        
        scheduler.step()
        epochs_trained += 1
        
        if (epoch + 1) % 10 == 0 or patience_counter >= patience:
            print(f"Epoch {epoch+1:3d} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | "
                  f"Train Loss: {train_loss:.4f} | Patience: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    # Load best model and evaluate
    model.load_state_dict(torch.load(f'checkpoints/fold_{fold_idx+1}_best.pth'))
    model.eval()
    
    test_preds = []
    test_labels_np = []
    
    with torch.no_grad():
        for videos, motion_feats, labels in test_loader:
            videos = videos.to(device)
            motion_feats = motion_feats.to(device)
            outputs = model(videos, motion_feats)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            test_preds.extend(preds.detach().cpu().numpy().flatten())
            test_labels_np.extend(labels.detach().cpu().numpy().flatten())
    
    fold_acc = accuracy_score([int(l) for l in test_labels_np], [int(p) for p in test_preds])
    fold_f1 = f1_score([int(l) for l in test_labels_np], [int(p) for p in test_preds])
    cm = confusion_matrix([int(l) for l in test_labels_np], [int(p) for p in test_preds])
    
    print(f"\nFold {fold_idx + 1} Results:")
    print(f"  Test Accuracy: {fold_acc:.4f}")
    print(f"  Test F1-Score: {fold_f1:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    print(f"  Epochs Trained: {epochs_trained}")
    
    return fold_acc

# Main
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load all videos
all_videos, all_labels = get_all_videos_and_labels('dataset_new')
print(f"\nTotal videos: {len(all_videos)}")
print(f"Autism: {sum(all_labels)}, Normal: {len(all_labels) - sum(all_labels)}")

# K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []

print("\n" + "="*70)
print("5-FOLD STRATIFIED CROSS-VALIDATION (CNN-LSTM)")
print("="*70)

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(all_videos, all_labels)):
    fold_acc = train_fold(fold_idx, train_idx, test_idx, all_videos, all_labels, device)
    fold_accuracies.append(fold_acc)

# Summary
print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)

for fold_idx, acc in enumerate(fold_accuracies):
    print(f"Fold {fold_idx + 1}: {acc:.4f} ({acc*100:.2f}%)")

mean_acc = np.mean(fold_accuracies)
std_acc = np.std(fold_accuracies)

print(f"\nMean Accuracy: {mean_acc:.4f} ({mean_acc*100:.2f}%)")
print(f"Std Dev: {std_acc:.4f} ({std_acc*100:.2f}%)")
print(f"\nResult: {mean_acc*100:.2f}% +/- {std_acc*100:.2f}%")

print(f"\nComparison with Baseline:")
print(f"  RF+SVM (K-Fold): 70.70% +/- 6.61%")
print(f"  CNN-LSTM (K-Fold): {mean_acc*100:.2f}% +/- {std_acc*100:.2f}%")

if mean_acc > 0.7070:
    improvement = (mean_acc - 0.7070) * 100
    print(f"  Improvement: +{improvement:.2f}%")
else:
    degradation = (0.7070 - mean_acc) * 100
    print(f"  Degradation: -{degradation:.2f}%")
