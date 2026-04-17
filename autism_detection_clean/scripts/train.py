import sys
import os
import warnings
warnings.filterwarnings('ignore')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import numpy as np
import random
from models.classifier import IntegratedModel
from scripts.dataset import get_dataloaders

class LabelSmoothingBCELoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, x, target):
        target = target.float() * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(x, target)


class FocalLoss(nn.Module):
    """Focal Loss for small datasets (focuses on hard examples)"""
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


def mixup_batch(videos, motion_feats, labels, alpha=1.0):
    """Mixup augmentation for small datasets"""
    batch_size = videos.size(0)
    index = torch.randperm(batch_size)
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    
    mixed_videos = lam * videos + (1 - lam) * videos[index]
    mixed_motion = lam * motion_feats + (1 - lam) * motion_feats[index]
    mixed_labels = lam * labels + (1 - lam) * labels[index]
    
    return mixed_videos, mixed_motion, mixed_labels


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train():
    set_seed(42)
    
    # Hyperparameters
    # Hyperparameters: Surgical precision with longer temporal context
    NUM_FRAMES = 24
    BATCH_SIZE = 2
    LR = 5e-4
    EPOCHS = 100
    PATIENCE = 25
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    train_loader, val_loader, test_loader = get_dataloaders(
        'dataset_new', batch_size=BATCH_SIZE, num_frames=NUM_FRAMES
    )
    
    # Model (Spatio-Temporal Two-Stream + Kinematic Descriptors)
    model = IntegratedModel(num_classes=1, num_frames=NUM_FRAMES, hidden_size=64).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Initialize
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    # Standard Loss for better peak performance
    # Using Focal Loss for small dataset (focuses on hard examples)
    criterion = FocalLoss(alpha=1.0, gamma=2.0)
    
    # Optimizer: SGD with Momentum is often superior for Action Recognition
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    
    # Adaptive Scheduler: Warm Restarts to escape local minima
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5)
    
    # Training
    best_val_acc = 0.0
    best_val_f1 = 0.0
    patience_counter = 0
    os.makedirs('checkpoints', exist_ok=True)
    
    print(f"\nStarting Training (3D CNN with Motion Features)...")
    print(f"Epochs: {EPOCHS}, LR: {LR}, Batch: {BATCH_SIZE}, Frames: {NUM_FRAMES}")
    print("=" * 70)
    
    for epoch in range(EPOCHS):
        # ---- Train ----
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []
        
        for batch_idx, (videos, motion_feats, labels) in enumerate(train_loader):
            videos = videos.to(device)
            motion_feats = motion_feats.to(device)
            labels = labels.to(device).unsqueeze(1).float()
            
            # Apply Mixup augmentation (50% chance)
            if np.random.random() > 0.5:
                videos, motion_feats, labels = mixup_batch(videos, motion_feats, labels, alpha=0.2)
            
            optimizer.zero_grad()
            outputs = model(videos, motion_feats)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_preds.extend(preds.detach().cpu().numpy().flatten())
            train_labels.extend(labels.detach().cpu().numpy().flatten())
        train_acc = accuracy_score([int(l) for l in train_labels], [int(p) for p in train_preds])
        
        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for videos, motion_feats, labels in val_loader:
                videos = videos.to(device)
                motion_feats = motion_feats.to(device)
                labels = labels.to(device).unsqueeze(1)
                outputs = model(videos, motion_feats)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_preds.extend(preds.cpu().numpy().flatten())
                val_labels.extend(labels.cpu().numpy().flatten())
        
        val_acc = accuracy_score([int(l) for l in val_labels], [int(p) for p in val_preds])
        val_f1 = f1_score([int(l) for l in val_labels], [int(p) for p in val_preds], zero_division=0)
        
        avg_train_loss = train_loss / max(len(train_loader), 1)
        avg_val_loss = val_loss / max(len(val_loader), 1)
        # Step scheduler
        scheduler.step(epoch + batch_idx / len(train_loader))
        lr_now = optimizer.param_groups[0]['lr']
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{EPOCHS}] "
                  f"Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                  f"Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f} | "
                  f"Val F1: {val_f1:.3f} | LR: {lr_now:.6f}")

        # Save best
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_f1 > best_val_f1):
            best_val_acc = val_acc
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
            if (epoch + 1) % 5 == 0:
                print(f"  >>> Best model! Val Acc: {val_acc:.3f}, F1: {val_f1:.3f}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # ===== Final Test =====
    print("\n" + "=" * 70)
    print("FINAL EVALUATION (Test Set)")
    print("=" * 70)
    
    if os.path.exists('checkpoints/best_model.pth'):
        model.load_state_dict(torch.load('checkpoints/best_model.pth', weights_only=True, map_location=device))
    
    # ===== Final Test with TTA (Test-Time Augmentation) =====
    print("\n" + "=" * 70)
    print("FINAL EVALUATION with Test-Time Augmentation (TTA)")
    print("=" * 70)
    
    model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=device, weights_only=True))
    model.eval()
    
    tta_preds, tta_labels = [], []
    
    # We use the test_loader but perform multiple passes
    with torch.no_grad():
        for videos, motion_feats, labels in test_loader:
            batch_scores = []
            # 5 TTA samples per video in batch
            for _ in range(5):
                v_in = videos.to(device)
                m_in = motion_feats.to(device)
                outputs = model(v_in, m_in)
                scores = torch.sigmoid(outputs).cpu().numpy()
                batch_scores.append(scores)
            
            # Average the scores over the 5 TTA samples
            avg_scores = np.mean(batch_scores, axis=0)
            preds = (avg_scores > 0.5).astype(int).flatten()
            tta_preds.extend(preds)
            tta_labels.extend(labels.numpy().flatten())
    
    tta_acc = accuracy_score(tta_labels, tta_preds)
    tta_f1 = f1_score(tta_labels, tta_preds, zero_division=0)
    cm = confusion_matrix(tta_labels, tta_preds)
    
    print(f"TTA Test Accuracy: {tta_acc:.4f} ({tta_acc*100:.1f}%)")
    print(f"TTA Test F1-Score: {tta_f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        print(f"Precision:   {tp/(tp+fp) if tp+fp>0 else 0:.4f}")
        print(f"Recall:      {tp/(tp+fn) if tp+fn>0 else 0:.4f}")
        print(f"Specificity: {tn/(tn+fp) if tn+fp>0 else 0:.4f}")
    
    with open('eval_results.txt', 'w') as f:
        f.write(f"TTA Test Accuracy: {tta_acc:.4f}\n")
        f.write(f"TTA Test F1-Score: {tta_f1:.4f}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")
        f.write(f"Best Val Acc: {best_val_acc:.4f}\n")
        f.write(f"Best Val F1:  {best_val_f1:.4f}\n")
    
    print(f"\nFinal TTA Results saved to eval_results.txt")


if __name__ == "__main__":
    train()
