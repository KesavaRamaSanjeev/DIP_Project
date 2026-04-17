"""
Train a Neural Network classifier on pre-extracted features (149 dims)
Dataset: 427 samples (dataset_new)
Split: 70% train, 15% val, 15% test
Epochs: 100
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load pre-extracted features
print("\nLoading pre-extracted features...")
X_combined = np.load('X_combined_new.npy')  # (427, 149)
Y_labels = np.load('Y_labels_new.npy')      # (427,)

print(f"Features shape: {X_combined.shape}")
print(f"Labels shape: {Y_labels.shape}")
print(f"Class distribution: {np.bincount(Y_labels.astype(int))}")

# Split: 70% train, 15% val, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X_combined, Y_labels, test_size=0.3, random_state=42, stratify=Y_labels
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Val set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Convert to torch tensors
X_train = torch.FloatTensor(X_train).to(device)
X_val = torch.FloatTensor(X_val).to(device)
X_test = torch.FloatTensor(X_test).to(device)
y_train = torch.FloatTensor(y_train).to(device).unsqueeze(1)
y_val = torch.FloatTensor(y_val).to(device).unsqueeze(1)
y_test = torch.FloatTensor(y_test).to(device).unsqueeze(1)

# Create DataLoaders
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Neural Network Custom Classifier
class FeatureClassifier(nn.Module):
    def __init__(self, input_dim=149):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 1)  # Binary classification
        )
    
    def forward(self, x):
        return self.net(x)

model = FeatureClassifier(input_dim=149).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel Parameters: {total_params:,}")

# Loss and Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

# Training
EPOCHS = 100
PATIENCE = 30
best_val_acc = 0.0
patience_counter = 0

print(f"\nStarting Training (NN on Features)...")
print(f"Epochs: {EPOCHS}, Batch: 16, LR: 1e-3")
print("=" * 70)

for epoch in range(EPOCHS):
    # ---- Train ----
    model.train()
    train_loss = 0.0
    train_preds = []
    train_labels_list = []
    
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        train_preds.extend(preds.detach().cpu().numpy().flatten())
        train_labels_list.extend(y_batch.detach().cpu().numpy().flatten())
    
    train_acc = accuracy_score([int(l) for l in train_labels_list], [int(p) for p in train_preds])
    train_loss /= len(train_loader)
    
    # ---- Validate ----
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val).item()
        val_preds = (torch.sigmoid(val_outputs) > 0.5).float()
        val_acc = accuracy_score(
            [int(l) for l in y_val.cpu().numpy().flatten()],
            [int(p) for p in val_preds.cpu().numpy().flatten()]
        )
    
    # ---- Test ----
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test).item()
        test_preds = (torch.sigmoid(test_outputs) > 0.5).float()
        test_acc = accuracy_score(
            [int(l) for l in y_test.cpu().numpy().flatten()],
            [int(p) for p in test_preds.cpu().numpy().flatten()]
        )
    
    # Early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), 'checkpoints/best_nn_features.pth')
    else:
        patience_counter += 1
    
    scheduler.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

# Load best model and evaluate
print("\n" + "=" * 70)
print("Loading best model...")
model.load_state_dict(torch.load('checkpoints/best_nn_features.pth'))
model.eval()

with torch.no_grad():
    # Train
    train_outputs = model(X_train)
    train_preds = (torch.sigmoid(train_outputs) > 0.5).float().cpu().numpy().flatten()
    train_acc = accuracy_score([int(l) for l in y_train.cpu().numpy().flatten()], [int(p) for p in train_preds])
    train_f1 = f1_score([int(l) for l in y_train.cpu().numpy().flatten()], [int(p) for p in train_preds])
    
    # Val
    val_outputs = model(X_val)
    val_preds = (torch.sigmoid(val_outputs) > 0.5).float().cpu().numpy().flatten()
    val_acc = accuracy_score([int(l) for l in y_val.cpu().numpy().flatten()], [int(p) for p in val_preds])
    val_f1 = f1_score([int(l) for l in y_val.cpu().numpy().flatten()], [int(p) for p in val_preds])
    
    # Test
    test_outputs = model(X_test)
    test_preds = (torch.sigmoid(test_outputs) > 0.5).float().cpu().numpy().flatten()
    test_acc = accuracy_score([int(l) for l in y_test.cpu().numpy().flatten()], [int(p) for p in test_preds])
    test_f1 = f1_score([int(l) for l in y_test.cpu().numpy().flatten()], [int(p) for p in test_preds])
    
    cm = confusion_matrix([int(l) for l in y_test.cpu().numpy().flatten()], [int(p) for p in test_preds])

print(f"\nFinal Results (Best Model):")
print(f"Train Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}")
print(f"Val Accuracy:   {val_acc:.4f}, F1: {val_f1:.4f}")
print(f"Test Accuracy:  {test_acc:.4f}, F1: {test_f1:.4f}")

print(f"\nConfusion Matrix (Test):")
print(cm)

print(f"\nClassification Report (Test):")
print(classification_report([int(l) for l in y_test.cpu().numpy().flatten()], 
                          [int(p) for p in test_preds],
                          target_names=['Normal', 'Autism']))

print(f"\nComparison with K-Fold RF+SVM:")
print(f"K-Fold RF+SVM Accuracy:  70.70%")
print(f"NN on Features Accuracy: {test_acc*100:.2f}%")
