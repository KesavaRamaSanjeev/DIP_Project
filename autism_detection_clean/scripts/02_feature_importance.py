"""
STEP 2: FEATURE IMPORTANCE ANALYSIS
====================================
Extract which features matter most using Random Forest feature importance.
This shows if your 3 novel features are in the top performers.

Output: feature_importance.json + visualization
"""

import os
import sys
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

print("\n" + "="*70)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*70 + "\n")

# Load Phase 1 complete features
X_complete = np.load('X_novelty3.npy')
Y_labels = np.load('Y_novelty3.npy')

print(f"[*] Loaded features: {X_complete.shape}\n")

# Feature names for interpretation
feature_names = []

# Baseline features (136)
for i in range(128):
    feature_names.append(f"LSTM_{i}")
for i in range(8):
    feature_names.append(f"Motion_{i}")

# Novel features
feature_names.append("Bilateral_Symmetry")  # 136
for i in range(4):
    feature_names.append(f"Motion_Entropy_{i}")  # 137-140
for i in range(5):
    feature_names.append(f"Jerk_Analysis_{i}")  # 141-145

print(f"Feature count: {len(feature_names)}")
assert len(feature_names) == 146, "Feature name mismatch!"

# Train Random Forest on full dataset (with CV) to get importance
print("Training Random Forest for feature importance extraction...\n")

importance_scores = np.zeros((5, 146))  # 5 folds
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_complete, Y_labels)):
    print(f"  Fold {fold_idx + 1}/5", end=" ")
    
    X_train, X_test = X_complete[train_idx], X_complete[test_idx]
    y_train, y_test = Y_labels[train_idx], Y_labels[test_idx]
    
    # Augment
    noise = np.random.normal(0, X_train.std(axis=0) * 0.05, X_train.shape)
    X_train_aug = np.vstack([X_train, X_train + noise])
    y_train_aug = np.hstack([y_train, y_train])
    
    # Standardize
    scaler = StandardScaler()
    X_train_aug = scaler.fit_transform(X_train_aug)
    
    # Train RF
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train_aug, y_train_aug)
    
    importance_scores[fold_idx] = rf.feature_importances_
    print("✓")

# Average importance across folds
mean_importance = importance_scores.mean(axis=0)
std_importance = importance_scores.std(axis=0)

# Top 30 features
top_indices = np.argsort(mean_importance)[::-1][:30]

print("\n" + "="*70)
print("TOP 30 MOST IMPORTANT FEATURES")
print("="*70 + "\n")

importance_data = []
for rank, idx in enumerate(top_indices, 1):
    name = feature_names[idx]
    importance = mean_importance[idx]
    is_novel = "✓ NOVEL" if idx >= 136 else "   baseline"
    
    print(f"{rank:2d}. {name:30s} {importance*100:6.2f}% {is_novel}")
    
    importance_data.append({
        'rank': rank,
        'index': int(idx),
        'name': name,
        'importance': float(importance),
        'std_dev': float(std_importance[idx]),
        'is_novel': bool(int(idx >= 136))
    })

# Count novel features in top 20
top_20_indices = top_indices[:20]
novel_in_top_20 = sum(1 for idx in top_20_indices if idx >= 136)
novel_in_top_30 = sum(1 for idx in top_indices if idx >= 136)

print(f"\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)
print(f"\nNovel features (10) in Top 20: {novel_in_top_20} ✓")
print(f"Novel features (10) in Top 30: {novel_in_top_30} ✓")
print(f"\nTop 3 features importance: {mean_importance[top_indices[0]]*100:.2f}%, {mean_importance[top_indices[1]]*100:.2f}%, {mean_importance[top_indices[2]]*100:.2f}%")

# Breakdown by type
lstm_importance = mean_importance[:128].sum()
motion_importance = mean_importance[128:136].sum()
symmetry_importance = mean_importance[136]
entropy_importance = mean_importance[137:141].sum()
jerk_importance = mean_importance[141:146].sum()

print(f"\nFeature Group Importance:")
print(f"  LSTM features (128):        {lstm_importance*100:6.2f}%")
print(f"  Motion features (8):        {motion_importance*100:6.2f}%")
print(f"  Symmetry (1):               {symmetry_importance*100:6.2f}%")
print(f"  Entropy (4):                {entropy_importance*100:6.2f}%")
print(f"  Jerk (5):                   {jerk_importance*100:6.2f}%")
print(f"  Total Novel (10):           {(symmetry_importance + entropy_importance + jerk_importance)*100:6.2f}%")

# Save results
results = {
    'all_features': [
        {
            'index': int(i),
            'name': feature_names[i],
            'importance': float(mean_importance[i]),
            'std_dev': float(std_importance[i]),
            'is_novel': bool(int(i >= 136))
        }
        for i in range(146)
    ],
    'top_30': importance_data,
    'novel_in_top_20': int(novel_in_top_20),
    'novel_in_top_30': int(novel_in_top_30),
    'feature_group_importance': {
        'lstm': float(lstm_importance),
        'motion': float(motion_importance),
        'symmetry': float(symmetry_importance),
        'entropy': float(entropy_importance),
        'jerk': float(jerk_importance),
        'total_novel': float(symmetry_importance + entropy_importance + jerk_importance)
    }
}

with open('feature_importance.json', 'w') as f:
    json.dump(results, f, indent=2)

# Create visualization
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Top 30 features
ax = axes[0]
top_30_names = [feature_names[i] for i in top_indices]
top_30_importance = mean_importance[top_indices]
colors = ['red' if i >= 136 else 'blue' for i in top_indices]

ax.barh(range(len(top_30_names)), top_30_importance, color=colors, alpha=0.7)
ax.set_yticks(range(len(top_30_names)))
ax.set_yticklabels(top_30_names, fontsize=9)
ax.set_xlabel('Importance Score', fontsize=11)
ax.set_title('Top 30 Most Important Features (Red=Novel, Blue=Baseline)', fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Plot 2: Feature group importance
ax = axes[1]
groups = ['LSTM\n(128)', 'Motion\n(8)', 'Symmetry\n(1)', 'Entropy\n(4)', 'Jerk\n(5)']
importance_by_group = [lstm_importance, motion_importance, symmetry_importance, entropy_importance, jerk_importance]
colors_group = ['blue', 'blue', 'red', 'red', 'red']

ax.bar(groups, importance_by_group, color=colors_group, alpha=0.7)
ax.set_ylabel('Total Importance Score', fontsize=11)
ax.set_title('Feature Group Importance Breakdown', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

for i, (group, imp) in enumerate(zip(groups, importance_by_group)):
    ax.text(i, imp + 0.01, f'{imp*100:.1f}%', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('feature_importance_plot.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Results saved:")
print(f"   - feature_importance.json")
print(f"   - feature_importance_plot.png\n")
