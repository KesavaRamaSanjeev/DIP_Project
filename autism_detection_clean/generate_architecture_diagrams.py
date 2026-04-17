"""
Clean Architecture Diagrams for Autism Detection Paper
Similar to professional references (DUNet, CNN architectures)
Uses matplotlib with custom styling
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

# Professional colors
COLOR_INPUT = '#3498db'          # Blue - Input
COLOR_PROCESS = '#2ecc71'        # Green - Processing
COLOR_EXTRACTION = '#e74c3c'     # Red - Feature extraction
COLOR_MODEL = '#f39c12'          # Orange - Model
COLOR_OUTPUT = '#9b59b6'         # Purple - Output
COLOR_BASELINE = '#3498db'       # Blue - Baseline
COLOR_NOVEL = '#e74c3c'          # Red - Novel

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_figure(filename, dpi=300):
    """Save figure with high resolution"""
    plt.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    file_size = __import__('os').path.getsize(filename) / 1024
    print(f"✅ Saved: {filename} ({file_size:.1f} KB)")
    plt.close()

def add_box(ax, x, y, width, height, label, color, fontsize=10, fontweight='normal'):
    """Add a colored box with label"""
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.05",
        linewidth=2,
        edgecolor='#2c3e50',
        facecolor=color,
        alpha=0.85
    )
    ax.add_patch(box)
    
    # Add text
    ax.text(x, y, label, 
            ha='center', va='center',
            fontsize=fontsize,
            fontweight=fontweight,
            color='white',
            wrap=True)

def add_arrow(ax, x1, y1, x2, y2, label='', color='#34495e', width=2):
    """Add arrow between boxes"""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->', 
        mutation_scale=30,
        linewidth=width,
        color=color
    )
    ax.add_patch(arrow)
    
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.3, mid_y, label, 
               fontsize=8, color=color, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

# ============================================================================
# FIGURE 1: SYSTEM ARCHITECTURE (Similar to CNN diagram reference)
# ============================================================================

def create_system_architecture():
    """Create main system architecture diagram"""
    print("📊 Creating System Architecture Diagram...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(-1, 15)
    ax.set_ylim(-1, 12)
    ax.axis('off')
    
    # Title
    ax.text(7, 11.5, 'Autism Detection System Architecture', 
           fontsize=18, fontweight='bold', ha='center', color='#2c3e50')
    ax.text(7, 11, 'Video Input → Pose Estimation → Feature Extraction → Classification', 
           fontsize=11, ha='center', color='#34495e', style='italic')
    
    # Stage 1: VIDEO INPUT
    add_box(ax, 1, 9, 1.8, 1, 'Input Video\n(30-120 sec)', COLOR_INPUT, fontsize=10, fontweight='bold')
    add_arrow(ax, 1.9, 9, 3.5, 9)
    
    # Stage 2: POSE ESTIMATION
    add_box(ax, 4, 9, 1.8, 1.2, 'CustomHRNet\nPose Estimation\n(17 joints)', COLOR_PROCESS, fontsize=9, fontweight='bold')
    add_arrow(ax, 4.9, 9, 6.5, 9)
    
    # Stage 3: BASELINE FEATURES
    ax.text(6.5, 10.2, 'Baseline Features (136D)', fontsize=10, fontweight='bold', color='#2c3e50', ha='left')
    
    add_box(ax, 7, 8.5, 1.6, 0.8, 'LSTM\nTemporal\n(128D)', COLOR_BASELINE, fontsize=8)
    add_box(ax, 7, 7.5, 1.6, 0.8, 'Optical\nFlow (8D)', COLOR_BASELINE, fontsize=8)
    
    add_arrow(ax, 7.8, 8.5, 8.5, 9)
    add_arrow(ax, 7.8, 7.5, 8.5, 9)
    
    # Stage 4: NOVEL FEATURES
    ax.text(8.5, 10.2, 'Novel Features (10D)', fontsize=10, fontweight='bold', color='#2c3e50', ha='left')
    
    add_box(ax, 8.5, 8.3, 1.4, 0.65, 'Symmetry\n(1D)', COLOR_NOVEL, fontsize=8)
    add_box(ax, 8.5, 7.5, 1.4, 0.65, 'Entropy\n(4D)', COLOR_NOVEL, fontsize=8)
    add_box(ax, 8.5, 6.7, 1.4, 0.65, 'Jerk\n(5D)', COLOR_NOVEL, fontsize=8)
    
    add_arrow(ax, 9.2, 8.3, 10, 9)
    add_arrow(ax, 9.2, 7.5, 10, 9)
    add_arrow(ax, 9.2, 6.7, 10, 9)
    
    # Stage 5: COMBINED FEATURES
    add_box(ax, 10, 9, 1.8, 1, 'Combined\nFeature Matrix\n(146D)', COLOR_EXTRACTION, fontsize=9, fontweight='bold')
    add_arrow(ax, 10.9, 9, 12, 9)
    
    # Stage 6: ENSEMBLE CLASSIFIER
    add_box(ax, 12.5, 9, 1.6, 1.2, 'SVM\n(RBF)', COLOR_MODEL, fontsize=9, fontweight='bold')
    add_box(ax, 13.5, 7.8, 1.6, 1.2, 'Random\nForest\n(200 trees)', COLOR_MODEL, fontsize=9, fontweight='bold')
    
    # Soft voting
    add_arrow(ax, 12.5, 8.6, 12, 7.2)
    add_arrow(ax, 13.5, 7.2, 12, 7.2)
    add_box(ax, 12, 6.2, 1.4, 0.8, 'Soft Voting\nEnsemble', COLOR_PROCESS, fontsize=9, fontweight='bold')
    add_arrow(ax, 12, 5.8, 12, 4.8)
    
    # Stage 7: OUTPUT
    add_box(ax, 12, 4, 1.8, 1, 'Prediction\nAutism/Control', COLOR_OUTPUT, fontsize=10, fontweight='bold')
    
    # Add performance metrics box
    metrics_text = (
        'Performance Metrics:\n'
        '━━━━━━━━━━━━━━━\n'
        'Accuracy: 90.99%\n'
        'Sensitivity: 84.64%\n'
        'Specificity: 93.36%\n'
        'F1-Score: 0.8873'
    )
    ax.text(2, 4.5, metrics_text, 
           fontsize=9, color='#2c3e50',
           bbox=dict(boxstyle='round', facecolor='#f0f0f0', edgecolor='#bdc3c7', linewidth=2, pad=0.8),
           verticalalignment='top', family='monospace')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_INPUT, edgecolor='#2c3e50', linewidth=2, label='Input/Data'),
        mpatches.Patch(facecolor=COLOR_PROCESS, edgecolor='#2c3e50', linewidth=2, label='Processing'),
        mpatches.Patch(facecolor=COLOR_BASELINE, edgecolor='#2c3e50', linewidth=2, label='Baseline Features'),
        mpatches.Patch(facecolor=COLOR_NOVEL, edgecolor='#2c3e50', linewidth=2, label='Novel Features'),
        mpatches.Patch(facecolor=COLOR_EXTRACTION, edgecolor='#2c3e50', linewidth=2, label='Feature Extraction'),
        mpatches.Patch(facecolor=COLOR_MODEL, edgecolor='#2c3e50', linewidth=2, label='Classification Model'),
        mpatches.Patch(facecolor=COLOR_OUTPUT, edgecolor='#2c3e50', linewidth=2, label='Output/Prediction')
    ]
    ax.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=9, frameon=True, fancybox=True, shadow=True)
    
    save_figure('Figure_Architecture_System.png')
    print("✅ System Architecture Complete!\n")

# ============================================================================
# FIGURE 2: FEATURE EXTRACTION PIPELINE
# ============================================================================

def create_feature_extraction_pipeline():
    """Create detailed feature extraction pipeline"""
    print("📊 Creating Feature Extraction Pipeline...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(-0.5, 14)
    ax.set_ylim(-0.5, 12)
    ax.axis('off')
    
    # Title
    ax.text(7, 11.5, 'Feature Extraction Pipeline', 
           fontsize=18, fontweight='bold', ha='center', color='#2c3e50')
    
    # Stage 1: Skeleton sequence
    ax.text(1.5, 10.5, 't=1', fontsize=9, ha='center', fontweight='bold')
    for i, t in enumerate([1, 2, 3, 4, 5, 6, 'N']):
        if i < 6:
            add_box(ax, 0.7 + i*1.2, 9.5, 0.9, 0.9, f'Frame\n{t}', COLOR_INPUT, fontsize=7)
        else:
            ax.text(7.5, 9.5, '...', fontsize=14, ha='center', fontweight='bold', color='#34495e')
            add_box(ax, 8.3, 9.5, 0.9, 0.9, f'Frame\nN', COLOR_INPUT, fontsize=7)
    
    # Arrows down
    for i in range(6):
        add_arrow(ax, 0.7 + i*1.2, 9, 0.7 + i*1.2, 8)
    add_arrow(ax, 8.3, 9, 8.3, 8)
    
    # Stage 2: Skeleton joints
    ax.text(4, 8.5, 'Skeleton Sequences (17-joint keypoints per frame)', 
           fontsize=10, ha='center', fontweight='bold', color='#2c3e50')
    
    add_box(ax, 4, 7.5, 8, 0.7, '3D Joint Coordinates: Head, Shoulders, Elbows, Wrists, Hips, Knees, Ankles', 
           COLOR_PROCESS, fontsize=9)
    
    # Stage 3: Feature extraction branches
    add_arrow(ax, 4, 7.1, 4, 6.2)
    
    # Branch 1: LSTM Temporal
    add_box(ax, 2, 5.5, 1.5, 0.8, 'LSTM\nTemporal\nDynamics', COLOR_BASELINE, fontsize=9, fontweight='bold')
    ax.text(2, 4.9, '(128D)', fontsize=8, ha='center', color='#2c3e50', fontweight='bold')
    
    # Branch 2: Optical Flow
    add_box(ax, 4, 5.5, 1.5, 0.8, 'Optical\nFlow\nMotion', COLOR_BASELINE, fontsize=9, fontweight='bold')
    ax.text(4, 4.9, '(8D)', fontsize=8, ha='center', color='#2c3e50', fontweight='bold')
    
    # Branch 3: Symmetry
    add_box(ax, 6, 5.5, 1.5, 0.8, 'Bilateral\nSymmetry\nAsymmetry', COLOR_NOVEL, fontsize=9, fontweight='bold')
    ax.text(6, 4.9, '(1D)', fontsize=8, ha='center', color='#2c3e50', fontweight='bold')
    
    # Branch 4: Entropy
    add_box(ax, 8, 5.5, 1.5, 0.8, 'Motion\nEntropy\nStatistics', COLOR_NOVEL, fontsize=9, fontweight='bold')
    ax.text(8, 4.9, '(4D)', fontsize=8, ha='center', color='#2c3e50', fontweight='bold')
    
    # Branch 5: Jerk
    add_box(ax, 10, 5.5, 1.5, 0.8, 'Jerk\nAnalysis\nSmoothing', COLOR_NOVEL, fontsize=9, fontweight='bold')
    ax.text(10, 4.9, '(5D)', fontsize=8, ha='center', color='#2c3e50', fontweight='bold')
    
    # Arrows down to concatenation
    for x in [2, 4, 6, 8, 10]:
        add_arrow(ax, x, 4.6, x, 3.8)
    
    # Stage 4: Concatenation
    add_box(ax, 6, 3.2, 8, 0.8, 'Feature Concatenation', COLOR_EXTRACTION, fontsize=11, fontweight='bold')
    
    # Final feature vector
    add_arrow(ax, 6, 2.8, 6, 2)
    add_box(ax, 6, 1.2, 6, 0.8, 'Combined Feature Vector (146D)', COLOR_MODEL, fontsize=11, fontweight='bold')
    
    # Add feature summary box
    summary_text = (
        'Feature Summary:\n'
        '━━━━━━━━━━━━━━━━━━━\n'
        'Baseline: 136D\n'
        '  • LSTM: 128D\n'
        '  • Flow: 8D\n'
        'Novel: 10D\n'
        '  • Symmetry: 1D\n'
        '  • Entropy: 4D\n'
        '  • Jerk: 5D'
    )
    ax.text(12.5, 5, summary_text, 
           fontsize=8, color='#2c3e50',
           bbox=dict(boxstyle='round', facecolor='#f0f0f0', edgecolor='#bdc3c7', linewidth=2, pad=0.6),
           verticalalignment='top', family='monospace')
    
    save_figure('Figure_Feature_Extraction_Pipeline.png')
    print("✅ Feature Extraction Pipeline Complete!\n")

# ============================================================================
# FIGURE 3: CLASSIFICATION MODEL ARCHITECTURE
# ============================================================================

def create_classification_architecture():
    """Create classification model architecture"""
    print("📊 Creating Classification Architecture...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(-0.5, 14)
    ax.set_ylim(-0.5, 12)
    ax.axis('off')
    
    # Title
    ax.text(7, 11.5, 'Ensemble Classification Architecture', 
           fontsize=18, fontweight='bold', ha='center', color='#2c3e50')
    
    # Input
    add_box(ax, 2, 10, 2, 1, 'Feature Vector\n(146D)', COLOR_INPUT, fontsize=11, fontweight='bold')
    add_arrow(ax, 3, 9.5, 3, 8.8)
    
    # Classifier 1: SVM
    ax.text(1, 9.2, 'Classifier 1:', fontsize=10, fontweight='bold', color='#2c3e50')
    add_box(ax, 1, 8.2, 1.8, 1.2, 'SVM\n(RBF Kernel)\nC=100', COLOR_MODEL, fontsize=10, fontweight='bold')
    
    ax.text(4, 8.5, 'Probability\nMapping', fontsize=8, ha='center', color='#34495e')
    ax.text(1, 7.5, 'Output: P(Autism)', fontsize=8, ha='center', color='#2c3e50', fontweight='bold')
    add_arrow(ax, 1.9, 8.2, 2.5, 7)
    
    # Classifier 2: Random Forest
    ax.text(5, 9.2, 'Classifier 2:', fontsize=10, fontweight='bold', color='#2c3e50')
    add_box(ax, 5, 8.2, 1.8, 1.2, 'Random\nForest\n(200 trees)', COLOR_MODEL, fontsize=10, fontweight='bold')
    
    ax.text(7, 8.5, 'Ensemble\nVoting', fontsize=8, ha='center', color='#34495e')
    ax.text(5, 7.5, 'Output: P(Autism)', fontsize=8, ha='center', color='#2c3e50', fontweight='bold')
    add_arrow(ax, 5.9, 8.2, 5.5, 7)
    
    # Voting mechanism
    add_box(ax, 3, 6.5, 2, 0.8, 'Soft Voting\n(Average Probabilities)', COLOR_PROCESS, fontsize=10, fontweight='bold')
    
    # Decision threshold
    add_arrow(ax, 3, 6.1, 3, 5.3)
    add_box(ax, 3, 4.7, 2, 0.8, 'Decision Threshold\n(t = 0.5)', COLOR_EXTRACTION, fontsize=9, fontweight='bold')
    
    # Final prediction
    add_arrow(ax, 2.3, 4.3, 1.5, 3.5)
    add_arrow(ax, 3.7, 4.3, 4.5, 3.5)
    
    add_box(ax, 1.5, 2.5, 1.5, 0.8, 'AUTISM', COLOR_NOVEL, fontsize=11, fontweight='bold')
    add_box(ax, 4.5, 2.5, 1.5, 0.8, 'CONTROL', COLOR_BASELINE, fontsize=11, fontweight='bold')
    
    # Performance metrics
    metrics_text = (
        'Performance:\n'
        '━━━━━━━━━━━━━\n'
        'SVM Alone: 88.92%\n'
        'RF Alone: 87.45%\n'
        'Ensemble: 90.99%\n'
        'Synergy Gain: +2.07%'
    )
    ax.text(9, 8, metrics_text, 
           fontsize=9, color='#2c3e50',
           bbox=dict(boxstyle='round', facecolor='#f0f0f0', edgecolor='#bdc3c7', linewidth=2, pad=0.8),
           verticalalignment='top', family='monospace', fontweight='bold')
    
    # Hyperparameters box
    params_text = (
        'Hyperparameters:\n'
        '━━━━━━━━━━━━━━━━━\n'
        'SVM:\n'
        '  Kernel: RBF\n'
        '  C: 100\n'
        '  Gamma: auto\n'
        'RF:\n'
        '  Trees: 200\n'
        '  Max depth: None\n'
        '  CV: 5-Fold'
    )
    ax.text(9, 4.5, params_text, 
           fontsize=8, color='#2c3e50',
           bbox=dict(boxstyle='round', facecolor='#f0f0f0', edgecolor='#bdc3c7', linewidth=2, pad=0.6),
           verticalalignment='top', family='monospace')
    
    save_figure('Figure_Classification_Architecture.png')
    print("✅ Classification Architecture Complete!\n")

# ============================================================================
# FIGURE 4: VALIDATION STRATEGY
# ============================================================================

def create_validation_strategy():
    """Create validation strategy diagram"""
    print("📊 Creating Validation Strategy...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(-0.5, 14)
    ax.set_ylim(-0.5, 12)
    ax.axis('off')
    
    # Title
    ax.text(7, 11.5, 'Validation Strategy: 5-Fold Cross-Validation with Augmentation', 
           fontsize=16, fontweight='bold', ha='center', color='#2c3e50')
    
    # Dataset
    add_box(ax, 2, 10, 2.2, 1, 'Full Dataset\n333 Videos\n(156 ASD, 177 Control)', COLOR_INPUT, fontsize=9, fontweight='bold')
    add_arrow(ax, 3.1, 9.5, 4, 9)
    
    # Data augmentation
    add_box(ax, 5, 9, 2, 1, '2× Gaussian\nAugmentation', COLOR_PROCESS, fontsize=10, fontweight='bold')
    ax.text(5, 8.4, '666 samples', fontsize=8, ha='center', color='#2c3e50', fontweight='bold')
    add_arrow(ax, 6, 8.5, 7, 8)
    
    # 5-Fold splits
    ax.text(7, 10.2, '5-Fold Stratified Split', fontsize=11, fontweight='bold', color='#2c3e50', ha='center')
    
    folds_y = 8.5
    for i in range(5):
        fold_color = COLOR_EXTRACTION if i % 2 == 0 else COLOR_MODEL
        add_box(ax, 1.5 + i*2.5, folds_y, 1.8, 0.7, f'Fold {i+1}\nTest', fold_color, fontsize=8)
        add_arrow(ax, 1.5 + i*2.5, folds_y - 0.35, 1.5 + i*2.5, 6.8)
    
    # Training & Testing
    ax.text(1.5, 6.3, 'Train', fontsize=10, fontweight='bold', color='#2c3e50', ha='center')
    ax.text(1.5, 5.8, '266 samples\n(80%)', fontsize=8, ha='center', color='#2c3e50')
    
    add_box(ax, 1.5, 5, 1.8, 0.8, 'SVM + RF\nEnsemble', COLOR_MODEL, fontsize=9, fontweight='bold')
    add_arrow(ax, 1.5, 4.6, 1.5, 3.8)
    
    ax.text(4, 6.3, 'Test', fontsize=10, fontweight='bold', color='#2c3e50', ha='center')
    ax.text(4, 5.8, '66 samples\n(20%)', fontsize=8, ha='center', color='#2c3e50')
    
    add_box(ax, 4, 5, 1.8, 0.8, 'Evaluate\nMetrics', COLOR_EXTRACTION, fontsize=9, fontweight='bold')
    add_arrow(ax, 4, 4.6, 4, 3.8)
    
    # Metrics collection
    add_box(ax, 2.75, 3.2, 2.5, 0.8, 'Collect 5 Fold Results', COLOR_PROCESS, fontsize=10, fontweight='bold')
    
    # Final results
    add_arrow(ax, 2.75, 2.8, 7, 2)
    
    results_text = (
        'Final Results (Mean ± Std):\n'
        '━━━━━━━━━━━━━━━━━━━━━━━━━\n'
        'Accuracy:     90.99% ± 0.59%\n'
        'Sensitivity:  84.64% ± 0.50%\n'
        'Specificity:  93.36% ± 0.31%\n'
        'Precision:    92.81% ± 0.38%\n'
        'F1-Score:     0.8873 ± 0.004\n'
        'ROC-AUC:      0.9360'
    )
    ax.text(7, 2.8, results_text, 
           fontsize=9, color='#2c3e50',
           bbox=dict(boxstyle='round', facecolor='#f0f0f0', edgecolor='#2ecc71', linewidth=3, pad=1),
           verticalalignment='top', family='monospace', fontweight='bold')
    
    save_figure('Figure_Validation_Strategy.png')
    print("✅ Validation Strategy Complete!\n")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 70)
    print("🎨 CREATING CLEAN ARCHITECTURE DIAGRAMS")
    print("=" * 70 + "\n")
    
    try:
        create_system_architecture()
        create_feature_extraction_pipeline()
        create_classification_architecture()
        create_validation_strategy()
        
        print("\n" + "=" * 70)
        print("✅ ALL ARCHITECTURE DIAGRAMS CREATED SUCCESSFULLY!")
        print("=" * 70)
        print("\n📊 Output Files:")
        print("  1. Figure_Architecture_System.png - Complete system pipeline")
        print("  2. Figure_Feature_Extraction_Pipeline.png - Feature extraction details")
        print("  3. Figure_Classification_Architecture.png - Ensemble classifier")
        print("  4. Figure_Validation_Strategy.png - Cross-validation approach")
        print("\n✨ All diagrams are professional quality and ready for your paper!")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
