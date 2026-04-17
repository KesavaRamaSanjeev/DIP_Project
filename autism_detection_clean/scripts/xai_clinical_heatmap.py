"""
XAI: Clinical Heatmap Visualization
====================================
Maps SHAP feature importance back to body parts for clinical interpretation.
Generates skeleton heatmaps showing which joints drive autism predictions.

IMPORTANT: This script does NOT modify the baseline model.
It only visualizes existing SHAP analysis results.
"""

import os
import sys
import numpy as np
import json
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
warnings.filterwarnings('ignore')

# Body part to dimensions mapping
BODY_PARTS = {
    'Left_Hand': {
        'dims': [0, 1, 2, 3, 64, 65, 66, 67, 128, 130],
        'pos': (0.35, 0.8),
        'label': 'Left Hand\n(Jerk, Asymmetry)',
        'color_key': 'left_limb'
    },
    'Right_Hand': {
        'dims': [8, 9, 10, 11, 72, 73, 74, 75, 129, 130],
        'pos': (0.65, 0.8),
        'label': 'Right Hand\n(Jerk, Asymmetry)',
        'color_key': 'right_limb'
    },
    'Left_Elbow': {
        'dims': [4, 5, 6, 7, 68, 69, 70, 71],
        'pos': (0.30, 0.65),
        'label': 'Left Elbow',
        'color_key': 'left_limb'
    },
    'Right_Elbow': {
        'dims': [12, 13, 14, 15, 76, 77, 78, 79],
        'pos': (0.70, 0.65),
        'label': 'Right Elbow',
        'color_key': 'right_limb'
    },
    'Head': {
        'dims': [24, 25, 26, 27, 88, 89, 90, 91, 131],
        'pos': (0.50, 0.25),
        'label': 'Head\n(Stability, Banging)',
        'color_key': 'head'
    },
    'Torso': {
        'dims': [28, 29, 30, 31, 92, 93, 94, 95],
        'pos': (0.50, 0.55),
        'label': 'Torso',
        'color_key': 'torso'
    },
    'Left_Knee': {
        'dims': [16, 17, 18, 19, 80, 81, 82, 83],
        'pos': (0.35, 0.40),
        'label': 'Left Knee',
        'color_key': 'left_limb'
    },
    'Right_Knee': {
        'dims': [12, 13, 14, 15, 84, 85, 86, 87],
        'pos': (0.65, 0.40),
        'label': 'Right Knee',
        'color_key': 'right_limb'
    },
    'Global_Motion': {
        'dims': [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
                 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
                 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127],
        'pos': (0.50, 0.10),
        'label': 'Global Motion Context',
        'color_key': 'global'
    },
    'Motion_Analysis': {
        'dims': [132, 133, 134, 135],
        'pos': (0.50, 0.05),
        'label': 'Motion Features\n(Repetition, Rigidity, Sync)',
        'color_key': 'motion'
    }
}


def aggregate_importance_by_body_part(shap_importance):
    """
    Aggregate SHAP values to body parts.
    Returns importance score for each body part.
    """
    body_part_importance = {}
    
    for body_part, info in BODY_PARTS.items():
        dims = info['dims']
        # Average SHAP importance for this body part
        importance_values = [shap_importance[d] for d in dims if d < len(shap_importance)]
        avg_importance = np.mean(importance_values) if importance_values else 0
        body_part_importance[body_part] = {
            'importance': avg_importance,
            'num_dims': len(dims),
            'dims': dims
        }
    
    return body_part_importance


def normalize_importance(body_part_importance):
    """Normalize importance to 0-1 range for color mapping"""
    importances = [bp['importance'] for bp in body_part_importance.values()]
    min_imp = np.min(importances)
    max_imp = np.max(importances)
    
    for bp in body_part_importance.values():
        if max_imp > min_imp:
            bp['normalized'] = (bp['importance'] - min_imp) / (max_imp - min_imp)
        else:
            bp['normalized'] = 0.5
    
    return body_part_importance


def get_color(importance, cmap='RdYlGn_r'):
    """Map importance score to color (Red = more important)"""
    cmap_func = plt.cm.get_cmap(cmap)
    return cmap_func(importance)


def create_clinical_heatmap(shap_importance, output_file='clinical_heatmap.png'):
    """
    Create skeleton visualization with body parts colored by SHAP importance.
    Red = High importance (strong autism indicator)
    Green = Low importance (weak signal)
    Returns normalized body part importance for downstream use.
    """
    
    print("\n[*] Creating clinical heatmap visualization...")
    
    # Aggregate importance by body part
    body_part_importance = aggregate_importance_by_body_part(shap_importance)
    body_part_importance = normalize_importance(body_part_importance)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    
    # Title
    fig.suptitle('SHAP Feature Importance - Clinical Heatmap\nAutism Detection from Video',
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Draw skeleton connections
    connections = [
        ((0.35, 0.65), (0.30, 0.65)),  # Right shoulder - Right elbow
        ((0.30, 0.65), (0.35, 0.80)),  # Right elbow - Right hand
        ((0.65, 0.65), (0.70, 0.65)),  # Left shoulder - Left elbow
        ((0.70, 0.65), (0.65, 0.80)),  # Left elbow - Left hand
        ((0.50, 0.55), (0.35, 0.65)),  # Torso - Right shoulder
        ((0.50, 0.55), (0.65, 0.65)),  # Torso - Left shoulder
        ((0.50, 0.55), (0.50, 0.25)),  # Torso - Head
        ((0.50, 0.55), (0.35, 0.40)),  # Torso - Right hip
        ((0.50, 0.55), (0.65, 0.40)),  # Torso - Left hip
    ]
    
    for start, end in connections:
        ax.plot([start[0], end[0]], [start[1], end[1]], 'gray', alpha=0.3, linewidth=2)
    
    # Draw body parts as circles with importance colors
    for body_part, info in BODY_PARTS.items():
        pos = info['pos']
        importance = body_part_importance[body_part]['normalized']
        color = get_color(importance)
        
        # Circle radius based on importance
        radius = 0.04 + (importance * 0.03)
        
        circle = patches.Circle(pos, radius, color=color, ec='black', linewidth=2, zorder=10)
        ax.add_patch(circle)
        
        # Label
        ax.text(pos[0], pos[1] + 0.12, info['label'], 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Importance value
        ax.text(pos[0], pos[1] - 0.10, f"{importance:.2f}",
                ha='center', va='top', fontsize=8, style='italic')
    
    # Legend
    ax.text(0.05, 0.02, 
            "Red = High importance (strong ASD indicator) | Green = Low importance",
            fontsize=10, style='italic', transform=ax.transAxes)
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, aspect=30, 
                        shrink=0.6, label='Normalized SHAP Importance')
    
    # Remove axes
    ax.axis('off')
    
    # Save
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  [OK] Heatmap saved to: {output_file}")
    
    return body_part_importance


def create_importance_bar_chart(shap_importance, top_n=15, 
                                output_file='shap_bar_chart.png'):
    """Create bar chart of top N features"""
    
    print("[*] Creating SHAP bar chart...")
    
    # Get top features
    top_indices = np.argsort(shap_importance)[::-1][:top_n]
    top_names = [f"Dim {i}" for i in top_indices]
    top_values = shap_importance[top_indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_values)))
    bars = ax.barh(range(len(top_values)), top_values, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_yticks(range(len(top_values)))
    ax.set_yticklabels(top_names)
    ax.set_xlabel('Mean |SHAP| Value (Importance)', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Most Important Features for Autism Detection\n(SHAP Feature Importance)',
                 fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, top_values)):
        ax.text(value, i, f' {value:.6f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  [OK] Bar chart saved to: {output_file}")


def create_body_part_summary_table(body_part_importance, 
                                   output_file='body_part_importance.txt'):
    """Create text summary of body part importances"""
    
    print("[*] Creating body part summary table...")
    
    # Sort by importance
    sorted_bp = sorted(body_part_importance.items(), 
                       key=lambda x: x[1]['importance'], reverse=True)
    
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CLINICAL INTERPRETATION: Body Part Importance\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"{'Body Part':<25} {'Importance':<15} {'Norm. (0-1)':<15} {'# Features':<12}\n")
        f.write("-"*80 + "\n")
        
        for body_part, info in sorted_bp:
            f.write(f"{body_part:<25} {info['importance']:<15.8f} "
                   f"{info['normalized']:<15.4f} {info['num_dims']:<12}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("INTERPRETATION GUIDE:\n")
        f.write("="*80 + "\n\n")
        
        f.write("Top Body Parts (High Importance):\n")
        for i, (body_part, info) in enumerate(sorted_bp[:5], 1):
            f.write(f"  {i}. {body_part}: {info['normalized']:.4f} - ")
            if body_part == 'Left_Hand' or body_part == 'Right_Hand':
                f.write("ARM FLAPPING or repetitive hand movements (ASD marker)\n")
            elif body_part == 'Head':
                f.write("HEAD BANGING or repetitive head movements (ASD marker)\n")
            elif body_part == 'Motion_Analysis':
                f.write("Stereotyped/repetitive motion patterns (ASD marker)\n")
            elif body_part == 'Global_Motion':
                f.write("Overall body motion context and consistency\n")
            else:
                f.write("Other motor patterns\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Clinical Recommendation:\n")
        f.write("Share this visualization with clinical experts to validate that\n")
        f.write("the model learned genuine autism behavioral markers, not spurious\n")
        f.write("correlations. Alignment with clinical observations strengthens\n")
        f.write("the model's reliability for potential clinical deployment.\n")
        f.write("="*80 + "\n")
    
    print(f"  [OK] Summary saved to: {output_file}")


def generate_clinical_heatmaps():
    """Main function to generate all clinical visualizations"""
    
    print(f"\n{'='*70}")
    print(f"CLINICAL HEATMAP GENERATION")
    print(f"{'='*70}\n")
    
    # Load SHAP results
    shap_file = 'shap_feature_importance.json'
    
    if not os.path.exists(shap_file):
        print(f"\n[ERROR] {shap_file} not found!")
        print(f"Please run: python scripts/xai_shap_analysis.py")
        return
    
    with open(shap_file, 'r') as f:
        shap_results = json.load(f)
    
    # Get first fold results
    fold_result = shap_results[0]
    print(f"[*] Using Fold {fold_result['fold']} results")
    print(f"    Model Test Accuracy: {fold_result['model_test_accuracy']:.4f}\n")
    
    # Convert to numpy array
    shap_importance = np.array(fold_result['all_feature_importance'])
    
    # Create visualizations
    print(f"\n{'─'*70}\n")
    body_part_imp = create_clinical_heatmap(shap_importance, 'clinical_heatmap.png')
    print()
    create_importance_bar_chart(shap_importance, top_n=15, 
                               output_file='shap_feature_importance_bars.png')
    print()
    create_body_part_summary_table(body_part_imp,
        'body_part_importance_summary.txt'
    )
    
    print(f"\n{'='*70}")
    print(f"[OK] All visualizations generated!")
    print(f"{'='*70}\n")
    
    print("Generated Files:")
    print("  ✓ clinical_heatmap.png - Skeleton with body part importances")
    print("  ✓ shap_feature_importance_bars.png - Top 15 features bar chart")
    print("  ✓ body_part_importance_summary.txt - Clinical interpretation table")
    print("\nNext: Review visualizations and share with clinical experts!")
    print()


if __name__ == '__main__':
    try:
        generate_clinical_heatmaps()
        print("[SUCCESS] Clinical heatmap generation completed!\n")
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
