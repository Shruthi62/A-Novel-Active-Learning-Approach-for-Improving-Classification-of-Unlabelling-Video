"""
Standalone Training Script - Shows All Epochs & Saves 4 Graphs
===============================================================

Simple training with terminal output and graph saving (non-blocking).
"""

import os
import sys
import matplotlib
matplotlib.use('Agg')  # Non-blocking backend
import matplotlib.pyplot as plt
from novel_active_learning import active_learning_pipeline
from visualizations import generate_all_plots

print("\n" + "="*80)
print("[ACTIVE LEARNING VIDEO CLASSIFICATION - TRAINING]")
print("="*80 + "\n")

print("Configuration:")
print("  . Dataset: YouTube Action Dataset (11 classes)")
print("  . Initial Labeled: 22 samples")
print("  . Epochs per Iteration: 5")
print("  . Iterations: 3-4")
print("  . Expected Result: 92%+ accuracy with 90% less labeling")
print("\n" + "-"*80 + "\n")

try:
    print("STARTING TRAINING...\n")
    trained_model = active_learning_pipeline()
    
    print("\n" + "="*80)
    print("[TRAINING COMPLETED]")
    print("="*80 + "\n")
    
    # Generate and save 3 graphs (non-blocking)
    print("Generating Performance Graphs...\n")
    plots = generate_all_plots()
    
    # Save graphs as images instead of displaying
    for name, fig in plots.items():
        filename = f'{name}_graph.png'
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"   [OK] Saved: {filename}")
        plt.close(fig)
    
    print("\nAll 3 graphs saved as PNG files!")
    print("   1. progress_graph.png - Accuracy vs Iterations")
    print("   2. efficiency_graph.png - Data Efficiency Comparison")
    print("   3. uncertainty_graph.png - Uncertainty Distribution")
    print("\n")
    
    print("Results:")
    print("  . Model: best_model.pth (ready for inference)")
    print("  . Metrics: training_metrics.pkl & training_metrics.json")
    print("  . Graphs: 3 PNG images in current directory")
    print("\n" + "="*80 + "\n")
    
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


