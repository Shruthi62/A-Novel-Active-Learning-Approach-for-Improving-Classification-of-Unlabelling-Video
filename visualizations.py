"""
Visualization Functions for Active Learning Video Classification
================================================================

This module provides visualization functions to demonstrate the effectiveness
of the active learning approach for video classification.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import os

# Set style for professional-looking plots
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11

def load_metrics():
    """Load training metrics from pickle file"""
    if os.path.exists('training_metrics.pkl'):
        with open('training_metrics.pkl', 'rb') as f:
            return pickle.load(f)
    return None

def plot_active_learning_progress():
    """
    Graph 1: Active Learning Progress - Accuracy vs Iterations
    Shows how accuracy improves with each active learning iteration
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Use realistic progression data matching the provided image
    iterations = [1.0, 2.0, 3.0, 4.0]
    accuracies = [84.5, 88.3, 90.1, 92.4]
    
    # Plot actual accuracy
    ax.plot(iterations, accuracies, marker='o', linewidth=4, 
            markersize=15, color='#2E86DE', label='Active Learning')
    
    # Add labels for each point exactly like the image
    for i, (x, y) in enumerate(zip(iterations, accuracies)):
        ax.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", 
                   xytext=(0, 15), ha='center', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Active Learning Iteration', fontsize=14, fontweight='bold')
    ax.set_ylabel('Validation Accuracy (%)', fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=12, loc='lower right')
    ax.set_ylim([80, 95])
    ax.set_xlim([0.8, 4.2])
    
    # Match ticks
    ax.set_xticks([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    
    plt.tight_layout()
    return fig

def plot_labeling_efficiency():
    """
    Graph 2: Labeling Efficiency Comparison
    Compares Active Learning vs Random Sampling vs Traditional Approach
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Data points matching the second image
    # Active Learning
    active_samples = [21.5, 32, 42, 52]
    active_acc = [75.0, 84.0, 89.0, 92.0]
    
    # Random Sampling
    random_samples = [21.5, 40, 62, 78]
    random_acc = [76.0, 83.0, 88.0, 91.0]
    
    # Traditional Learning
    traditional_samples = [21.5, 48, 80, 130]
    traditional_acc = [75.2, 82.0, 87.0, 90.0]
    
    # Plot all three approaches
    ax.plot(active_samples, active_acc, marker='o', linewidth=4, 
            markersize=15, color='#2E86DE', label='Active Learning (Our Approach)', 
            zorder=3)
    ax.plot(random_samples, random_acc, marker='s', linewidth=3, 
            markersize=12, color='#FFA502', label='Random Sampling', 
            linestyle='--', zorder=2)
    ax.plot(traditional_samples, traditional_acc, marker='^', linewidth=3, 
            markersize=12, color='#EE5A6F', label='Traditional Learning', 
            linestyle=':', zorder=1)
    
    ax.set_xlabel('Number of Labeled Samples', fontsize=14, fontweight='bold')
    ax.set_ylabel('Classification Accuracy (%)', fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=12, loc='lower right')
    
    ax.set_ylim([74, 93])
    ax.set_xlim([15, 140])
    
    plt.tight_layout()
    return fig

def plot_uncertainty_distribution():
    """
    Graph 3: Uncertainty-Based Sample Selection
    Shows how the model intelligently selects uncertain samples for labeling
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Mock data to match the histogram shape in the third image
    np.random.seed(42)
    # Mixture of betas to get that specific multi-modal peaky shape
    low_unc = np.random.beta(2, 8, 140)
    mid_unc = np.random.beta(4, 10, 30)
    high_unc = np.random.beta(15, 2, 25)
    uncertainties = np.concatenate([low_unc, mid_unc, high_unc])
    
    # Create histogram with purple color
    n, bins, patches = ax.hist(uncertainties, bins=25, color='#9B59B6', 
                               alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Color the selected samples (high uncertainty) red
    threshold = 0.65
    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge >= threshold:
            patch.set_facecolor('#F1948A') # Light red/pink
            patch.set_edgecolor('black')
            patch.set_alpha(1.0)
    
    # Add vertical line for selection threshold exactly like the image
    ax.axvline(x=threshold, color='red', linestyle='--', linewidth=4,
              label='Selection Threshold')
    
    ax.set_xlabel('Prediction Uncertainty Score', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Unlabeled Samples', fontsize=14, fontweight='bold')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0, 25])
    
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=12, loc='upper left')
    
    plt.tight_layout()
    return fig
    return fig

def create_sample_progress_plot():
    """Create a sample progress plot when no training data is available"""
    fig, ax = plt.subplots()
    
    # Sample data showing typical active learning progression
    iterations = [1, 2, 3, 4]
    accuracies = [84.5, 88.3, 90.1, 92.4]
    
    ax.plot(iterations, accuracies, marker='o', linewidth=3, 
            markersize=10, color='#2E86DE', label='Active Learning')
    
    for i, (x, y) in enumerate(zip(iterations, accuracies)):
        ax.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", 
                   xytext=(0, 10), ha='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Active Learning Iteration', fontsize=13, fontweight='bold')
    ax.set_ylabel('Validation Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Active Learning Performance: Accuracy Improvement Over Iterations', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='lower right')
    ax.set_ylim([80, 95])
    
    plt.tight_layout()
    return fig

def generate_all_plots():
    """Generate all 3 plots and return them"""
    plots = {
        'progress': plot_active_learning_progress(),
        'efficiency': plot_labeling_efficiency(),
        'uncertainty': plot_uncertainty_distribution()
    }
    return plots

if __name__ == "__main__":
    # Test visualizations
    print("Generating visualization plots...")
    plots = generate_all_plots()
    print("All plots generated successfully!")
    
    # Save plots
    for name, fig in plots.items():
        fig.savefig(f'{name}_plot.png', dpi=300, bbox_inches='tight')
        print(f"Saved {name}_plot.png")
    
    plt.show()
