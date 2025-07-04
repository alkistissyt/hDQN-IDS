"""
Visualization Diagrams for Hierarchical DQN Intrusion Detection
Connects with training results and evaluation metrics from train.py
Generates publication-ready plots
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
import pickle
import os
from typing import Dict, List, Tuple, Optional

# Configure matplotlib for publication-quality plots
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'Computer Modern'],
    'mathtext.fontset': 'cm',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.2,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.8
})

class ThesisVisualizer:
    """Generate publication-ready visualizations for thesis"""
    
    def __init__(self, results_path: str = None):
        """
        Initialize visualizer with training results
        
        Args:
            results_path: Path to training results pickle file
        """
        self.results_path = results_path
        self.results = None
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']  # Professional color palette
        
        if results_path and os.path.exists(results_path):
            self.load_results()
    
    def load_results(self):
        """Load training results from pickle file"""
        try:
            with open(self.results_path, 'rb') as f:
                self.results = pickle.load(f)
            print(f"Loaded results from {self.results_path}")
        except Exception as e:
            print(f"Could not load results: {e}")
            self.results = None
    
    def plot_training_progress(self, save_path: str = 'training_progress.png'):
        """Plot training progress for both agents"""
        if not self.results or 'training_history' not in self.results:
            print("No training history available. Generating example plot...")
            self._plot_example_training_progress(save_path)
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Extract training data
        for history in self.results['training_history']:
            agent_name = history['agent']
            metrics = history['metrics_history']
            
            if not metrics:
                continue
            
            timesteps = [m['timestep'] for m in metrics]
            accuracy = [m['avg_accuracy'] for m in metrics]
            reward = [m['avg_total_reward'] for m in metrics]
            
            color = '#1f77b4' if agent_name == 'worker' else '#ff7f0e'
            marker = 'o' if agent_name == 'worker' else '^'
            
            # Plot accuracy
            ax1.plot(timesteps, accuracy, f'{marker}-', color=color, linewidth=2,
                    markersize=4, alpha=0.8, label=f'{agent_name.title()} Agent')
            
            # Plot reward
            ax2.plot(timesteps, reward, f'{marker}-', color=color, linewidth=2,
                    markersize=4, alpha=0.8, label=f'{agent_name.title()} Agent')
        
        # Styling
        ax1.set_title('Accuracy During Training', fontweight='bold')
        ax1.set_ylabel('Accuracy', fontweight='bold')
        ax1.grid(True)
        ax1.legend()
        
        ax2.set_title('Average Reward During Training', fontweight='bold')
        ax2.set_xlabel('Timesteps', fontweight='bold')
        ax2.set_ylabel('Average Reward', fontweight='bold')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Training progress plot saved to {save_path}")
    
    def _plot_example_training_progress(self, save_path: str):
        """Generate example training progress plot"""
        timesteps = np.arange(0, 200000, 2000)
        np.random.seed(42)
        
        # Worker accuracy
        worker_acc = []
        for t in timesteps:
            if t < 20000:
                base = 0.5 + (t / 20000) * 0.4 + np.random.normal(0, 0.05)
            else:
                base = 0.92 + np.random.normal(0, 0.02)
            worker_acc.append(np.clip(base, 0, 1))
        
        # Manager accuracy  
        manager_acc = []
        for t in timesteps:
            if t < 15000:
                base = 0.3 + (t / 15000) * 0.2
            else:
                trend = 0.55 + 0.1 * (t - 15000) / 185000
                base = trend + 0.05 * np.sin(t / 15000) + np.random.normal(0, 0.02)
            manager_acc.append(np.clip(base, 0.2, 0.75))
        
        # Rewards
        worker_reward = 70 + 15 * np.sin(timesteps / 8000) + np.random.normal(0, 5, len(timesteps))
        manager_reward = 35 + 5 * np.sin(timesteps / 10000) + np.random.normal(0, 3, len(timesteps))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot accuracy
        ax1.plot(timesteps, worker_acc, 'o-', color='#1f77b4', linewidth=2,
                markersize=3, alpha=0.8, label='Worker Agent')
        ax1.plot(timesteps, manager_acc, '^-', color='#ff7f0e', linewidth=2,
                markersize=3, alpha=0.8, label='Manager Agent')
        
        ax1.set_title('Accuracy During Training', fontweight='bold')
        ax1.set_ylabel('Accuracy', fontweight='bold')
        ax1.grid(True)
        ax1.legend()
        ax1.set_xticks(np.arange(0, 250000, 50000))
        ax1.set_xticklabels([f'{int(x/1000)}k' for x in np.arange(0, 250000, 50000)])
        
        # Plot reward
        ax2.plot(timesteps, worker_reward, 'o-', color='#1f77b4', linewidth=2,
                markersize=3, alpha=0.8, label='Worker Agent')
        ax2.plot(timesteps, manager_reward, '^-', color='#ff7f0e', linewidth=2,
                markersize=3, alpha=0.8, label='Manager Agent')
        
        ax2.set_title('Average Reward During Training', fontweight='bold')
        ax2.set_xlabel('Timesteps', fontweight='bold')
        ax2.set_ylabel('Average Reward', fontweight='bold')
        ax2.grid(True)
        ax2.legend()
        ax2.set_xticks(np.arange(0, 250000, 50000))
        ax2.set_xticklabels([f'{int(x/1000)}k' for x in np.arange(0, 250000, 50000)])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_performance_comparison(self, performance_data: Dict, save_path: str = 'performance_comparison.png'):
        """
        Plot performance comparison across datasets
        
        Args:
            performance_data: Dict with dataset results
            save_path: Output file path
        """
        datasets = list(performance_data.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [performance_data[dataset][metric] for dataset in datasets]
            
            bars = axes[i].bar(datasets, values, color=self.colors, alpha=0.8, 
                             edgecolor='black', linewidth=1)
            
            # Add value labels
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{value:.1f}%', ha='center', va='bottom', 
                           fontweight='bold', fontsize=10)
            
            axes[i].set_title(f'{metric.replace("_", "-").title()} (%)', fontweight='bold')
            axes[i].set_ylabel('Percentage (%)', fontweight='bold')
            axes[i].set_ylim(0, 100)
            axes[i].grid(axis='y', alpha=0.3)
            
            # Rotate x-labels if needed
            if len(max(datasets, key=len)) > 8:
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Performance comparison saved to {save_path}")
    
    def plot_efficiency_analysis(self, efficiency_data: Dict, save_path: str = 'efficiency_analysis.png'):
        """
        Plot efficiency trade-off analysis
        
        Args:
            efficiency_data: Dict with adoption, resource_reduction, accuracy_loss
            save_path: Output file path
        """
        datasets = efficiency_data['datasets']
        adoption = efficiency_data['lightweight_adoption']
        resource_reduction = efficiency_data['resource_reduction']
        accuracy_loss = efficiency_data['accuracy_loss']
        
        # Calculate efficiency scores
        efficiency_scores = [a * r / max(l, 0.1) for a, r, l in 
                           zip(adoption, resource_reduction, accuracy_loss)]
        
        fig, ax = plt.subplots(figsize=(12, 9))
        
        # Bubble sizes proportional to efficiency score
        bubble_sizes = [score * 0.6 for score in efficiency_scores]
        
        # Create scatter plot
        scatter = ax.scatter(accuracy_loss, resource_reduction, s=bubble_sizes, 
                           c=self.colors[:len(datasets)], alpha=0.8,
                           edgecolors='black', linewidth=2.5, zorder=5)
        
        # Add dataset labels
        for i, dataset in enumerate(datasets):
            ax.annotate(f'{dataset}\n(Efficiency: {efficiency_scores[i]:.0f})',
                       (accuracy_loss[i], resource_reduction[i]),
                       xytext=(15, 15), textcoords='offset points',
                       fontweight='bold', fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                                edgecolor=self.colors[i], alpha=0.95, linewidth=1.5),
                       ha='center', va='center')
        
        # Styling
        ax.set_title('Efficiency Trade-off: Accuracy Loss vs Resource Reduction\n(Bubble size ∝ Efficiency Score)',
                    fontweight='bold', fontsize=16, pad=25)
        ax.set_xlabel('Accuracy Loss (%)', fontweight='bold', fontsize=13)
        ax.set_ylabel('Resource Reduction (%)', fontweight='bold', fontsize=13)
        ax.grid(True, alpha=0.3)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Efficiency analysis saved to {save_path}")
    
    def plot_confusion_matrices(self, confusion_data: Dict, save_path: str = 'confusion_matrices.png'):
        """
        Plot confusion matrices for different models/datasets
        
        Args:
            confusion_data: Dict with confusion matrix data
            save_path: Output file path
        """
        num_matrices = len(confusion_data)
        cols = min(2, num_matrices)
        rows = (num_matrices + 1) // 2
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
        if num_matrices == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for i, (name, cm_data) in enumerate(confusion_data.items()):
            if i >= len(axes):
                break
                
            cm = np.array(cm_data['matrix'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Benign', 'Attack'],
                       yticklabels=['Benign', 'Attack'],
                       ax=axes[i], cbar=True)
            
            axes[i].set_title(f'{name}', fontweight='bold')
            axes[i].set_xlabel('Predicted', fontweight='bold')
            axes[i].set_ylabel('Actual', fontweight='bold')
        
        # Hide empty subplots
        for i in range(len(confusion_data), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Confusion matrices saved to {save_path}")
    
    def generate_all_plots(self, dataset_results: Dict = None):
        """Generate all thesis plots"""
        print("Generating thesis visualizations...")
        
        # 1. Training Progress
        self.plot_training_progress()
        
        # 2. Performance Comparison (if data provided)
        if dataset_results:
            self.plot_performance_comparison(dataset_results)
        
        # 3. Example efficiency analysis
        example_efficiency = {
            'datasets': ['CICIDS2017', 'NF-ToN-IoT', 'Edge-IIoTset', 'BoT-IoT'],
            'lightweight_adoption': [41.8, 28.3, 48.6, 61.4],
            'resource_reduction': [21.5, 17.9, 23.2, 42.8],
            'accuracy_loss': [1.3, 2.9, 3.1, 0.7]
        }
        self.plot_efficiency_analysis(example_efficiency)
        
        # 4. Example confusion matrices
        example_confusion = {
            'Hierarchical Model': {
                'matrix': [[850, 45], [25, 180]]
            },
            'Worker-Only Model': {
                'matrix': [[820, 75], [35, 170]]
            }
        }
        self.plot_confusion_matrices(example_confusion)
        
        print("All visualizations generated successfully!")

def load_and_visualize(results_file: str = None):
    """
    Load results and generate visualizations
    
    Args:
        results_file: Path to training results pickle file
    """
    visualizer = ThesisVisualizer(results_file)
    
    # Example dataset results (replace with actual results)
    example_results = {
        'CICIDS2017': {'accuracy': 94.4, 'precision': 93.0, 'recall': 94.2, 'f1_score': 93.6},
        'NF-ToN-IoT': {'accuracy': 72.7, 'precision': 86.8, 'recall': 83.4, 'f1_score': 85.1},
        'Edge-IIoTset': {'accuracy': 91.1, 'precision': 93.5, 'recall': 92.9, 'f1_score': 93.2},
        'BoT-IoT': {'accuracy': 92.3, 'precision': 91.1, 'recall': 89.8, 'f1_score': 90.4}
    }
    
    visualizer.generate_all_plots(example_results)
    return visualizer

def create_summary_table(results: Dict):
    """Create a summary table of results"""
    print("\n" + "="*70)
    print("HIERARCHICAL DQN PERFORMANCE SUMMARY")
    print("="*70)
    print(f"{'Dataset':<15} {'Accuracy':<10} {'Precision':<11} {'Recall':<10} {'F1-Score':<10}")
    print("-"*70)
    
    for dataset, metrics in results.items():
        print(f"{dataset:<15} {metrics['accuracy']:<10.1f}% {metrics['precision']:<11.1f}% "
              f"{metrics['recall']:<10.1f}% {metrics['f1_score']:<10.1f}%")
    
    # Calculate averages
    avg_acc = np.mean([m['accuracy'] for m in results.values()])
    avg_prec = np.mean([m['precision'] for m in results.values()])
    avg_rec = np.mean([m['recall'] for m in results.values()])
    avg_f1 = np.mean([m['f1_score'] for m in results.values()])
    
    print("-"*70)
    print(f"{'AVERAGE':<15} {avg_acc:<10.1f}% {avg_prec:<11.1f}% {avg_rec:<10.1f}% {avg_f1:<10.1f}%")

def main():
    """Main function to generate thesis visualizations"""
    print("Thesis Visualization Generator")
    print("=" * 40)
    
    # Check for any .pkl files in current directory
    pkl_files = [f for f in os.listdir('.') if f.endswith('.pkl') and 'training_results' in f]
    
    results_file = None
    if pkl_files:
        results_file = pkl_files[0]  # Use first found file
        print(f"Found results file: {results_file}")
    else:
        print("No training results file found. Generating example visualizations...")
    
    # Generate visualizations
    visualizer = load_and_visualize(results_file)
    
    # Print summary
    example_results = {
        'CICIDS2017': {'accuracy': 94.4, 'precision': 93.0, 'recall': 94.2, 'f1_score': 93.6},
        'NF-ToN-IoT': {'accuracy': 72.7, 'precision': 86.8, 'recall': 83.4, 'f1_score': 85.1},
        'Edge-IIoTset': {'accuracy': 91.1, 'precision': 93.5, 'recall': 92.9, 'f1_score': 93.2},
        'BoT-IoT': {'accuracy': 92.3, 'precision': 91.1, 'recall': 89.8, 'f1_score': 90.4}
    }
    
    create_summary_table(example_results)
    
    print(f"\n All visualizations generated successfully!")
    print(f"Files created:")
    print(f"- training_progress.png")
    print(f"- performance_comparison.png") 
    print(f"- efficiency_analysis.png")
    print(f"- confusion_matrices.png")

if __name__ == "__main__":
    main()
