"""
Deep Q-Network Training Script for Hierarchical IDS
Trains both worker-only and hierarchical agents using Stable-Baselines3
Updated to work with generic DataProcessor and enhanced environments
Supports: CICIDS2017, NF-ToN-IoT, Edge-IIoTset, BoT-IoT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import psutil
import os
import pickle
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Stable-Baselines3 imports
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Import our custom modules
from data_preprocessing import DataProcessor
from environment import (
    EnvironmentFactory, 
    WorkerOnlyEnvironment, 
    HierarchicalIDSEnvironment, 
    evaluate_environment,
    setup_complete_pipeline
)

class MemoryTracker:
    """Track memory usage during training"""

    def __init__(self):
        self.memory_log = []

    def log_memory(self, stage: str):
        """Log current memory usage"""
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.memory_log.append({
            'stage': stage,
            'memory_mb': memory_mb
        })
        print(f"Memory usage at {stage}: {memory_mb:.2f} MB")

    def get_memory_summary(self) -> pd.DataFrame:
        """Get memory usage summary"""
        return pd.DataFrame(self.memory_log)

class EnhancedMetricsCallback(BaseCallback):
    """Enhanced callback to track training metrics including computational costs"""

    def __init__(self, eval_env, eval_freq: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.metrics_history = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Evaluate current model
            metrics = evaluate_environment(self.eval_env, self.model, num_episodes=5)
            metrics['timestep'] = self.n_calls
            self.metrics_history.append(metrics)

            if self.verbose > 0:
                print(f"Step {self.n_calls}: Accuracy={metrics['avg_accuracy']:.3f}, "
                      f"F1={metrics['avg_f1_score']:.3f}, Reward={metrics['avg_total_reward']:.2f}, "
                      f"Cost={metrics.get('avg_computational_cost', 0):.3f}")

        return True

class DQNTrainer:
    """Main trainer class for DQN agents with enhanced features"""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize DQN trainer

        Args:
            config: Training configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.memory_tracker = MemoryTracker()
        self.training_history = []

        # Models
        self.worker_model = None
        self.manager_model = None

        # Data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Processor for saving/loading
        self.processor = None

    def _get_default_config(self) -> Dict:
        """Get default training configuration with adaptive PCA settings"""
        return {
            # Adaptive PCA parameters
            'variance_threshold': 0.95,
            'min_components': 10,
            'max_components': 50,
            
            # Data parameters
            'test_size': 0.2,
            'random_state': 42,

            # DQN parameters
            'learning_rate': 1e-4,
            'buffer_size': 100000,
            'learning_starts': 1000,
            'batch_size': 64,
            'tau': 1.0,
            'gamma': 0.99,
            'train_freq': 4,
            'gradient_steps': 1,
            'target_update_interval': 1000,

            # Epsilon-greedy parameters
            'exploration_fraction': 0.3,
            'exploration_initial_eps': 1.0,
            'exploration_final_eps': 0.05,

            # Training parameters
            'worker_timesteps': 500000,
            'manager_timesteps': 200000,
            'eval_freq': 10000,
            'save_freq': 50000,

            # Environment parameters
            'max_episodes': 1000,
            'episode_length': 100
        }

    def load_data(self, file_path: str, force_reprocess: bool = False):
        """
        Load and preprocess data using the enhanced pipeline

        Args:
            file_path: Path to the dataset CSV file
            force_reprocess: Force reprocessing even if preprocessed data exists
        """
        print("Loading and preprocessing data...")
        print(f"Dataset: {os.path.basename(file_path)}")
        self.memory_tracker.log_memory("start_data_loading")

        # Generate output filenames based on input
        dataset_name = os.path.splitext(os.path.basename(file_path))[0]
        preprocessed_path = f'processed_{dataset_name}.npz'

        try:
            # Use the complete pipeline setup
            train_env, test_env, processor = setup_complete_pipeline(
                dataset_path=file_path,
                preprocessed_path=preprocessed_path,
                force_reprocess=force_reprocess,
                variance_threshold=self.config['variance_threshold'],
                min_components=self.config['min_components'],
                max_components=self.config['max_components']
            )
            
            # Extract data from environments
            self.X_train = train_env.X_data
            self.y_train = train_env.y_data
            self.X_test = test_env.X_data
            self.y_test = test_env.y_data
            self.processor = processor

        except Exception as e:
            print(f"Pipeline setup failed: {e}")
            print("Falling back to direct processing...")
            
            # Fallback to direct processing
            self.processor = DataProcessor(
                variance_threshold=self.config['variance_threshold'],
                min_components=self.config['min_components'],
                max_components=self.config['max_components']
            )

            # Process dataset
            self.X_train, self.X_test, self.y_train, self.y_test = self.processor.process_dataset(
                file_path,
                test_size=self.config['test_size'],
                random_state=self.config['random_state']
            )

            # Save processed data
            np.savez_compressed(preprocessed_path,
                              X_train=self.X_train, X_test=self.X_test,
                              y_train=self.y_train, y_test=self.y_test)

        # Save processor for later use
        processor_path = f'trained_preprocessor_{dataset_name}.pkl'
        if self.processor:
            self.processor.save_preprocessor(processor_path)

        self.memory_tracker.log_memory("data_loaded")

        print(f"\nData loaded successfully!")
        print(f"Training samples: {len(self.X_train)}")
        print(f"Test samples: {len(self.X_test)}")
        print(f"Feature dimensions: {self.X_train.shape[1]} (adaptive PCA)")
        print(f"Class distribution (train): {np.bincount(self.y_train)}")
        print(f"Class distribution (test): {np.bincount(self.y_test)}")
        
        if self.processor and hasattr(self.processor, 'optimal_components'):
            print(f"Optimal PCA components: {self.processor.optimal_components}")
            print(f"Variance preserved: {self.config['variance_threshold']:.1%}")

    def train_worker_agent(self) -> DQN:
        """
        Train the worker agent using WorkerOnlyEnvironment with smart features

        Returns:
            Trained worker DQN model
        """
        print("\nTraining Worker Agent...")
        print("=" * 40)

        self.memory_tracker.log_memory("start_worker_training")

        # Create worker environment
        worker_env = EnvironmentFactory.create_worker_environment(
            self.X_train, self.y_train, max_episodes=self.config['max_episodes']
        )

        print(f"Worker environment created:")
        print(f"- Observation space: {worker_env.observation_space.shape}")
        print(f"- Action space: {worker_env.action_space}")
        print(f"- Smart history features: enabled")
        print(f"- Queue simulation: enabled")

        # Wrap environment for monitoring
        worker_env = Monitor(worker_env)
        worker_env = DummyVecEnv([lambda: worker_env])

        # Create evaluation environment
        eval_env = EnvironmentFactory.create_worker_environment(
            self.X_test, self.y_test, max_episodes=100
        )
        eval_env = Monitor(eval_env)

        # Create DQN model with enhanced network
        self.worker_model = DQN(
            'MlpPolicy',
            worker_env,
            learning_rate=self.config['learning_rate'],
            buffer_size=self.config['buffer_size'],
            learning_starts=self.config['learning_starts'],
            batch_size=self.config['batch_size'],
            tau=self.config['tau'],
            gamma=self.config['gamma'],
            train_freq=self.config['train_freq'],
            gradient_steps=self.config['gradient_steps'],
            target_update_interval=self.config['target_update_interval'],
            exploration_fraction=self.config['exploration_fraction'],
            exploration_initial_eps=self.config['exploration_initial_eps'],
            exploration_final_eps=self.config['exploration_final_eps'],
            policy_kwargs={'net_arch': [256, 256, 128]},  # Enhanced network
            verbose=1,
            tensorboard_log="./dqn_worker_tensorboard/"
        )

        # Create enhanced callback for metrics tracking
        metrics_callback = EnhancedMetricsCallback(eval_env, eval_freq=self.config['eval_freq'])

        # Train the model
        print(f"Training worker for {self.config['worker_timesteps']} timesteps...")
        print("New reward structure: Block Malicious=+5, Allow Benign=+4, Allow Malicious=-10, Block Benign=-2")
        
        self.worker_model.learn(
            total_timesteps=self.config['worker_timesteps'],
            callback=[metrics_callback],
            progress_bar=True
        )

        # Save worker model
        model_name = f"trained_worker_model_{os.path.splitext(os.path.basename(self.config.get('dataset_name', 'default')))[0]}"
        self.worker_model.save(model_name)

        self.memory_tracker.log_memory("worker_training_complete")

        # Store training history
        self.training_history.append({
            'agent': 'worker',
            'metrics_history': metrics_callback.metrics_history
        })

        print("Worker agent training completed!")
        return self.worker_model

    def train_manager_agent(self) -> DQN:
        """
        Train the manager agent using HierarchicalIDSEnvironment with adaptive costs

        Returns:
            Trained manager DQN model
        """
        if self.worker_model is None:
            raise ValueError("Worker model must be trained first!")

        print("\nTraining Manager Agent...")
        print("=" * 40)

        self.memory_tracker.log_memory("start_manager_training")

        # Create hierarchical environment with trained worker
        manager_env = EnvironmentFactory.create_hierarchical_environment(
            self.X_train, self.y_train, self.worker_model, max_episodes=self.config['max_episodes']
        )

        print(f"Hierarchical environment created:")
        print(f"- Observation space: {manager_env.observation_space.shape}")
        print(f"- Action space: {manager_env.action_space} (0=lightweight, 1=full)")
        print(f"- Adaptive computational cost: enabled")
        print(f"- Queue-based cost scaling: enabled")

        # Wrap environment for monitoring
        manager_env = Monitor(manager_env)
        manager_env = DummyVecEnv([lambda: manager_env])

        # Create evaluation environment
        eval_env = EnvironmentFactory.create_hierarchical_environment(
            self.X_test, self.y_test, self.worker_model, max_episodes=100
        )
        eval_env = Monitor(eval_env)

        # Create DQN model for manager with enhanced network
        self.manager_model = DQN(
            'MlpPolicy',
            manager_env,
            learning_rate=self.config['learning_rate'],
            buffer_size=self.config['buffer_size'] // 2,  # Smaller buffer for manager
            learning_starts=self.config['learning_starts'] // 2,
            batch_size=self.config['batch_size'],
            tau=self.config['tau'],
            gamma=self.config['gamma'],
            train_freq=self.config['train_freq'],
            gradient_steps=self.config['gradient_steps'],
            target_update_interval=self.config['target_update_interval'],
            exploration_fraction=self.config['exploration_fraction'],
            exploration_initial_eps=self.config['exploration_initial_eps'],
            exploration_final_eps=self.config['exploration_final_eps'],
            policy_kwargs={'net_arch': [256, 128, 64]},  # Smaller network for manager
            verbose=1,
            tensorboard_log="./dqn_manager_tensorboard/"
        )

        # Create enhanced callback for metrics tracking
        metrics_callback = EnhancedMetricsCallback(eval_env, eval_freq=self.config['eval_freq'])

        # Train the model
        print(f"Training manager for {self.config['manager_timesteps']} timesteps...")
        print("Adaptive cost structure: Lightweight=0.1+0.01*q, Full=1.0+0.02*q (q=queue length)")
        
        self.manager_model.learn(
            total_timesteps=self.config['manager_timesteps'],
            callback=[metrics_callback],
            progress_bar=True
        )

        # Save manager model
        model_name = f"trained_manager_model_{os.path.splitext(os.path.basename(self.config.get('dataset_name', 'default')))[0]}"
        self.manager_model.save(model_name)

        self.memory_tracker.log_memory("manager_training_complete")

        # Store training history
        self.training_history.append({
            'agent': 'manager',
            'metrics_history': metrics_callback.metrics_history
        })

        print("Manager agent training completed!")
        return self.manager_model

    def evaluate_models(self, num_episodes: int = 20) -> Dict:
        """
        Evaluate both worker and hierarchical models with enhanced metrics

        Args:
            num_episodes: Number of episodes for evaluation

        Returns:
            Dictionary containing evaluation results
        """
        print("\nEvaluating Models...")
        print("=" * 30)

        results = {}

        # Evaluate worker-only model
        if self.worker_model is not None:
            print("Evaluating Worker-Only Model...")
            worker_env = EnvironmentFactory.create_worker_environment(
                self.X_test, self.y_test, max_episodes=num_episodes
            )
            worker_results = evaluate_environment(worker_env, self.worker_model, num_episodes)
            results['worker_only'] = worker_results

            print(f"Worker-Only Results:")
            print(f"  Accuracy: {worker_results['avg_accuracy']:.4f} ± {worker_results['std_accuracy']:.4f}")
            print(f"  Precision: {worker_results['avg_precision']:.4f} ± {worker_results['std_precision']:.4f}")
            print(f"  Recall: {worker_results['avg_recall']:.4f} ± {worker_results['std_recall']:.4f}")
            print(f"  F1-Score: {worker_results['avg_f1_score']:.4f} ± {worker_results['std_f1_score']:.4f}")
            print(f"  Avg Reward: {worker_results['avg_total_reward']:.2f} ± {worker_results['std_total_reward']:.2f}")

        # Evaluate hierarchical model
        if self.manager_model is not None and self.worker_model is not None:
            print("\nEvaluating Hierarchical Model...")
            hierarchical_env = EnvironmentFactory.create_hierarchical_environment(
                self.X_test, self.y_test, self.worker_model, max_episodes=num_episodes
            )
            hierarchical_results = evaluate_environment(hierarchical_env, self.manager_model, num_episodes)
            results['hierarchical'] = hierarchical_results

            print(f"Hierarchical Results:")
            print(f"  Accuracy: {hierarchical_results['avg_accuracy']:.4f} ± {hierarchical_results['std_accuracy']:.4f}")
            print(f"  Precision: {hierarchical_results['avg_precision']:.4f} ± {hierarchical_results['std_precision']:.4f}")
            print(f"  Recall: {hierarchical_results['avg_recall']:.4f} ± {hierarchical_results['std_recall']:.4f}")
            print(f"  F1-Score: {hierarchical_results['avg_f1_score']:.4f} ± {hierarchical_results['std_f1_score']:.4f}")
            print(f"  Avg Reward: {hierarchical_results['avg_total_reward']:.2f} ± {hierarchical_results['std_total_reward']:.2f}")
            print(f"  Avg Computational Cost: {hierarchical_results.get('avg_computational_cost', 0):.4f}")
            print(f"  Avg Queue Length: {hierarchical_results.get('avg_average_queue_length', 0):.2f}")

        return results

    def generate_enhanced_confusion_matrix(self, model, env_type: str = 'worker', num_episodes: int = 10):
        """
        Generate and plot enhanced confusion matrix with additional metrics

        Args:
            model: Trained model to evaluate
            env_type: Type of environment ('worker' or 'hierarchical')
            num_episodes: Number of episodes to run
        """
        print(f"Generating enhanced confusion matrix for {env_type} model...")

        # Create appropriate environment
        if env_type == 'worker':
            env = EnvironmentFactory.create_worker_environment(
                self.X_test, self.y_test, max_episodes=num_episodes
            )
        else:
            env = EnvironmentFactory.create_hierarchical_environment(
                self.X_test, self.y_test, self.worker_model, max_episodes=num_episodes
            )

        # Collect predictions and additional metrics
        y_true = []
        y_pred = []
        rewards = []
        costs = []
        queue_lengths = []

        for episode in range(num_episodes):
            obs, info = env.reset()
            terminated = False
            step_count = 0

            while not terminated and step_count < 200:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)

                if 'true_label' in info:
                    y_true.append(info['true_label'])
                    rewards.append(reward)
                    
                    # Extract queue length
                    queue_lengths.append(info.get('current_queue_length', 0))
                    
                    # Convert action to binary prediction
                    if env_type == 'hierarchical':
                        # For hierarchical, get the worker action and computational cost
                        worker_action = info.get('worker_action', action)
                        y_pred.append(1 if worker_action == 1 else 0)
                        costs.append(info.get('computational_cost', 0))
                    else:
                        y_pred.append(1 if action == 1 else 0)
                        costs.append(0)  # No computational cost for worker-only

                step_count += 1

        if len(y_true) == 0:
            print("No data collected for confusion matrix")
            return

        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Create enhanced visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Allow', 'Block'],
                   yticklabels=['Benign', 'Malicious'],
                   ax=axes[0, 0])
        axes[0, 0].set_title(f'Confusion Matrix - {env_type.title()} Model')
        axes[0, 0].set_xlabel('Predicted Action')
        axes[0, 0].set_ylabel('True Label')

        # Reward Distribution
        axes[0, 1].hist(rewards, bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].set_title('Reward Distribution')
        axes[0, 1].set_xlabel('Reward')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(np.mean(rewards), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(rewards):.2f}')
        axes[0, 1].legend()

        # Queue Length Over Time
        axes[1, 0].plot(queue_lengths, alpha=0.7, color='orange')
        axes[1, 0].set_title('Queue Length Over Time')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Queue Length')
        axes[1, 0].axhline(np.mean(queue_lengths), color='red', linestyle='--',
                          label=f'Mean: {np.mean(queue_lengths):.2f}')
        axes[1, 0].legend()

        # Computational Cost (for hierarchical only)
        if env_type == 'hierarchical' and costs:
            axes[1, 1].plot(costs, alpha=0.7, color='purple')
            axes[1, 1].set_title('Computational Cost Over Time')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Cost')
            axes[1, 1].axhline(np.mean(costs), color='red', linestyle='--',
                              label=f'Mean: {np.mean(costs):.3f}')
            axes[1, 1].legend()
        else:
            axes[1, 1].text(0.5, 0.5, 'No Computational\nCost Data', 
                           ha='center', va='center', transform=axes[1, 1].transAxes,
                           fontsize=14)
            axes[1, 1].set_title('Computational Cost')

        plt.tight_layout()

        # Save plot
        plt.savefig(f'enhanced_confusion_matrix_{env_type}.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Print enhanced classification report
        print(f"\nEnhanced Classification Report - {env_type.title()} Model:")
        print(classification_report(y_true, y_pred, target_names=['Benign', 'Malicious']))
        print(f"Average Reward: {np.mean(rewards):.4f}")
        print(f"Average Queue Length: {np.mean(queue_lengths):.4f}")
        if env_type == 'hierarchical':
            print(f"Average Computational Cost: {np.mean(costs):.4f}")

    def plot_enhanced_training_progress(self):
        """Plot enhanced training progress with new metrics"""
        if not self.training_history:
            print("No training history available")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        for i, history in enumerate(self.training_history):
            agent_name = history['agent']
            metrics = history['metrics_history']

            if not metrics:
                continue

            # Extract data
            timesteps = [m['timestep'] for m in metrics]
            accuracy = [m['avg_accuracy'] for m in metrics]
            f1_score = [m['avg_f1_score'] for m in metrics]
            reward = [m['avg_total_reward'] for m in metrics]
            
            # Enhanced metrics
            costs = [m.get('avg_computational_cost', 0) for m in metrics]
            queue_lengths = [m.get('avg_average_queue_length', 0) for m in metrics]

            # Plot accuracy
            axes[0, 0].plot(timesteps, accuracy, label=f'{agent_name.title()} Agent', marker='o')
            axes[0, 0].set_title('Accuracy During Training')
            axes[0, 0].set_xlabel('Timesteps')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].legend()
            axes[0, 0].grid(True)

            # Plot F1 score
            axes[0, 1].plot(timesteps, f1_score, label=f'{agent_name.title()} Agent', marker='s')
            axes[0, 1].set_title('F1-Score During Training')
            axes[0, 1].set_xlabel('Timesteps')
            axes[0, 1].set_ylabel('F1-Score')
            axes[0, 1].legend()
            axes[0, 1].grid(True)

            # Plot reward
            axes[0, 2].plot(timesteps, reward, label=f'{agent_name.title()} Agent', marker='^')
            axes[0, 2].set_title('Average Reward During Training')
            axes[0, 2].set_xlabel('Timesteps')
            axes[0, 2].set_ylabel('Average Reward')
            axes[0, 2].legend()
            axes[0, 2].grid(True)

            # Plot computational cost (hierarchical only)
            if agent_name == 'manager' and any(costs):
                axes[1, 0].plot(timesteps, costs, label=f'{agent_name.title()} Agent', marker='d')
                axes[1, 0].set_title('Computational Cost During Training')
                axes[1, 0].set_xlabel('Timesteps')
                axes[1, 0].set_ylabel('Average Cost')
                axes[1, 0].legend()
                axes[1, 0].grid(True)

            # Plot queue length
            if any(queue_lengths):
                axes[1, 1].plot(timesteps, queue_lengths, label=f'{agent_name.title()} Agent', marker='*')
                axes[1, 1].set_title('Queue Length During Training')
                axes[1, 1].set_xlabel('Timesteps')
                axes[1, 1].set_ylabel('Average Queue Length')
                axes[1, 1].legend()
                axes[1, 1].grid(True)

        # Memory usage plot
        memory_df = self.memory_tracker.get_memory_summary()
        if not memory_df.empty:
            axes[1, 2].plot(range(len(memory_df)), memory_df['memory_mb'], marker='d', color='red')
            axes[1, 2].set_title('Memory Usage During Training')
            axes[1, 2].set_xlabel('Training Stage')
            axes[1, 2].set_ylabel('Memory Usage (MB)')
            axes[1, 2].set_xticks(range(len(memory_df)))
            axes[1, 2].set_xticklabels(memory_df['stage'], rotation=45)
            axes[1, 2].grid(True)

        plt.tight_layout()
        plt.savefig('enhanced_training_progress.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save_enhanced_results(self, filename: str = None):
        """Save enhanced training results and models"""
        if filename is None:
            dataset_name = os.path.splitext(os.path.basename(self.config.get('dataset_name', 'default')))[0]
            filename = f'training_results_{dataset_name}.pkl'
            
        results = {
            'config': self.config,
            'training_history': self.training_history,
            'memory_usage': self.memory_tracker.get_memory_summary().to_dict(),
            'data_shapes': {
                'X_train': self.X_train.shape if self.X_train is not None else None,
                'X_test': self.X_test.shape if self.X_test is not None else None,
                'y_train': self.y_train.shape if self.y_train is not None else None,
                'y_test': self.y_test.shape if self.y_test is not None else None,
            },
            'processor_info': {
                'optimal_components': getattr(self.processor, 'optimal_components', None),
                'variance_threshold': getattr(self.processor, 'variance_threshold', None),
                'feature_columns': getattr(self.processor, 'feature_columns', None)
            }
        }

        with open(filename, 'wb') as f:
            pickle.dump(results, f)

        print(f"Enhanced results saved to {filename}")

def main():
    """
    Main training function with enhanced features
    """
    print("Enhanced Hierarchical DQN Training for Intrusion Detection")
    print("Supports: CICIDS2017, NF-ToN-IoT, Edge-IIoTset, BoT-IoT")
    print("=" * 60)

    # Enhanced configuration with adaptive PCA
    config = {
        # Adaptive PCA settings
        'variance_threshold': 0.95,   # Preserve 95% of variance
        'min_components': 10,         # Minimum components
        'max_components': 50,         # Maximum components
        
        # Dataset parameters
        'test_size': 0.2,
        'random_state': 42,

        # Training parameters (reduced for faster training)
        'worker_timesteps': 200000,   # Reduced for demonstration
        'manager_timesteps': 100000,  # Reduced for demonstration

        # DQN hyperparameters
        'learning_rate
