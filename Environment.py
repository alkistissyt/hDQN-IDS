# -*- coding: utf-8 -*-
"""
Hierarchical Intrusion Detection System Environment
Implements Gym-compatible environments for training DQN agents
Updated with new reward formulas and adaptive computational costs
Integrated with data_preprocessing.py for real intrusion detection datasets
Supports multiple datasets (CIC-IDS2017, NSL-KDD, UNSW-NB15 CICIDS2017, NF-
ToN-IoT, Edge-IIoTset, and BoT-IoT, etc.)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional
import random
from collections import deque
import os

# Try to import the preprocessor - handle case where it's not available
try:
    from data_preprocessing import DataProcessor
    PREPROCESSOR_AVAILABLE = True
except ImportError:
    print("Warning: data_preprocessing.py not found.")
    PREPROCESSOR_AVAILABLE = False

class BaseIDSEnvironment(gym.Env):
    """Base class for IDS environments"""

    def __init__(self, X_data: np.ndarray, y_data: np.ndarray, max_episodes: int = 1000):
        """
        Initialize base IDS environment

        Args:
            X_data: Feature data
            y_data: Labels (0=benign, 1=malicious)
            max_episodes: Maximum number of episodes
        """
        super().__init__()

        self.X_data = X_data
        self.y_data = y_data
        self.max_episodes = max_episodes
        self.data_size = len(X_data)

        # Episode tracking
        self.current_episode = 0
        self.current_step = 0
        self.episode_length = min(100, self.data_size // 10)  # Adaptive episode length

        # Data indexing
        self.data_indices = np.arange(self.data_size)
        self.current_data_idx = 0

        # Queue length tracking for adaptive computational cost
        self.queue_length_history = deque(maxlen=20)  # Track last 20 queue lengths
        self.current_queue_length = 0.0

        # Metrics tracking
        self.episode_metrics = {
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0,
            'total_reward': 0,
            'computational_cost': 0,
            'lightweight_mode_usage': 0,
            'full_mode_usage': 0,
            'average_queue_length': 0.0
        }

        # History for observation
        self.observation_history = deque(maxlen=5)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment"""
        super().reset(seed=seed)

        # Process data sequentially - no shuffling
        # Continue from where we left off, or restart if we've reached the end
        if self.current_data_idx >= self.data_size:
            self.current_data_idx = 0
            
        self.current_step = 0

        # Reset queue length tracking
        self.queue_length_history.clear()
        self.current_queue_length = 0.0

        # Reset metrics
        self.episode_metrics = {
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0,
            'total_reward': 0,
            'computational_cost': 0,
            'lightweight_mode_usage': 0,
            'full_mode_usage': 0,
            'average_queue_length': 0.0
        }

        # Clear history
        self.observation_history.clear()

        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def _get_current_sample(self) -> Tuple[np.ndarray, int]:
        """Get current data sample sequentially"""
        if self.current_data_idx >= self.data_size:
            self.current_data_idx = 0

        idx = self.data_indices[self.current_data_idx]
        return self.X_data[idx], self.y_data[idx]

    def _get_observation(self) -> np.ndarray:
        """Get current observation - to be implemented by subclasses"""
        raise NotImplementedError

    def _get_info(self) -> Dict:
        """Get info dictionary"""
        current_sample, true_label = self._get_current_sample()
        return {
            'current_step': self.current_step,
            'episode_length': self.episode_length,
            'true_label': true_label,
            'current_queue_length': self.current_queue_length,
            'average_queue_length': self._get_average_queue_length(),
            'metrics': self.episode_metrics.copy()
        }

    def _create_smart_history_entry(self, current_sample: np.ndarray, true_label: int) -> Dict:
        """Create intelligent history entry with rich statistics"""
        mean_val = np.mean(current_sample)
        std_val = np.std(current_sample)
        abs_mean = np.mean(np.abs(current_sample))
        
        return {
            'mean': mean_val,
            'std': std_val,
            'max': np.max(current_sample),
            'min': np.min(current_sample),
            'anomaly_score': np.sum(current_sample > mean_val + 2 * std_val),
            'true_label': true_label,
            'complexity': std_val / (abs_mean + 1e-8)
        }

    def _get_smart_history_stats(self) -> np.ndarray:
        """Extract smart statistics from observation history"""
        if not self.observation_history:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # Extract statistics from history
        means = [obs['mean'] for obs in self.observation_history]
        stds = [obs['std'] for obs in self.observation_history]
        anomaly_scores = [obs['anomaly_score'] for obs in self.observation_history]
        complexities = [obs['complexity'] for obs in self.observation_history]
        true_labels = [obs['true_label'] for obs in self.observation_history]
        
        return np.array([
            np.mean(means),                           # Average feature mean
            np.std(means),                            # Variability in feature means
            np.mean(anomaly_scores),                  # Average anomaly score
            np.std(anomaly_scores),                   # Variability in anomaly scores
            np.mean(complexities),                    # Average complexity
            np.std(complexities),                     # Variability in complexity
            np.mean(true_labels),                     # Proportion of malicious samples in history
            len(self.observation_history) / 5.0      # History fullness ratio
        ], dtype=np.float32)

    def _update_queue_length(self):
        """Update queue length based on system load simulation"""
        # Simulate queue length based on recent activity
        base_queue = max(0, np.random.normal(2.0, 1.0))  # Base queue length
        
        # Adjust based on recent false positives/negatives (system stress)
        recent_errors = (self.episode_metrics['false_positives'] + 
                        self.episode_metrics['false_negatives'])
        stress_factor = min(recent_errors * 0.1, 2.0)  # Cap stress impact
        
        # Adjust based on detection mode usage
        mode_factor = 0.0
        if self.current_step > 0:
            full_mode_ratio = self.episode_metrics['full_mode_usage'] / self.current_step
            mode_factor = full_mode_ratio * 0.5  # Full mode increases queue
        
        self.current_queue_length = base_queue + stress_factor + mode_factor
        self.queue_length_history.append(self.current_queue_length)

    def _get_average_queue_length(self) -> float:
        """Get average queue length from recent history"""
        if len(self.queue_length_history) == 0:
            return 0.0
        return np.mean(self.queue_length_history)

    def _update_metrics(self, action: int, true_label: int, reward: float, computational_cost: float = 0.0):
        """Update episode metrics"""
        self.episode_metrics['total_reward'] += reward
        self.episode_metrics['computational_cost'] += computational_cost
        self.episode_metrics['average_queue_length'] = self._get_average_queue_length()

        # For binary classification with actions: 0=allow, 1=block, 2=log
        if action == 1:  # Block
            if true_label == 1:  # Correctly blocked malicious
                self.episode_metrics['true_positives'] += 1
            else:  # Incorrectly blocked benign
                self.episode_metrics['false_positives'] += 1
        elif action == 0:  # Allow
            if true_label == 0:  # Correctly allowed benign
                self.episode_metrics['true_negatives'] += 1
            else:  # Incorrectly allowed malicious
                self.episode_metrics['false_negatives'] += 1
        # Action 2 (log) doesn't affect classification metrics directly

class WorkerOnlyEnvironment(BaseIDSEnvironment):
    """
    Worker-only environment for initial training
    Worker agent chooses: 0=allow, 1=block, 2=log
    """

    def __init__(self, X_data: np.ndarray, y_data: np.ndarray, max_episodes: int = 1000):
        super().__init__(X_data, y_data, max_episodes)

        # Action space: 0=allow, 1=block, 2=log
        self.action_space = spaces.Discrete(3)

        # Observation space: current sample + basic statistics
        feature_dim = X_data.shape[1]
        obs_dim = feature_dim + 16  # features + extended smart statistics
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Updated reward parameters based on new formulas
        self.reward_block_malicious = 5.0       # Block Malicious (a=1, y=1)
        self.reward_allow_benign = 4.0          # Allow Benign (a=0, y=0)
        self.penalty_allow_malicious = -10.0    # Allow Malicious (a=0, y=1)
        self.penalty_block_benign = -2.0        # Block Benign (a=1, y=0)
        self.reward_log = 0.5                   # Log Action (a=2)

    def _get_observation(self) -> np.ndarray:
        """Get observation including current sample and smart statistics"""
        current_sample, true_label = self._get_current_sample()

        # Update queue length
        self._update_queue_length()

        # Create smart history entry
        smart_entry = self._create_smart_history_entry(current_sample, true_label)
        
        # Store in history (keep last 5 entries)
        self.observation_history.append(smart_entry)

        # Basic episode statistics
        basic_stats = np.array([
            self.current_step / self.episode_length,  # Progress
            self.episode_metrics['true_positives'],
            self.episode_metrics['false_positives'],
            self.episode_metrics['true_negatives'],
            self.episode_metrics['false_negatives'],
            self.episode_metrics['total_reward'] / max(1, self.current_step),  # Avg reward
            self.episode_metrics['computational_cost'] / max(1, self.current_step),  # Avg cost
            self.current_queue_length,  # Current queue length
            self._get_average_queue_length(),  # Average queue length
        ], dtype=np.float32)

        # Get smart history statistics
        smart_stats = self._get_smart_history_stats()

        # Combine current sample with all statistics
        observation = np.concatenate([current_sample, basic_stats, smart_stats])

        return observation

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step"""
        current_sample, true_label = self._get_current_sample()

        # Calculate reward using new formula
        reward = self._calculate_worker_reward(action, true_label)

        # Update metrics (no computational cost for worker-only environment)
        self._update_metrics(action, true_label, reward, 0.0)

        # Move to next step
        self.current_step += 1
        self.current_data_idx += 1

        # Check if episode is done
        terminated = self.current_step >= self.episode_length
        truncated = False

        # Get next observation
        if not terminated:
            observation = self._get_observation()
        else:
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)

        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _calculate_worker_reward(self, action: int, true_label: int) -> float:
        """
        Calculate worker reward based on new formula:
        R_worker(a, y) = {
            +5.0 if a=1 and y=1 (Block Malicious)
            +4.0 if a=0 and y=0 (Allow Benign)
            -10.0 if a=0 and y=1 (Allow Malicious)
            -2.0 if a=1 and y=0 (Block Benign)
            +0.5 if a=2 (Log Action)
        }
        """
        if action == 1:  # Block
            if true_label == 1:  # Block Malicious
                return self.reward_block_malicious
            else:  # Block Benign
                return self.penalty_block_benign
        elif action == 0:  # Allow
            if true_label == 0:  # Allow Benign
                return self.reward_allow_benign
            else:  # Allow Malicious
                return self.penalty_allow_malicious
        else:  # Log (action == 2)
            return self.reward_log

class HierarchicalIDSEnvironment(BaseIDSEnvironment):
    """
    Hierarchical environment with manager and worker agents
    Manager chooses detection mode: 0=lightweight, 1=full
    Worker chooses action: 0=allow, 1=block, 2=log
    """

    def __init__(self, X_data: np.ndarray, y_data: np.ndarray, worker_agent=None, max_episodes: int = 1000):
        super().__init__(X_data, y_data, max_episodes)

        self.worker_agent = worker_agent

        # Action space for manager: 0=lightweight, 1=full
        self.action_space = spaces.Discrete(2)

        # Observation space: current sample + extended statistics
        feature_dim = X_data.shape[1]
        obs_dim = feature_dim + 22  # features + extended smart statistics
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Mode tracking
        self.current_mode = 0

        # Updated reward parameters for worker
        self.reward_block_malicious = 5.0       # Block Malicious (a=1, y=1)
        self.reward_allow_benign = 4.0          # Allow Benign (a=0, y=0)
        self.penalty_allow_malicious = -10.0    # Allow Malicious (a=0, y=1)
        self.penalty_block_benign = -2.0        # Block Benign (a=1, y=0)
        self.reward_log = 0.5                   # Log Action (a=2)

    def _get_observation(self) -> np.ndarray:
        """Get observation for manager agent with smart statistics"""
        current_sample, true_label = self._get_current_sample()

        # Update queue length
        self._update_queue_length()

        # Create smart history entry
        smart_entry = self._create_smart_history_entry(current_sample, true_label)
        
        # Store in history (keep last 5 entries)
        self.observation_history.append(smart_entry)

        # Extended statistics for manager decision including queue information
        manager_stats = np.array([
            self.current_step / self.episode_length,  # Progress
            self.episode_metrics['true_positives'],
            self.episode_metrics['false_positives'],
            self.episode_metrics['true_negatives'],
            self.episode_metrics['false_negatives'],
            self.episode_metrics['total_reward'] / max(1, self.current_step),  # Avg reward
            self.episode_metrics['computational_cost'] / max(1, self.current_step),  # Avg cost
            self.episode_metrics['lightweight_mode_usage'] / max(1, self.current_step),
            self.episode_metrics['full_mode_usage'] / max(1, self.current_step),
            self.current_queue_length,  # Current queue length
            self._get_average_queue_length(),  # Average queue length
            # Current sample complexity indicators
            smart_entry['anomaly_score'],
            smart_entry['complexity'],
            smart_entry['std'],
        ], dtype=np.float32)

        # Get smart history statistics
        smart_stats = self._get_smart_history_stats()

        # Combine current sample with all statistics
        observation = np.concatenate([current_sample, manager_stats, smart_stats])

        return observation

    def step(self, manager_action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute hierarchical environment step"""
        if self.worker_agent is None:
            raise ValueError("Worker agent must be provided for hierarchical environment")

        current_sample, true_label = self._get_current_sample()
        self.current_mode = manager_action

        # Update mode usage
        if manager_action == 0:
            self.episode_metrics['lightweight_mode_usage'] += 1
        else:
            self.episode_metrics['full_mode_usage'] += 1

        # Get worker observation (simulated based on detection mode)
        worker_obs = self._get_worker_observation(current_sample, manager_action)

        # Get worker action
        worker_action, _ = self.worker_agent.predict(worker_obs, deterministic=False)

        # Calculate reward using new hierarchical formula
        reward, computational_cost = self._calculate_hierarchical_reward(manager_action, worker_action, true_label)

        # Update metrics
        self._update_metrics(worker_action, true_label, reward, computational_cost)

        # Move to next step
        self.current_step += 1
        self.current_data_idx += 1

        # Check if episode is done
        terminated = self.current_step >= self.episode_length
        truncated = False

        # Get next observation
        if not terminated:
            observation = self._get_observation()
        else:
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)

        info = self._get_info()
        info['manager_action'] = manager_action
        info['worker_action'] = worker_action
        info['detection_mode'] = 'lightweight' if manager_action == 0 else 'full'
        info['computational_cost'] = computational_cost

        return observation, reward, terminated, truncated, info

    def _get_worker_observation(self, sample: np.ndarray, mode: int) -> np.ndarray:
        """
        Get worker observation based on detection mode
        Lightweight mode: reduced feature set
        Full mode: complete feature set
        """
        if mode == 0:  # Lightweight mode - use subset of features
            # Use first 60% of features for lightweight mode
            feature_subset_size = int(0.6 * len(sample))
            reduced_sample = sample[:feature_subset_size]

            # Pad to match worker's expected input size
            padding_size = len(sample) - feature_subset_size
            padded_sample = np.concatenate([reduced_sample, np.zeros(padding_size)])

            # Add noise to simulate reduced accuracy
            noise = np.random.normal(0, 0.01, padded_sample.shape)
            worker_sample = padded_sample + noise
        else:  # Full mode - use complete feature set
            worker_sample = sample

        # Create worker observation (simplified version of WorkerOnlyEnvironment observation)
        basic_stats = np.array([
            self.current_step / self.episode_length,
            self.episode_metrics['total_reward'] / max(1, self.current_step),
            self.episode_metrics['computational_cost'] / max(1, self.current_step),
            self.current_queue_length,
            self._get_average_queue_length(),
            mode,  # Detection mode as feature
            # Add some smart statistics if available
            self._get_smart_history_stats()[:6].mean() if self.observation_history else 0,  # Avg of smart stats
            0, 0, 0, 0, 0  # Placeholder stats for compatibility
        ], dtype=np.float32)

        worker_obs = np.concatenate([worker_sample, basic_stats])
        return worker_obs

    def _calculate_hierarchical_reward(self, manager_action: int, worker_action: int, true_label: int) -> Tuple[float, float]:
        """
        Calculate hierarchical reward based on new formulas:
        R_total = R_worker - C_computational
        
        C_computational = {
            0.1 + 0.01 * q  if Manager action = 0 (Lightweight)
            1.0 + 0.02 * q  if Manager action = 1 (Full)
        }
        where q is the average system queue length
        """
        # Calculate base worker reward using same formula as WorkerOnlyEnvironment
        worker_reward = 0.0
        if worker_action == 1:  # Block
            if true_label == 1:  # Block Malicious
                worker_reward = self.reward_block_malicious
            else:  # Block Benign
                worker_reward = self.penalty_block_benign
        elif worker_action == 0:  # Allow
            if true_label == 0:  # Allow Benign
                worker_reward = self.reward_allow_benign
            else:  # Allow Malicious
                worker_reward = self.penalty_allow_malicious
        else:  # Log (worker_action == 2)
            worker_reward = self.reward_log

        # Calculate adaptive computational cost
        q = self._get_average_queue_length()
        if manager_action == 0:  # Lightweight mode
            computational_cost = 0.1 + 0.01 * q
        else:  # Full mode
            computational_cost = 1.0 + 0.02 * q

        # Total reward = worker reward - computational cost
        total_reward = worker_reward - computational_cost

        return total_reward, computational_cost

class EnvironmentFactory:
    """Factory class for creating IDS environments"""

    @staticmethod
    def create_worker_environment(X_train: np.ndarray, y_train: np.ndarray, max_episodes: int = 1000) -> WorkerOnlyEnvironment:
        """Create worker-only environment for initial training"""
        return WorkerOnlyEnvironment(X_train, y_train, max_episodes)

    @staticmethod
    def create_hierarchical_environment(X_train: np.ndarray, y_train: np.ndarray, worker_agent, max_episodes: int = 1000) -> HierarchicalIDSEnvironment:
        """Create hierarchical environment with trained worker agent"""
        return HierarchicalIDSEnvironment(X_train, y_train, worker_agent, max_episodes)
    
    @staticmethod
    def create_from_dataset(dataset_path: str, use_train_data: bool = True, max_episodes: int = 1000, **kwargs):
        """
        Create environment directly from dataset file
        
        Args:
            dataset_path: Path to original dataset CSV file
            use_train_data: Whether to use training split for environment
            max_episodes: Maximum episodes for environment
            **kwargs: Additional arguments for DataProcessor
            
        Returns:
            WorkerOnlyEnvironment instance
            
        Raises:
            ImportError: If data_preprocessing module is not available
            FileNotFoundError: If dataset file is not found
        """
        if not PREPROCESSOR_AVAILABLE:
            raise ImportError(
                "data_preprocessing module not available. "
                "Please ensure data_preprocessing.py is in your Python path."
            )
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Dataset file '{dataset_path}' not found. "
                f"Please provide the correct path to your dataset file."
            )
        
        # Initialize processor
        processor = DataProcessor(**kwargs)
        
        # Process dataset
        X_train, X_test, y_train, y_test = processor.process_dataset(dataset_path)
        
        # Choose data split
        if use_train_data:
            X_data, y_data = X_train, y_train
            print("Using training data for environment...")
        else:
            X_data, y_data = X_test, y_test
            print("Using test data for environment...")
        
        return WorkerOnlyEnvironment(X_data, y_data, max_episodes)
    
    @staticmethod
    def create_from_preprocessed(data_path: str = 'processed_data.npz', 
                               use_train_data: bool = True, 
                               max_episodes: int = 1000):
        """
        Create environment from preprocessed data file
        
        Args:
            data_path: Path to preprocessed data (.npz file)
            use_train_data: Whether to use training data
            max_episodes: Maximum episodes for environment
            
        Returns:
            WorkerOnlyEnvironment instance
        """
        X_train, X_test, y_train, y_test = load_preprocessed_data(data_path)
        
        if use_train_data:
            X_data, y_data = X_train, y_train
            print("Using training data for environment...")
        else:
            X_data, y_data = X_test, y_test
            print("Using test data for environment...")
        
        return WorkerOnlyEnvironment(X_data, y_data, max_episodes)

def evaluate_environment(env: BaseIDSEnvironment, agent, num_episodes: int = 10) -> Dict:
    """
    Evaluate agent performance in environment

    Args:
        env: Environment to evaluate in
        agent: Trained agent
        num_episodes: Number of episodes to evaluate

    Returns:
        Dictionary of evaluation metrics
    """
    total_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'total_reward': [],
        'computational_cost': [],
        'average_queue_length': [],
        'episode_length': []
    }

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_cost = 0
        episode_metrics = {
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0
        }

        terminated = False
        step_count = 0

        while not terminated and step_count < 1000:  # Safety limit
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            step_count += 1

            # Update episode metrics from environment
            if 'metrics' in info:
                episode_metrics = info['metrics'].copy()
                episode_cost = episode_metrics.get('computational_cost', 0)

        # Calculate metrics
        tp = episode_metrics['true_positives']
        fp = episode_metrics['false_positives']
        tn = episode_metrics['true_negatives']
        fn = episode_metrics['false_negatives']

        # Avoid division by zero
        accuracy = (tp + tn) / max(1, tp + fp + tn + fn)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1_score = 2 * (precision * recall) / max(1, precision + recall)

        total_metrics['accuracy'].append(accuracy)
        total_metrics['precision'].append(precision)
        total_metrics['recall'].append(recall)
        total_metrics['f1_score'].append(f1_score)
        total_metrics['total_reward'].append(episode_reward)
        total_metrics['computational_cost'].append(episode_cost)
        total_metrics['average_queue_length'].append(info.get('average_queue_length', 0))
        total_metrics['episode_length'].append(step_count)

    # Calculate averages
    avg_metrics = {}
    for metric, values in total_metrics.items():
        avg_metrics[f'avg_{metric}'] = np.mean(values)
        avg_metrics[f'std_{metric}'] = np.std(values)

    return avg_metrics

def load_preprocessed_data(data_path: str = 'processed_data.npz'):
    """
    Load preprocessed data from the data_preprocessing.py output
    
    Args:
        data_path: Path to the processed data file
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
        
    Raises:
        FileNotFoundError: If the preprocessed data file is not found
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Preprocessed data file '{data_path}' not found. "
            f"Please run data_preprocessing.py first to generate the processed data, "
            f"or provide the correct path to your preprocessed data file."
        )
    
    try:
        print(f"Loading preprocessed data from {data_path}...")
        data = np.load(data_path)
        X_train = data['X_train']
        X_test = data['X_test'] 
        y_train = data['y_train']
        y_test = data['y_test']
        
        print(f"Data loaded successfully:")
        print(f"- Training set: X={X_train.shape}, y={y_train.shape}")
        print(f"- Test set: X={X_test.shape}, y={y_test.shape}")
        print(f"- Training class distribution: {np.bincount(y_train)}")
        print(f"- Test class distribution: {np.bincount(y_test)}")
        
        return X_train, X_test, y_train, y_test
        
    except KeyError as e:
        raise KeyError(
            f"Required key {e} not found in {data_path}. "
            f"Expected keys: 'X_train', 'X_test', 'y_train', 'y_test'. "
            f"Please ensure the file was generated by data_preprocessing.py."
        )
    except Exception as e:
        raise RuntimeError(f"Error loading data from {data_path}: {str(e)}")

def create_environment_with_real_data(data_path: str = 'processed_data.npz', 
                                    use_train_data: bool = True,
                                    max_episodes: int = 1000):
    """
    Create environment using preprocessed data
    
    Args:
        data_path: Path to preprocessed data
        use_train_data: Whether to use training data (True) or test data (False)
        max_episodes: Maximum episodes for environment
        
    Returns:
        WorkerOnlyEnvironment instance
    """
    return EnvironmentFactory.create_from_preprocessed(data_path, use_train_data, max_episodes)

def setup_complete_pipeline(dataset_path: str = None, 
                          preprocessed_path: str = 'processed_data.npz',
                          force_reprocess: bool = False,
                          **preprocessing_kwargs):
    """
    Complete setup pipeline: preprocessing + environment creation
    
    Args:
        dataset_path: Path to original CSV dataset file
        preprocessed_path: Path where to save/load preprocessed data
        force_reprocess: Force reprocessing even if preprocessed data exists
        **preprocessing_kwargs: Arguments for DataProcessor
        
    Returns:
        Tuple of (train_env, test_env, processor)
        
    Raises:
        FileNotFoundError: If required files are not found
        ImportError: If data_preprocessing module is not available
        ValueError: If neither preprocessed data nor dataset path is provided
    """
    # Check if preprocessed data exists and we don't want to force reprocess
    if os.path.exists(preprocessed_path) and not force_reprocess:
        print(f"Loading existing preprocessed data from {preprocessed_path}")
        train_env = EnvironmentFactory.create_from_preprocessed(preprocessed_path, use_train_data=True)
        test_env = EnvironmentFactory.create_from_preprocessed(preprocessed_path, use_train_data=False)
        processor = None
        
    elif dataset_path and PREPROCESSOR_AVAILABLE:
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Dataset file '{dataset_path}' not found. "
                f"Please provide the correct path to your dataset file."
            )
        
        print(f"Processing dataset from {dataset_path}")
        
        # Initialize processor
        processor = DataProcessor(**preprocessing_kwargs)
        
        # Process dataset
        X_train, X_test, y_train, y_test = processor.process_dataset(dataset_path)
        
        # Save processed data
        np.savez_compressed(preprocessed_path,
                          X_train=X_train, X_test=X_test,
                          y_train=y_train, y_test=y_test)
        print(f"Preprocessed data saved to {preprocessed_path}")
        
        # Save preprocessor
        processor_path = preprocessed_path.replace('.npz', '_processor.pkl')
        processor.save_preprocessor(processor_path)
        
        # Create environments
        train_env = WorkerOnlyEnvironment(X_train, y_train)
        test_env = WorkerOnlyEnvironment(X_test, y_test)
        
    elif not PREPROCESSOR_AVAILABLE:
        raise ImportError(
            "data_preprocessing module not available and no preprocessed data found. "
            "Please ensure data_preprocessing.py is in your Python path or provide preprocessed data."
        )
    elif not dataset_path:
        raise ValueError(
            f"Neither preprocessed data file '{preprocessed_path}' exists nor dataset_path provided. "
            f"Please either run data_preprocessing.py first or provide a valid dataset_path."
        )
    else:
        raise FileNotFoundError(
            f"Dataset file '{dataset_path}' not found. "
            f"Please provide the correct path to your dataset file."
        )
    
    return train_env, test_env, processor

def main():
    """
    Example usage of the updated environments with real preprocessed data
    """
    print("Updated IDS Environment with Real Data")
    print("=" * 40)

    # Load preprocessed data and create environment
    print("Creating WorkerOnlyEnvironment with real preprocessed data...")
    worker_env = create_environment_with_real_data(
        data_path='processed_data.npz',
        use_train_data=True,
        max_episodes=100
    )

    print(f"\nWorker Environment Details:")
    print(f"- Action space: {worker_env.action_space}")
    print(f"- Observation space: {worker_env.observation_space}")
    print(f"- Data size: {worker_env.data_size}")
    print(f"- Episode length: {worker_env.episode_length}")
    print(f"- Feature dimensions: {worker_env.X_data.shape[1]}")
    print(f"- Class distribution: {np.bincount(worker_env.y_data)}")
    print(f"\n- Updated Reward Values:")
    print(f"  * Block Malicious (TP): +{worker_env.reward_block_malicious}")
    print(f"  * Allow Benign (TN): +{worker_env.reward_allow_benign}")
    print(f"  * Allow Malicious (FN): {worker_env.penalty_allow_malicious}")
    print(f"  * Block Benign (FP): {worker_env.penalty_block_benign}")
    print(f"  * Log Action: +{worker_env.reward_log}")

    # Reset environment and test with real data
    obs, info = worker_env.reset()
    print(f"\nInitial state with real data:")
    print(f"- Observation shape: {obs.shape}")
    print(f"- Current sample label: {info['true_label']} ({'Malicious' if info['true_label'] == 1 else 'Benign'})")
    print(f"- Queue length: {info['current_queue_length']:.2f}")
    print(f"- Average queue length: {info['average_queue_length']:.2f}")

    # Test different actions on real samples
    print(f"\nTesting actions on real network traffic data:")
    for i, action in enumerate([0, 1, 2]):
        if i > 0:  # Get new sample for each action test
            obs, info = worker_env.reset()
        
        obs, reward, terminated, truncated, info = worker_env.step(action)
        action_names = ['Allow', 'Block', 'Log']
        true_label = 'Malicious' if info['metrics']['true_positives'] + info['metrics']['false_negatives'] > 0 else 'Benign'
        print(f"- Action {action} ({action_names[action]}) on {true_label} traffic: reward={reward:.2f}, queue={info['current_queue_length']:.2f}")

    print(f"\nEnvironment metrics after testing: {info['metrics']}")
    
    # Demonstrate hierarchical environment if we have a worker agent
    print(f"\nTo create hierarchical environment, you would use:")
    print(f"hierarchical_env = EnvironmentFactory.create_hierarchical_environment(")
    print(f"    X_train, y_train, trained_worker_agent, max_episodes)")
    
    print(f"\n Environment successfully connected to real intrusion detection dataset!")
    print(f"   Ready for training with {worker_env.data_size} real network traffic samples.")

if __name__ == "__main__":
    main()
