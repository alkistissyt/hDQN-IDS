# -*- coding: utf-8 -*-
"""
Data Preprocessing Script for Intrusion Detection Datasets
Handles loading, cleaning, scaling, and adaptive PCA transformation
Supports: CICIDS2017, NF-ToN-IoT, Edge-IIoTset, BoT-IoT, and other similar datasets
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pickle
import psutil
import os
import gc
from typing import Tuple, Optional, Union

class DataProcessor:
    def __init__(self, 
                 variance_threshold: float = 0.95, 
                 min_components: int = 10, 
                 max_components: int = 100):
        """
        Initialize the data processor with adaptive PCA parameters

        Args:
            variance_threshold: Target percentage of variance to preserve (0.0-1.0)
            min_components: Minimum number of PCA components to retain
            max_components: Maximum number of PCA components to retain
        """
        self.variance_threshold = variance_threshold
        self.min_components = min_components
        self.max_components = max_components
        self.optimal_components = None
        
        self.scaler = MinMaxScaler()
        self.pca = None  # Will be initialized with optimal components
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.memory_usage = {}

    def log_memory_usage(self, stage: str):
        """Log current memory usage"""
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.memory_usage[stage] = memory_mb
        print(f"Memory usage at {stage}: {memory_mb:.2f} MB")

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load intrusion detection dataset CSV file

        Args:
            file_path: Path to the CSV file

        Returns:
            Loaded DataFrame
        """
        print("Loading dataset...")
        self.log_memory_usage("before_loading")

        try:
            # Try loading with different encodings if needed
            df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_path, encoding='latin-1', low_memory=False)
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='iso-8859-1', low_memory=False)

        print(f"Dataset loaded: {df.shape}")
        self.log_memory_usage("after_loading")
        return df

    def detect_label_column(self, df: pd.DataFrame) -> str:
        """
        Automatically detect the label column in the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            Name of the label column
        """
        # Common label column names across different datasets
        possible_label_cols = [
            'Label', 'label', 'LABEL',
            'Class', 'class', 'CLASS', 
            'Attack', 'attack', 'ATTACK',
            'Category', 'category', 'CATEGORY',
            'Target', 'target', 'TARGET'
        ]
        
        for col in possible_label_cols:
            if col in df.columns:
                print(f"Detected label column: '{col}'")
                return col
        
        # If no standard label column found, use the last column
        last_col = df.columns[-1]
        print(f"No standard label column found. Using last column: '{last_col}'")
        return last_col

    def clean_data(self, df: pd.DataFrame, label_col: str) -> pd.DataFrame:
        """
        Clean the dataset by handling NaNs, infs, and problematic columns

        Args:
            df: Input DataFrame
            label_col: Name of the label column

        Returns:
            Cleaned DataFrame
        """
        print("Cleaning data...")
        initial_shape = df.shape

        # Handle column names (remove leading/trailing spaces)
        df.columns = df.columns.str.strip()

        # Replace infinite values with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Handle NaN values
        print(f"NaN values before cleaning: {df.isnull().sum().sum()}")

        # For numeric columns, fill NaN with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != label_col:  # Don't modify the label column
                df[col].fillna(df[col].median(), inplace=True)

        # Drop constant columns (columns with only one unique value)
        constant_cols = []
        for col in df.columns:
            if col != label_col and df[col].nunique() <= 1:
                constant_cols.append(col)

        if constant_cols:
            print(f"Dropping {len(constant_cols)} constant columns: {constant_cols}")
            df.drop(columns=constant_cols, inplace=True)

        # Drop columns with too many unique values (likely IDs or timestamps)
        high_cardinality_cols = []
        for col in df.columns:
            if col != label_col and df[col].dtype == 'object' and df[col].nunique() > 1000:
                high_cardinality_cols.append(col)

        if high_cardinality_cols:
            print(f"Dropping {len(high_cardinality_cols)} high cardinality columns: {high_cardinality_cols}")
            df.drop(columns=high_cardinality_cols, inplace=True)

        print(f"Data shape after cleaning: {df.shape} (was {initial_shape})")
        self.log_memory_usage("after_cleaning")
        return df

    def convert_to_binary_labels(self, df: pd.DataFrame, label_col: str) -> pd.DataFrame:
        """
        Convert multi-class labels to binary (0: benign, 1: attack)
        Works with multiple dataset formats

        Args:
            df: DataFrame with label column
            label_col: Name of the label column

        Returns:
            DataFrame with binary labels
        """
        print("Converting labels to binary...")

        # Print unique labels before conversion
        unique_labels = df[label_col].unique()
        print(f"Unique labels found: {unique_labels}")
        print(f"Label distribution:\n{df[label_col].value_counts()}")

        # Define benign/normal patterns for different datasets
        benign_patterns = [
            'BENIGN', 'NORMAL', 'LEGITIMATE', 'BACKGROUND', 
            'benign', 'normal', 'legitimate', 'background',
            'Benign', 'Normal', 'Legitimate', 'Background',
            '0', 0, 'No', 'no', 'NO'
        ]

        # Convert to binary: 0 for benign, 1 for any attack
        def is_benign(label):
            return str(label).strip() in benign_patterns

        df[label_col] = df[label_col].apply(lambda x: 0 if is_benign(x) else 1)

        # Rename to standard 'Label' for consistency
        if label_col != 'Label':
            df = df.rename(columns={label_col: 'Label'})

        print(f"Binary label distribution:\n{df['Label'].value_counts()}")
        return df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and labels for training

        Args:
            df: Cleaned DataFrame

        Returns:
            Tuple of (features, labels)
        """
        print("Preparing features...")

        # Separate features and labels
        if 'Label' not in df.columns:
            raise ValueError("Label column not found in dataset")

        X = df.drop('Label', axis=1)
        y = df['Label'].values

        # Convert all feature columns to numeric
        for col in X.columns:
            if X[col].dtype == 'object':
                # Try to convert string columns to numeric
                X[col] = pd.to_numeric(X[col], errors='coerce')
                X[col].fillna(X[col].median(), inplace=True)

        # Store feature column names
        self.feature_columns = X.columns.tolist()

        # Convert to numpy array and ensure float32 for memory efficiency
        X = X.values.astype(np.float32)
        y = y.astype(np.int32)

        print(f"Feature matrix shape: {X.shape}")
        print(f"Label vector shape: {y.shape}")

        return X, y

    def scale_features(self, X_train: np.ndarray, X_test: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Scale features using MinMaxScaler

        Args:
            X_train: Training features
            X_test: Test features (optional)

        Returns:
            Scaled features
        """
        print("Scaling features...")

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = None

        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)

        print("Feature scaling completed")
        return X_train_scaled, X_test_scaled

    def determine_optimal_components(self, X_train: np.ndarray) -> int:
        """
        Determine optimal number of PCA components based on variance threshold
        
        Args:
            X_train: Training features (scaled)
            
        Returns:
            Optimal number of components
        """
        print(f"Determining optimal PCA components...")
        print(f"Target variance preservation: {self.variance_threshold:.1%}")
        
        # Create temporary PCA with maximum possible components
        max_possible_components = min(X_train.shape[0], X_train.shape[1], self.max_components)
        temp_pca = PCA(n_components=max_possible_components)
        temp_pca.fit(X_train)
        
        # Calculate cumulative explained variance
        cumulative_variance = np.cumsum(temp_pca.explained_variance_ratio_)
        
        # Find minimum components that meet variance threshold
        optimal_idx = np.argmax(cumulative_variance >= self.variance_threshold)
        optimal_components = max(optimal_idx + 1, self.min_components)
        
        # Ensure we don't exceed maximum
        optimal_components = min(optimal_components, self.max_components)
        
        achieved_variance = cumulative_variance[optimal_components - 1]
        
        print(f"Optimal components determined: {optimal_components}")
        print(f"Achieved variance preservation: {achieved_variance:.1%}")
        print(f"Component range: [{self.min_components}, {self.max_components}]")
        
        self.optimal_components = optimal_components
        return optimal_components

    def apply_adaptive_pca(self, X_train: np.ndarray, X_test: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply PCA with adaptive component selection based on variance threshold

        Args:
            X_train: Training features (scaled)
            X_test: Test features (scaled, optional)

        Returns:
            PCA-transformed features
        """
        # Determine optimal number of components
        optimal_components = self.determine_optimal_components(X_train)
        
        # Initialize PCA with optimal components
        self.pca = PCA(n_components=optimal_components)
        
        print(f"Applying adaptive PCA with {optimal_components} components...")

        X_train_pca = self.pca.fit_transform(X_train)
        X_test_pca = None

        if X_test is not None:
            X_test_pca = self.pca.transform(X_test)

        explained_variance_ratio = self.pca.explained_variance_ratio_
        total_variance = np.sum(explained_variance_ratio)

        print(f"Adaptive PCA completed:")
        print(f"- Components selected: {optimal_components}")
        print(f"- Explained variance ratio: {total_variance:.4f}")
        print(f"- Dimensionality: {X_train.shape[1]} → {X_train_pca.shape[1]}")
        print(f"- Compression ratio: {X_train_pca.shape[1]/X_train.shape[1]:.3f}")

        return X_train_pca, X_test_pca

    def downcast_integers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Downcast int64 to int32 for memory efficiency

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with downcasted integers
        """
        print("Downcasting integers...")

        int64_cols = df.select_dtypes(include=['int64']).columns
        for col in int64_cols:
            df[col] = pd.to_numeric(df[col], downcast='integer')

        return df

    def process_dataset(self, file_path: str, test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Complete data processing pipeline with adaptive PCA

        Args:
            file_path: Path to the CSV file
            test_size: Proportion of test set
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        print("Starting complete data processing pipeline...")
        print(f"Dataset: {os.path.basename(file_path)}")
        print(f"Adaptive PCA settings: variance_threshold={self.variance_threshold:.1%}, "
              f"components_range=[{self.min_components}, {self.max_components}]")
        self.log_memory_usage("start")

        # Load data
        df = self.load_data(file_path)

        # Detect label column
        label_col = self.detect_label_column(df)

        # Clean data
        df = self.clean_data(df, label_col)

        # Downcast integers
        df = self.downcast_integers(df)

        # Convert labels to binary
        df = self.convert_to_binary_labels(df, label_col)

        # Prepare features
        X, y = self.prepare_features(df)

        # Free memory
        del df
        gc.collect()
        self.log_memory_usage("after_feature_preparation")

        # Split data
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)

        # Apply adaptive PCA
        X_train_final, X_test_final = self.apply_adaptive_pca(X_train_scaled, X_test_scaled)

        # Convert to float32 for memory efficiency
        X_train_final = X_train_final.astype(np.float32)
        X_test_final = X_test_final.astype(np.float32)
        y_train = y_train.astype(np.int32)
        y_test = y_test.astype(np.int32)

        self.log_memory_usage("final")

        print("\nData processing completed!")
        print(f"Training set: X={X_train_final.shape}, y={y_train.shape}")
        print(f"Test set: X={X_test_final.shape}, y={y_test.shape}")
        print(f"Class distribution in training set: {np.bincount(y_train)}")
        print(f"Class distribution in test set: {np.bincount(y_test)}")
        print(f"Final feature dimensions: {X_train_final.shape[1]} (adaptive PCA)")

        return X_train_final, X_test_final, y_train, y_test

    def save_preprocessor(self, filepath: str):
        """Save the preprocessor objects for later use"""
        preprocessor_data = {
            'scaler': self.scaler,
            'pca': self.pca,
            'feature_columns': self.feature_columns,
            'optimal_components': self.optimal_components,
            'variance_threshold': self.variance_threshold,
            'min_components': self.min_components,
            'max_components': self.max_components
        }

        with open(filepath, 'wb') as f:
            pickle.dump(preprocessor_data, f)

        print(f"Preprocessor saved to {filepath}")

    def load_preprocessor(self, filepath: str):
        """Load the preprocessor objects"""
        with open(filepath, 'rb') as f:
            preprocessor_data = pickle.load(f)

        self.scaler = preprocessor_data['scaler']
        self.pca = preprocessor_data['pca']
        self.feature_columns = preprocessor_data['feature_columns']
        self.optimal_components = preprocessor_data.get('optimal_components', None)
        self.variance_threshold = preprocessor_data.get('variance_threshold', 0.95)
        self.min_components = preprocessor_data.get('min_components', 10)
        self.max_components = preprocessor_data.get('max_components', 100)

        print(f"Preprocessor loaded from {filepath}")

def main():
    """
    Main function to run data preprocessing
    Works with multiple intrusion detection datasets
    """
    print("Generic Intrusion Detection Data Preprocessing")
    print("Supports: CICIDS2017, NF-ToN-IoT, Edge-IIoTset, BoT-IoT")
    print("=" * 60)

    # Initialize processor with adaptive PCA settings
    processor = DataProcessor(
        variance_threshold=0.95,  # Preserve 95% of variance
        min_components=10,        # Minimum 10 components
        max_components=50         # Maximum 50 components
    )

    # Example file path - adjust based on your setup
    file_path = input("Enter path to your dataset CSV file: ")

    try:
        # Process the dataset
        X_train, X_test, y_train, y_test = processor.process_dataset(file_path)

        # Save processed data
        output_name = f"processed_{os.path.splitext(os.path.basename(file_path))[0]}.npz"
        np.savez_compressed(output_name,
                          X_train=X_train, X_test=X_test,
                          y_train=y_train, y_test=y_test)

        # Save preprocessor
        preprocessor_name = f"preprocessor_{os.path.splitext(os.path.basename(file_path))[0]}.pkl"
        processor.save_preprocessor(preprocessor_name)

        print(f"\nData preprocessing completed successfully!")
        print(f"Files saved:")
        print(f"- {output_name}: Processed training and test data")
        print(f"- {preprocessor_name}: Scaler and PCA objects for future use")

        # Print memory usage summary
        print(f"\nMemory Usage Summary:")
        for stage, usage in processor.memory_usage.items():
            print(f"{stage}: {usage:.2f} MB")

        # Print adaptive PCA summary
        print(f"\nAdaptive PCA Summary:")
        print(f"- Optimal components selected: {processor.optimal_components}")
        print(f"- Variance threshold: {processor.variance_threshold:.1%}")
        print(f"- Achieved variance: {np.sum(processor.pca.explained_variance_ratio_):.1%}")

    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        print("Please ensure you have provided the correct path to your dataset.")
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        print("Please check your dataset format and try again.")

if __name__ == "__main__":
    main()
