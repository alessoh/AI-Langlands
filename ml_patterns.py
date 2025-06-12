"""
Machine learning analysis for discovering patterns in elliptic curves.

This module implements:
- PCA for dimensionality reduction and visualization
- Neural networks for rank prediction
- Pattern discovery algorithms
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class MurmurationAnalyzer:
    """Analyzes elliptic curve data to discover murmuration patterns."""
    
    def __init__(self, traces: np.ndarray, ranks: List[int]):
        """
        Initialize the analyzer.
        
        Args:
            traces: Frobenius traces array (n_curves, n_primes)
            ranks: List of ranks for each curve
        """
        self.traces = traces
        self.ranks = np.array(ranks)
        self.scaler = StandardScaler()
        self.pca = None
        self.model = None
        
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for machine learning.
        
        Returns:
            Scaled features and rank labels
        """
        # Scale the traces
        X_scaled = self.scaler.fit_transform(self.traces)
        
        # Convert ranks to categorical
        y = self.ranks
        
        return X_scaled, y
    
    def apply_pca(self, n_components: int = 2) -> np.ndarray:
        """
        Apply PCA to reduce dimensionality and reveal structure.
        
        Args:
            n_components: Number of principal components
            
        Returns:
            Transformed data
        """
        X_scaled, _ = self.prepare_data()
        
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Print explained variance
        explained_var = self.pca.explained_variance_ratio_
        print(f"\nPCA Explained Variance Ratios: {explained_var}")
        print(f"Total Explained Variance: {sum(explained_var):.3f}")
        
        return X_pca
    
    def build_neural_network(self, input_dim: int, num_classes: int) -> models.Model:
        """
        Build a neural network for rank prediction.
        
        Args:
            input_dim: Number of input features
            num_classes: Number of rank classes
            
        Returns:
            Compiled Keras model
        """
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_rank_predictor(self, test_size: float = 0.2, 
                           epochs: int = 50, 
                           verbose: int = 0) -> Dict:
        """
        Train a neural network to predict rank from traces.
        
        Args:
            test_size: Fraction of data for testing
            epochs: Number of training epochs
            verbose: Verbosity level
            
        Returns:
            Dictionary with training results
        """
        X_scaled, y = self.prepare_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Build and train model
        num_classes = len(np.unique(y))
        self.model = self.build_neural_network(X_train.shape[1], num_classes)
        
        # Use early stopping
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        
        history = self.model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stop],
            verbose=verbose
        )
        
        # Evaluate
        y_pred = np.argmax(self.model.predict(X_test), axis=1)
        
        results = {
            'accuracy': np.mean(y_pred == y_test),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'history': history.history,
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        print(f"\nRank Prediction Accuracy: {results['accuracy']:.3f}")
        
        return results
    
    def find_discriminative_primes(self, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Find which primes are most useful for distinguishing ranks.
        
        Args:
            top_k: Number of top primes to return
            
        Returns:
            List of (prime_index, importance_score) tuples
        """
        # Use the trained model's first layer weights
        if self.model is None:
            raise ValueError("Must train model first")
        
        # Get input layer weights
        weights = self.model.layers[0].get_weights()[0]
        
        # Compute importance scores (sum of absolute weights)
        importance = np.sum(np.abs(weights), axis=1)
        
        # Get top k indices
        top_indices = np.argsort(importance)[-top_k:][::-1]
        
        # Return with scores
        return [(idx, importance[idx]) for idx in top_indices]
    
    def detect_patterns(self) -> Dict:
        """
        Run comprehensive pattern detection analysis.
        
        Returns:
            Dictionary containing discovered patterns
        """
        patterns = {}
        
        # 1. Clustering structure in PCA
        X_pca = self.apply_pca(n_components=10)
        
        # Compute cluster separability
        from sklearn.metrics import silhouette_score
        silhouette = silhouette_score(X_pca[:, :2], self.ranks)
        patterns['cluster_separability'] = silhouette
        
        # 2. Rank predictability
        ml_results = self.train_rank_predictor()
        patterns['rank_predictability'] = ml_results['accuracy']
        patterns['ml_results'] = ml_results
        
        # 3. Find most important primes
        important_primes = self.find_discriminative_primes()
        patterns['important_primes'] = important_primes
        
        # 4. Statistical patterns
        patterns['trace_statistics'] = self._compute_trace_statistics()
        
        return patterns
    
    def _compute_trace_statistics(self) -> Dict:
        """Compute statistical properties of traces by rank."""
        stats = {}
        
        for rank in np.unique(self.ranks):
            mask = self.ranks == rank
            rank_traces = self.traces[mask]
            
            stats[f'rank_{rank}'] = {
                'mean': np.mean(rank_traces, axis=0),
                'std': np.std(rank_traces, axis=0),
                'skewness': self._skewness(rank_traces),
                'kurtosis': self._kurtosis(rank_traces)
            }
            
        return stats
    
    def _skewness(self, data: np.ndarray) -> np.ndarray:
        """Compute skewness along columns."""
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return np.mean(((data - mean) / std) ** 3, axis=0)
    
    def _kurtosis(self, data: np.ndarray) -> np.ndarray:
        """Compute kurtosis along columns."""
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return np.mean(((data - mean) / std) ** 4, axis=0) - 3

def discover_hidden_structure(traces: np.ndarray, ranks: List[int]) -> Dict:
    """
    Main function to discover hidden mathematical structure.
    
    Args:
        traces: Frobenius traces
        ranks: Curve ranks
        
    Returns:
        Dictionary of discovered patterns
    """
    analyzer = MurmurationAnalyzer(traces, ranks)
    patterns = analyzer.detect_patterns()
    
    # Add interpretation
    interpretation = interpret_patterns(patterns)
    patterns['interpretation'] = interpretation
    
    return patterns

def interpret_patterns(patterns: Dict) -> str:
    """
    Provide mathematical interpretation of discovered patterns.
    
    Args:
        patterns: Dictionary of patterns from analyzer
        
    Returns:
        Human-readable interpretation
    """
    interpretation = []
    
    # Cluster separability
    sep = patterns['cluster_separability']
    if sep > 0.5:
        interpretation.append(
            f"Strong cluster separation (silhouette={sep:.3f}) indicates "
            "that elliptic curves naturally group by rank in trace space."
        )
    
    # Rank predictability
    acc = patterns['rank_predictability']
    if acc > 0.7:
        interpretation.append(
            f"High rank prediction accuracy ({acc:.1%}) demonstrates that "
            "local information (Frobenius traces) encodes global properties (rank)."
        )
    
    # Important primes
    if 'important_primes' in patterns:
        top_prime_idx = patterns['important_primes'][0][0]
        interpretation.append(
            f"Prime at index {top_prime_idx} is most discriminative for rank, "
            "suggesting special arithmetic significance."
        )
    
    # Connection to Langlands
    interpretation.append(
        "\nThese patterns support the Langlands philosophy: "
        "arithmetic objects (curves) have analytic shadows (trace patterns) "
        "that reveal their deep structure."
    )
    
    return "\n\n".join(interpretation)