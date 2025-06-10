import numpy as np
from sklearn.metrics import accuracy_score

class ManhattanScaledDistance:
    """
    Manhattan Scaled Distance model for keystroke dynamics verification.
    
    This model computes the Manhattan (L1) distance between a reference template
    and a test sample, but scales each feature by its standard deviation to give
    less weight to features with high variance.
    """
    
    def __init__(self):
        self.reference_template = None
        self.feature_scales = None
        self.threshold = None
    
    def train(self, X, y):
        """
        Train the model by computing a reference template from genuine samples
        and calculating feature scaling factors.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Features matrix with shape (n_samples, n_features)
        y : numpy.ndarray
            Labels vector with shape (n_samples,)
            1 for genuine users, 0 for impostors
        """
        # Extract genuine samples
        genuine_samples = X[y == 1]
        
        # Compute reference template as the mean of genuine samples
        self.reference_template = np.mean(genuine_samples, axis=0)
        
        # Calculate standard deviation of each feature in genuine samples
        # Add a small constant to avoid division by zero
        self.feature_scales = np.std(genuine_samples, axis=0) + 1e-10
        
        # Compute scaled distances for all samples
        distances = np.array([self._calculate_distance(x) for x in X])
        
        # Find the optimal threshold
        self._find_optimal_threshold(distances, y)
        
        return self
    
    def _calculate_distance(self, sample):
        """Calculate scaled Manhattan distance between sample and reference template"""
        # Scale the difference by the feature standard deviations
        scaled_diff = np.abs(sample - self.reference_template) / self.feature_scales
        
        return np.sum(scaled_diff)
    
    def _find_optimal_threshold(self, distances, y):
        """
        Find the optimal threshold that maximizes accuracy
        
        Parameters:
        -----------
        distances : numpy.ndarray
            Distances from reference template
        y : numpy.ndarray
            True labels (1 for genuine, 0 for impostor)
        """
        # Try different thresholds and find the one that maximizes accuracy
        thresholds = np.linspace(np.min(distances), np.max(distances), 100)
        best_accuracy = 0
        best_threshold = 0
        
        for threshold in thresholds:
            # Predict genuine if distance is below threshold
            predictions = (distances <= threshold).astype(int)
            accuracy = accuracy_score(y, predictions)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        
        self.threshold = best_threshold
        print(f"Manhattan Scaled Distance - Optimal threshold: {self.threshold:.4f}, Accuracy: {best_accuracy:.4f}")
    
    def predict(self, X):
        """
        Predict if the samples are from the genuine user
        
        Parameters:
        -----------
        X : numpy.ndarray
            Features matrix with shape (n_samples, n_features)
            
        Returns:
        --------
        numpy.ndarray
            Predictions (1 for genuine, 0 for impostor)
        """
        if self.reference_template is None or self.threshold is None or self.feature_scales is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Calculate distances
        distances = np.array([self._calculate_distance(x) for x in X])
        
        # Predict genuine if distance is below threshold
        return (distances <= self.threshold).astype(int)
    
    def evaluate(self, X, y):
        """
        Evaluate the model on test data
        
        Parameters:
        -----------
        X : numpy.ndarray
            Features matrix with shape (n_samples, n_features)
        y : numpy.ndarray
            True labels (1 for genuine, 0 for impostor)
            
        Returns:
        --------
        float
            Accuracy score
        """
        predictions = self.predict(X)
        return accuracy_score(y, predictions) 