import numpy as np
from sklearn.metrics import accuracy_score

class ManhattanFilteredDistance:
    """
    Manhattan Filtered Distance model for keystroke dynamics verification.
    
    This model computes the Manhattan (L1) distance between a reference template
    and a test sample, but only considers the most consistent features.
    Features with high variance are filtered out.
    """
    
    def __init__(self, filter_ratio=0.5):
        self.reference_template = None
        self.threshold = None
        self.feature_mask = None
        self.filter_ratio = filter_ratio  # Ratio of features to keep
    
    def train(self, X, y):
        """
        Train the model by computing a reference template from genuine samples
        and filtering out high-variance features.
        
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
        
        # Calculate variance of each feature in genuine samples
        feature_variances = np.var(genuine_samples, axis=0)
        
        # Create a mask to keep only the most consistent features (lowest variance)
        num_features_to_keep = int(self.filter_ratio * len(feature_variances))
        feature_indices = np.argsort(feature_variances)[:num_features_to_keep]
        self.feature_mask = np.zeros(X.shape[1], dtype=bool)
        self.feature_mask[feature_indices] = True
        
        # Compute distances for all samples using filtered features
        distances = np.array([self._calculate_distance(x) for x in X])
        
        # Find the optimal threshold
        self._find_optimal_threshold(distances, y)
        
        return self
    
    def _calculate_distance(self, sample):
        """Calculate Manhattan distance between sample and reference template using filtered features"""
        # Apply feature mask to both sample and reference template
        filtered_sample = sample[self.feature_mask]
        filtered_template = self.reference_template[self.feature_mask]
        
        return np.sum(np.abs(filtered_sample - filtered_template))
    
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
        print(f"Manhattan Filtered Distance - Optimal threshold: {self.threshold:.4f}, Accuracy: {best_accuracy:.4f}")
    
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
        if self.reference_template is None or self.threshold is None or self.feature_mask is None:
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