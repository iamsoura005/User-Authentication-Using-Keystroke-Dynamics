import numpy as np
from sklearn.metrics import accuracy_score

class ManhattanDistance:
    """
    Manhattan Distance model for keystroke dynamics verification.
    
    This model computes the Manhattan (L1) distance between a reference template
    and a test sample. If the distance is below a threshold, the user is verified.
    """
    
    def __init__(self):
        self.reference_template = None
        self.threshold = None
    
    def train(self, X, y):
        """
        Train the model by computing a reference template from genuine samples.
        
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
        
        # Compute distances for all samples
        distances = np.array([self._calculate_distance(x) for x in X])
        
        # Find the optimal threshold
        self._find_optimal_threshold(distances, y)
        
        return self
    
    def _calculate_distance(self, sample):
        """Calculate Manhattan distance between sample and reference template"""
        return np.sum(np.abs(sample - self.reference_template))
    
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
        print(f"Manhattan Distance - Optimal threshold: {self.threshold:.4f}, Accuracy: {best_accuracy:.4f}")
    
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
        if self.reference_template is None or self.threshold is None:
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