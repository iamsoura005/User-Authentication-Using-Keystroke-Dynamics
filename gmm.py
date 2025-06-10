import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score

class GaussianMixtureModel:
    """
    Gaussian Mixture Model for keystroke dynamics verification.
    
    This model uses two GMMs: one for genuine users and one for impostors.
    Verification is based on the likelihood ratio between these models.
    """
    
    def __init__(self, n_components=3):
        self.genuine_gmm = None
        self.impostor_gmm = None
        self.threshold = None
        self.n_components = n_components
    
    def train(self, X, y):
        """
        Train two GMMs: one for genuine users and one for impostors.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Features matrix with shape (n_samples, n_features)
        y : numpy.ndarray
            Labels vector with shape (n_samples,)
            1 for genuine users, 0 for impostors
        """
        # Extract genuine and impostor samples
        genuine_samples = X[y == 1]
        impostor_samples = X[y == 0]
        
        # Train GMM for genuine users
        self.genuine_gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type='full',
            random_state=42
        )
        self.genuine_gmm.fit(genuine_samples)
        
        # Train GMM for impostors
        self.impostor_gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type='full',
            random_state=42
        )
        self.impostor_gmm.fit(impostor_samples)
        
        # Compute log-likelihood ratios for all samples
        log_likelihood_ratios = self._calculate_log_likelihood_ratios(X)
        
        # Find the optimal threshold
        self._find_optimal_threshold(log_likelihood_ratios, y)
        
        return self
    
    def _calculate_log_likelihood_ratios(self, X):
        """
        Calculate log-likelihood ratio between genuine and impostor models
        
        Parameters:
        -----------
        X : numpy.ndarray
            Features matrix with shape (n_samples, n_features)
            
        Returns:
        --------
        numpy.ndarray
            Log-likelihood ratios
        """
        # Calculate log-likelihood for both models
        genuine_log_probs = self.genuine_gmm.score_samples(X)
        impostor_log_probs = self.impostor_gmm.score_samples(X)
        
        # Calculate log-likelihood ratio
        return genuine_log_probs - impostor_log_probs
    
    def _find_optimal_threshold(self, log_likelihood_ratios, y):
        """
        Find the optimal threshold that maximizes accuracy
        
        Parameters:
        -----------
        log_likelihood_ratios : numpy.ndarray
            Log-likelihood ratios
        y : numpy.ndarray
            True labels (1 for genuine, 0 for impostor)
        """
        # Try different thresholds and find the one that maximizes accuracy
        thresholds = np.linspace(
            np.min(log_likelihood_ratios),
            np.max(log_likelihood_ratios),
            100
        )
        best_accuracy = 0
        best_threshold = 0
        
        for threshold in thresholds:
            # Predict genuine if log-likelihood ratio is above threshold
            predictions = (log_likelihood_ratios >= threshold).astype(int)
            accuracy = accuracy_score(y, predictions)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        
        self.threshold = best_threshold
        print(f"GMM - Optimal threshold: {self.threshold:.4f}, Accuracy: {best_accuracy:.4f}")
    
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
        if self.genuine_gmm is None or self.impostor_gmm is None or self.threshold is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Calculate log-likelihood ratios
        log_likelihood_ratios = self._calculate_log_likelihood_ratios(X)
        
        # Predict genuine if log-likelihood ratio is above threshold
        return (log_likelihood_ratios >= self.threshold).astype(int)
    
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