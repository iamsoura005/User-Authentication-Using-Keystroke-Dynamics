import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

class SVMModel:
    """
    Support Vector Machine model for keystroke dynamics verification.
    
    This model uses an SVM classifier to distinguish between genuine users
    and impostors based on keystroke features.
    """
    
    def __init__(self, kernel='rbf', C=1.0):
        self.svm = None
        self.scaler = StandardScaler()
        self.kernel = kernel
        self.C = C
    
    def train(self, X, y):
        """
        Train the SVM model with the provided data.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Features matrix with shape (n_samples, n_features)
        y : numpy.ndarray
            Labels vector with shape (n_samples,)
            1 for genuine users, 0 for impostors
        """
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize and train the SVM
        self.svm = SVC(
            kernel=self.kernel,
            C=self.C,
            probability=True,
            random_state=42
        )
        self.svm.fit(X_scaled, y)
        
        return self
    
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
        if self.svm is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Scale the features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        return self.svm.predict(X_scaled)
    
    def predict_proba(self, X):
        """
        Predict probability of samples being from the genuine user
        
        Parameters:
        -----------
        X : numpy.ndarray
            Features matrix with shape (n_samples, n_features)
            
        Returns:
        --------
        numpy.ndarray
            Probability of being genuine for each sample
        """
        if self.svm is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Scale the features
        X_scaled = self.scaler.transform(X)
        
        # Get probabilities
        probas = self.svm.predict_proba(X_scaled)
        
        # Return probability of being genuine (class 1)
        return probas[:, 1]
    
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
        accuracy = accuracy_score(y, predictions)
        print(f"SVM - Accuracy: {accuracy:.4f}")
        return accuracy 