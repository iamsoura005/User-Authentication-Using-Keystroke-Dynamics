from flask import Flask, render_template, request, jsonify
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import importlib
import matplotlib.pyplot as plt

# Import models
from models.manhattan import ManhattanDistance
from models.manhattan_filtered import ManhattanFilteredDistance
from models.manhattan_scaled import ManhattanScaledDistance
from models.gmm import GaussianMixtureModel
from models.svm import SVMModel
from models.utils import calculate_equal_error_rate

app = Flask(__name__)

# Global variables to store models and reference data
models = {
    'Manhattan Distance': None,
    'Manhattan Filtered Distance': None,
    'Manhattan Scaled Distance': None,
    'Gaussian Mixture Model': None,
    'SVM': None
}

# Global variable to store feature names
FEATURE_NAMES = []

# Target phrase for typing
TARGET_PHRASE = "sourasantadutta"

def load_data():
    """Load and preprocess the CMU Keystroke dataset"""
    # Check if CSV exists, otherwise try XLS
    data_path = os.path.join('data', 'keystroke.csv')
    if not os.path.exists(data_path):
        data_path = os.path.join('data', 'keystroke.xls')
        if not os.path.exists(data_path):
            print("Dataset not found. Generating sample data for demonstration.")
            # Import the sample data generator
            from data.sample_data import generate_sample_data
            # Generate sample data
            df = generate_sample_data()
            return df
    
    # Load the dataset based on file extension
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        df = pd.read_excel(data_path)
    
    return df

def preprocess_data(df):
    """Extract features from the keystroke data"""
    # Import the feature extraction function
    from models.utils import extract_keystroke_features
    
    # Extract features using the utility function
    X, y, feature_names = extract_keystroke_features(df)
    
    # Store feature names for later use
    global FEATURE_NAMES
    FEATURE_NAMES = feature_names
    
    return X, y

def train_models(X, y):
    """Train all the models with the preprocessed data"""
    # Import visualization module
    from models.visualize import plot_model_performance
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Initialize and train each model
    models['Manhattan Distance'] = ManhattanDistance()
    models['Manhattan Distance'].train(X_train, y_train)
    
    models['Manhattan Filtered Distance'] = ManhattanFilteredDistance()
    models['Manhattan Filtered Distance'].train(X_train, y_train)
    
    models['Manhattan Scaled Distance'] = ManhattanScaledDistance()
    models['Manhattan Scaled Distance'].train(X_train, y_train)
    
    models['Gaussian Mixture Model'] = GaussianMixtureModel()
    models['Gaussian Mixture Model'].train(X_train, y_train)
    
    models['SVM'] = SVMModel()
    models['SVM'].train(X_train, y_train)
    
    # Evaluate models and calculate EER
    print("\nModel Performance Evaluation:")
    print("-" * 50)
    print(f"{'Model':<30} {'Accuracy':<10} {'EER':<10}")
    print("-" * 50)
    
    # Lists to store performance metrics for visualization
    model_names = []
    accuracies = []
    eers = []
    
    for name, model in models.items():
        # Calculate accuracy
        accuracy = model.evaluate(X_test, y_test)
        
        # Calculate EER
        if name == 'SVM':
            # For SVM, use predict_proba to get scores
            scores = model.predict_proba(X_test)
        else:
            # For distance-based models, use negative distance as score
            # (higher score = more likely to be genuine)
            distances = np.array([model._calculate_distance(x) for x in X_test])
            scores = -distances
        
        eer, _ = calculate_equal_error_rate(y_test, scores)
        
        print(f"{name:<30} {accuracy:.4f}     {eer:.4f}")
        
        # Store metrics for visualization
        model_names.append(name)
        accuracies.append(accuracy)
        eers.append(eer)
    
    print("-" * 50)
    
    # Plot model performance
    plot_model_performance(model_names, accuracies, eers)
    
    # If SVM model is trained, plot feature importance
    if models['SVM'] is not None and hasattr(models['SVM'].svm, 'coef_'):
        from models.visualize import plot_feature_importance
        importances = np.abs(models['SVM'].svm.coef_[0])
        plot_feature_importance(FEATURE_NAMES, importances, 'SVM Feature Importance')

def extract_features_from_keystroke_data(keystroke_data):
    """Extract features from the keystroke data collected from the user"""
    # Import the feature extraction function
    from models.utils import extract_features_from_keystroke_data as extract_features
    
    # Use the utility function to extract features
    # Pass the global FEATURE_NAMES to ensure consistent feature extraction
    features = extract_features(keystroke_data, FEATURE_NAMES)
    
    return features

@app.route('/')
def index():
    """Render the main page"""
    # Check if feature importance visualization is available
    feature_importance_available = os.path.exists('static/feature_importance.png')
    return render_template('index.html', feature_importance_available=feature_importance_available)

@app.route('/verify', methods=['POST'])
def verify():
    """Process the keystroke data and verify the user"""
    from models.visualize import plot_keystroke_dynamics
    
    data = request.get_json()
    
    # Extract text and keystroke data
    text = data.get('text', '')
    keystroke_data = data.get('keyData', [])
    
    # Check if the text matches the target phrase
    if text.strip().lower() != TARGET_PHRASE.lower():
        return jsonify({
            'error': 'Text does not match the target phrase',
            'models': {model_name: False for model_name in models},
            'final_result': False
        })
    
    # Process keystroke data for visualization
    keydown_events = {k['key']: k['timestamp'] for k in keystroke_data if k['type'] == 'keydown'}
    keyup_events = {k['key']: k['timestamp'] for k in keystroke_data if k['type'] == 'keyup'}
    
    # Calculate hold times and flight times
    keys = []
    hold_times = []
    flight_times = []
    
    # Sort keys by timestamp
    sorted_keys = sorted(keydown_events.items(), key=lambda x: x[1])
    
    # Extract hold times
    for key, timestamp in sorted_keys:
        if key in keyup_events:
            keys.append(key)
            hold_time = keyup_events[key] - timestamp
            hold_times.append(hold_time)
    
    # Extract flight times
    for i in range(1, len(sorted_keys)):
        prev_key, prev_timestamp = sorted_keys[i-1]
        curr_key, curr_timestamp = sorted_keys[i]
        flight_time = curr_timestamp - prev_timestamp
        flight_times.append(flight_time)
    
    # Generate visualization
    plot_keystroke_dynamics(hold_times, flight_times, keys)
    
    # Extract features from keystroke data
    features = extract_features_from_keystroke_data(keystroke_data)
    
    # Run verification with each model
    results = {}
    for name, model in models.items():
        if model is not None:
            # Predict using the model
            results[name] = bool(model.predict(features)[0])
        else:
            # If model is not trained, default to False
            results[name] = False
    
    # Determine final result (majority vote)
    verified_count = sum(1 for result in results.values() if result)
    final_result = verified_count > len(models) / 2
    
    return jsonify({
        'models': results,
        'final_result': final_result,
        'visualization': '/static/keystroke_dynamics.png'
    })

def generate_initial_visualizations():
    """Generate initial visualizations with dummy data if needed"""
    from models.visualize import plot_model_performance
    
    # Check if model performance visualization exists
    if not os.path.exists('static/model_performance.png'):
        print("Generating initial model performance visualization...")
        # Create dummy data
        model_names = ['Manhattan Distance', 'Manhattan Filtered Distance', 
                      'Manhattan Scaled Distance', 'Gaussian Mixture Model', 'SVM']
        accuracies = [0.85, 0.87, 0.89, 0.82, 0.91]
        eers = [0.12, 0.10, 0.09, 0.15, 0.08]
        
        # Generate visualization
        plot_model_performance(model_names, accuracies, eers)
        print("Initial visualization generated.")

if __name__ == '__main__':
    # Load and preprocess data
    df = load_data()
    if df is not None:
        X, y = preprocess_data(df)
        train_models(X, y)
    else:
        # If data loading failed, at least generate visualizations
        generate_initial_visualizations()
    
    # Run the Flask app
    app.run(debug=True) 