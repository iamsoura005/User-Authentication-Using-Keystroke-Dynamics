import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def extract_keystroke_features(df):
    """
    Extract keystroke dynamics features from the raw dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw keystroke data
        
    Returns:
    --------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Label vector (1 for genuine, 0 for impostor)
    feature_names : list
        Names of the extracted features
    """
    # This function should be adapted based on the actual structure of the CMU dataset
    # For now, we'll implement a generic version that extracts common keystroke features
    
    # Assuming the dataset has columns for:
    # - subject_id: identifier for each user
    # - session_id: identifier for each typing session
    # - key: the key pressed
    # - press_time: timestamp of key press
    # - release_time: timestamp of key release
    
    # Create an empty list to store feature vectors
    feature_vectors = []
    labels = []
    feature_names = []
    
    # Process each user's data
    for subject_id in df['subject_id'].unique():
        # Get data for this user
        user_data = df[df['subject_id'] == subject_id]
        
        # Process each session
        for session_id in user_data['session_id'].unique():
            session_data = user_data[user_data['session_id'] == session_id]
            
            # Sort by press_time to ensure correct sequence
            session_data = session_data.sort_values('press_time')
            
            # Extract features
            features = {}
            
            # 1. Hold times (key release - key press)
            for _, row in session_data.iterrows():
                key = row['key']
                hold_time = row['release_time'] - row['press_time']
                feature_name = f"hold_time_{key}"
                features[feature_name] = hold_time
                if feature_name not in feature_names:
                    feature_names.append(feature_name)
            
            # 2. Flight times (time between consecutive key presses)
            press_times = session_data['press_time'].values
            for i in range(1, len(press_times)):
                key_pair = f"{session_data['key'].values[i-1]}_{session_data['key'].values[i]}"
                flight_time = press_times[i] - press_times[i-1]
                feature_name = f"flight_time_{key_pair}"
                features[feature_name] = flight_time
                if feature_name not in feature_names:
                    feature_names.append(feature_name)
            
            # Add feature vector to list
            feature_vector = [features.get(name, 0) for name in feature_names]
            feature_vectors.append(feature_vector)
            
            # Add label (1 for genuine user, 0 for impostor)
            # Assuming the first subject is the genuine user and others are impostors
            label = 1 if subject_id == df['subject_id'].unique()[0] else 0
            labels.append(label)
    
    # Convert to numpy arrays
    X = np.array(feature_vectors)
    y = np.array(labels)
    
    return X, y, feature_names

def preprocess_data(X, y, test_size=0.3, random_state=42):
    """
    Preprocess the data by scaling features and splitting into train/test sets.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Label vector
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    X_train : numpy.ndarray
        Training features
    X_test : numpy.ndarray
        Testing features
    y_train : numpy.ndarray
        Training labels
    y_test : numpy.ndarray
        Testing labels
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler for feature normalization
    """
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler

def extract_features_from_keystroke_data(keystroke_data, feature_names, scaler=None):
    """
    Extract features from keystroke data collected from the user interface.
    
    Parameters:
    -----------
    keystroke_data : list
        List of keystroke events (key down/up) with timestamps
    feature_names : list
        Names of features to extract
    scaler : sklearn.preprocessing.StandardScaler, optional
        Scaler for feature normalization
        
    Returns:
    --------
    features : numpy.ndarray
        Extracted features (scaled if scaler is provided)
    """
    # Process keystroke data
    keydown_events = {}
    keyup_events = {}
    
    for event in keystroke_data:
        key = event['key']
        timestamp = event['timestamp']
        
        if event['type'] == 'keydown':
            keydown_events[key] = timestamp
        elif event['type'] == 'keyup':
            keyup_events[key] = timestamp
    
    # Extract features
    features = {}
    
    # 1. Hold times
    for key in keydown_events:
        if key in keyup_events:
            hold_time = keyup_events[key] - keydown_events[key]
            feature_name = f"hold_time_{key}"
            features[feature_name] = hold_time
    
    # 2. Flight times
    keys = list(keydown_events.keys())
    timestamps = [keydown_events[k] for k in keys]
    
    # Sort by timestamp
    keys_sorted = [k for _, k in sorted(zip(timestamps, keys))]
    
    for i in range(1, len(keys_sorted)):
        key_pair = f"{keys_sorted[i-1]}_{keys_sorted[i]}"
        flight_time = keydown_events[keys_sorted[i]] - keydown_events[keys_sorted[i-1]]
        feature_name = f"flight_time_{key_pair}"
        features[feature_name] = flight_time
    
    # Create feature vector with the same structure as training data
    feature_vector = np.array([features.get(name, 0) for name in feature_names]).reshape(1, -1)
    
    # Scale if scaler is provided
    if scaler is not None:
        feature_vector = scaler.transform(feature_vector)
    
    return feature_vector

def calculate_equal_error_rate(y_true, scores):
    """
    Calculate the Equal Error Rate (EER) for a verification system.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True labels (1 for genuine, 0 for impostor)
    scores : numpy.ndarray
        Verification scores (higher means more likely to be genuine)
        
    Returns:
    --------
    eer : float
        Equal Error Rate
    threshold : float
        Threshold at EER
    """
    # Sort scores and corresponding labels
    indices = np.argsort(scores)
    y_true_sorted = y_true[indices]
    
    # Calculate False Accept Rate (FAR) and False Reject Rate (FRR) at different thresholds
    far = np.zeros(len(y_true_sorted) + 1)
    frr = np.zeros(len(y_true_sorted) + 1)
    
    for i in range(len(y_true_sorted) + 1):
        # Threshold: scores >= scores[i]
        predictions = np.zeros(len(y_true_sorted))
        if i < len(y_true_sorted):
            predictions[indices[i:]] = 1
        
        # Calculate FAR and FRR
        far[i] = np.sum((predictions == 1) & (y_true == 0)) / np.sum(y_true == 0)
        frr[i] = np.sum((predictions == 0) & (y_true == 1)) / np.sum(y_true == 1)
    
    # Find the threshold where FAR = FRR
    abs_diff = np.abs(far - frr)
    min_index = np.argmin(abs_diff)
    eer = (far[min_index] + frr[min_index]) / 2
    
    # Get the threshold at EER
    if min_index < len(scores):
        threshold = scores[indices[min_index]]
    else:
        threshold = scores[indices[-1]] + 1e-10
    
    return eer, threshold 