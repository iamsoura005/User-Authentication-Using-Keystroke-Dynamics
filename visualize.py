import matplotlib.pyplot as plt
import numpy as np
import os

def plot_model_performance(model_names, accuracies, eers):
    """
    Plot the performance of different models.
    
    Parameters:
    -----------
    model_names : list
        Names of the models
    accuracies : list
        Accuracy scores for each model
    eers : list
        Equal Error Rates for each model
    """
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Set width of bars
    barWidth = 0.35
    
    # Set position of bars on X axis
    r1 = np.arange(len(model_names))
    r2 = [x + barWidth for x in r1]
    
    # Create bars
    plt.bar(r1, accuracies, width=barWidth, label='Accuracy', color='skyblue')
    plt.bar(r2, eers, width=barWidth, label='EER', color='lightcoral')
    
    # Add labels and title
    plt.xlabel('Models', fontweight='bold', fontsize=12)
    plt.ylabel('Score', fontweight='bold', fontsize=12)
    plt.title('Model Performance Comparison', fontweight='bold', fontsize=14)
    
    # Add xticks on the middle of the group bars
    plt.xticks([r + barWidth/2 for r in range(len(model_names))], model_names, rotation=45, ha='right')
    
    # Create legend
    plt.legend()
    
    # Add grid
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Ensure the static directory exists
    os.makedirs('static', exist_ok=True)
    
    # Save the figure
    plt.savefig('static/model_performance.png')
    plt.close()

def plot_feature_importance(feature_names, importances, title='Feature Importance'):
    """
    Plot feature importance.
    
    Parameters:
    -----------
    feature_names : list
        Names of the features
    importances : list
        Importance scores for each feature
    title : str
        Title of the plot
    """
    # Sort features by importance
    indices = np.argsort(importances)
    
    # Select top 20 features if there are more than 20
    if len(indices) > 20:
        indices = indices[-20:]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create horizontal bars
    plt.barh(range(len(indices)), [importances[i] for i in indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    
    # Add labels and title
    plt.xlabel('Importance', fontweight='bold', fontsize=12)
    plt.title(title, fontweight='bold', fontsize=14)
    
    # Add grid
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Ensure the static directory exists
    os.makedirs('static', exist_ok=True)
    
    # Save the figure
    plt.savefig('static/feature_importance.png')
    plt.close()

def plot_keystroke_dynamics(hold_times, flight_times, keys):
    """
    Plot keystroke dynamics visualization.
    
    Parameters:
    -----------
    hold_times : list
        Hold times for each key
    flight_times : list
        Flight times between consecutive keys
    keys : list
        Key names
    """
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Plot hold times
    plt.subplot(2, 1, 1)
    plt.bar(keys, hold_times, color='skyblue')
    plt.xlabel('Keys', fontweight='bold', fontsize=12)
    plt.ylabel('Hold Time (ms)', fontweight='bold', fontsize=12)
    plt.title('Hold Times', fontweight='bold', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot flight times
    plt.subplot(2, 1, 2)
    plt.bar(keys[:-1], flight_times, color='lightcoral')
    plt.xlabel('Key Transitions', fontweight='bold', fontsize=12)
    plt.ylabel('Flight Time (ms)', fontweight='bold', fontsize=12)
    plt.title('Flight Times', fontweight='bold', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Ensure the static directory exists
    os.makedirs('static', exist_ok=True)
    
    # Save the figure
    plt.savefig('static/keystroke_dynamics.png')
    plt.close() 