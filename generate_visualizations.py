#!/usr/bin/env python
"""
Script to generate visualizations for the Keystroke Dynamics project.
This will create the model performance visualization.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

def plot_model_performance():
    """Generate model performance visualization with sample data"""
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Sample data
    model_names = ['Manhattan Distance', 'Manhattan Filtered', 'Manhattan Scaled', 'GMM', 'SVM']
    accuracies = [0.85, 0.87, 0.89, 0.82, 0.91]
    eers = [0.12, 0.10, 0.09, 0.15, 0.08]
    
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
    print("Model performance visualization saved to static/model_performance.png")
    plt.close()

def main():
    """Main function to generate all visualizations"""
    plot_model_performance()
    
    # Create a simple feature importance visualization as well
    plt.figure(figsize=(12, 8))
    features = ['Hold_T', 'Hold_h', 'Hold_e', 'Flight_T-h', 'Flight_h-e']
    importances = [0.8, 0.6, 0.7, 0.9, 0.5]
    
    plt.barh(features, importances, color='skyblue')
    plt.xlabel('Importance', fontweight='bold', fontsize=12)
    plt.title('Feature Importance (Sample)', fontweight='bold', fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig('static/feature_importance.png')
    print("Feature importance visualization saved to static/feature_importance.png")
    plt.close()

if __name__ == "__main__":
    main() 