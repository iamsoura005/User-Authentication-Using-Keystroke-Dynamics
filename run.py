#!/usr/bin/env python
"""
Run script for Keystroke Dynamics User Verification project.
This script checks if sample data exists, generates it if needed, and starts the Flask app.
"""

import os
import sys

def main():
    """Main entry point for the application."""
    # Check if data directory exists
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Check if sample data exists
    data_path = os.path.join('data', 'keystroke.csv')
    if not os.path.exists(data_path):
        print("Sample data not found. Generating...")
        try:
            from data.sample_data import generate_sample_data
            generate_sample_data()
        except Exception as e:
            print(f"Error generating sample data: {e}")
            sys.exit(1)
    
    # Start the Flask app
    from app import app
    app.run(debug=True)

if __name__ == "__main__":
    main() 