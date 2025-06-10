import pandas as pd
import numpy as np
import os

def generate_sample_data(output_file='data/keystroke.csv', num_subjects=5, num_sessions=4):
    """
    Generate sample keystroke data for demonstration purposes.
    
    Parameters:
    -----------
    output_file : str
        Path to save the CSV file
    num_subjects : int
        Number of subjects to generate
    num_sessions : int
        Number of sessions per subject
    """
    # The phrase to type
    phrase = "sourasantadutta"
    keys = list(phrase)
    
    # Create empty lists to store data
    data = []
    
    # Generate data for each subject
    for subject_id in range(1, num_subjects + 1):
        # Each subject has their own typing pattern
        # Genuine user (subject_id = 1) has consistent typing pattern
        # Impostors have different patterns
        
        # Base hold time and flight time for this subject
        base_hold_time = 80 + 20 * subject_id  # milliseconds
        base_flight_time = 150 + 30 * subject_id  # milliseconds
        
        # Variance in typing pattern
        # Genuine user has lower variance
        hold_variance = 10 if subject_id == 1 else 30
        flight_variance = 20 if subject_id == 1 else 50
        
        # Generate data for each session
        for session_id in range(1, num_sessions + 1):
            # Current timestamp
            timestamp = 0
            
            # Type each key in the phrase
            for key in keys:
                # Key press time
                press_time = timestamp
                
                # Hold time for this key
                hold_time = base_hold_time + np.random.normal(0, hold_variance)
                
                # Key release time
                release_time = press_time + hold_time
                
                # Add to data
                data.append({
                    'subject_id': subject_id,
                    'session_id': session_id,
                    'key': key,
                    'press_time': press_time,
                    'release_time': release_time
                })
                
                # Flight time to next key
                flight_time = base_flight_time + np.random.normal(0, flight_variance)
                
                # Update timestamp for next key
                timestamp = release_time + flight_time
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    
    # Also save as Excel (optional - only if openpyxl is available)
    try:
        if output_file.endswith('.csv'):
            excel_file = output_file.replace('.csv', '.xls')
            df.to_excel(excel_file, index=False)
    except Exception as e:
        print(f"Warning: Could not save Excel file: {e}")
        print("Continuing with CSV file only.")
    
    print(f"Sample data generated and saved to {output_file}")
    return df

if __name__ == "__main__":
    generate_sample_data() 