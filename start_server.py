import os
import sys

# Try to generate the visualizations first
try:
    from generate_visualizations import main as generate_viz
    generate_viz()
    print("Visualizations generated successfully")
except Exception as e:
    print(f"Warning: Could not generate visualizations: {e}")

# Start the Flask app
from app import app
print("Starting Flask server on http://localhost:5000")
app.run(host='0.0.0.0', port=5000, debug=True) 