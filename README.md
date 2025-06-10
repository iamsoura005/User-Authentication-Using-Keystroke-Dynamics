# Keystroke Dynamics User Verification

This project implements user verification based on keystroke dynamics using 5 different models:
- Manhattan Distance
- Manhattan Filtered Distance
- Manhattan Scaled Distance
- Gaussian Mixture Model (GMM)
- Support Vector Machine (SVM)

## Dataset
The project uses the CMU Keystroke Benchmark dataset.

## Setup Instructions

1. Clone the repository
```
git clone https://github.com/yourusername/keystroke-dynamics-project.git
cd keystroke-dynamics-project
```

2. Create a virtual environment (optional but recommended)
```
python -m venv venv
```

3. Activate the virtual environment
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

4. Install dependencies
```
pip install -r requirements.txt
```

5. Run the Flask application
```
python app.py
```

6. Open your browser and navigate to `http://localhost:5000`

## Project Structure
```
keystroke-dynamics-project/  
├── data/  
│   ├── keystroke.csv  
│   ├── keystroke.xls  
├── models/  
│   ├── manhattan.py  
│   ├── manhattan_filtered.py  
│   ├── manhattan_scaled.py  
│   ├── gmm.py  
│   ├── svm.py  
├── app.py (Flask app)  
├── static/  
│   ├── style.css  
├── templates/  
│   ├── index.html  
├── requirements.txt  
└── README.md  
```

## Usage
1. On the web interface, you'll be prompted to type a specific phrase
2. After typing, click the "Submit" button
3. The system will analyze your typing pattern and display verification results from all 5 models
4. The final decision is made by majority vote of all models 