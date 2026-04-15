#!/usr/bin/env python3
"""
FinTrack - Personal Finance Tracker
Run this file to start the web server.
"""
import subprocess, sys, os

# Auto-install dependencies if needed
try:
    import flask, sklearn, pandas, numpy
except ImportError:
    print("Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install",
        "flask", "scikit-learn", "pandas", "numpy", "--break-system-packages"])

os.chdir(os.path.dirname(os.path.abspath(__file__)))
from app import app
print("=" * 50)
print("  FinTrack - Personal Finance Tracker")
print("  Open your browser at: http://localhost:5050")
print("=" * 50)
app.run(debug=False, port=5050, host='0.0.0.0')
