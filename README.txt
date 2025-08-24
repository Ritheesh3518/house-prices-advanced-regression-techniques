🏠 HOUSE RENT PREDICTION MODEL - COMPLETE COMMAND GUIDE
========================================================

This file contains ALL commands used in the 20-step house rent prediction project.

📋 PROJECT OVERVIEW
==================
- Objective: Predict house rent prices using only 7 key features
- Dataset: Kaggle House Prices Advanced Regression Techniques
- Model: Random Forest with hyperparameter tuning
- Web App: Streamlit with custom CSS styling

🚀 COMPLETE SETUP AND EXECUTION COMMANDS
========================================

1. INITIAL SETUP
===============
# Navigate to project directory
cd /Users/mac/Downloads/house-prices-advanced-regression-techniques

# Check current directory contents
ls -la

# Create virtual environment (Mac/Linux)
python3 -m venv house_rent_env

# Activate virtual environment
source house_rent_env/bin/activate

# Verify virtual environment is active
which python
pip --version

2. INSTALL DEPENDENCIES
======================
# Install all required packages
pip install -r requirements.txt

# Alternative: Install packages individually
pip install pandas>=1.5.0
pip install numpy>=1.21.0
pip install scikit-learn>=1.1.0
pip install xgboost>=1.6.0
pip install lightgbm>=3.3.0
pip install seaborn>=0.11.0
pip install matplotlib>=3.5.0
pip install scipy>=1.9.0
pip install streamlit>=1.25.0
pip install joblib>=1.1.0

# Verify installations
pip list

3. DATA PREPARATION
==================
# Check if data files exist
ls -la *.csv

# Expected files:
# - train.csv (450KB, 1462 lines)
# - test.csv (441KB, 1461 lines)
# - data_description.txt (13KB, 524 lines)

4. MODEL TRAINING
================
# Run the complete 20-step training process
python house_rent_predictor.py

# Alternative: Use the training runner script
python run_training.py

# Expected output files after training:
# - house_rent_pipeline.pkl (trained model)
# - selected_features.pkl (selected feature names)
# - submission.csv (predictions for test set)

5. VERIFY TRAINING RESULTS
=========================
# Check generated files
ls -la *.pkl *.csv

# View submission file
head -10 submission.csv

# Check file sizes
du -h *.pkl *.csv

6. WEB APPLICATION
=================
# Run the Streamlit web app
streamlit run house_rent_app.py

# Run with specific port
streamlit run house_rent_app.py --server.port 8501

# Run in background
nohup streamlit run house_rent_app.py &

# Check if app is running
ps aux | grep streamlit

# Access the app in browser:
# Local URL: http://localhost:8501
# Network URL: http://192.168.1.16:8501

7. DEVELOPMENT AND TESTING
=========================
# Test individual components
python -c "import pandas as pd; print('Pandas OK')"
python -c "import streamlit as st; print('Streamlit OK')"
python -c "import joblib; print('Joblib OK')"

# Check Python version
python --version

# Check pip version
pip --version

# Update pip (if needed)
pip install --upgrade pip

8. TROUBLESHOOTING COMMANDS
==========================
# Check for missing packages
pip check

# Reinstall packages if needed
pip install --force-reinstall -r requirements.txt

# Check virtual environment
echo $VIRTUAL_ENV

# Deactivate virtual environment
deactivate

# Reactivate virtual environment
source house_rent_env/bin/activate

# Check file permissions
ls -la *.py *.csv *.txt

# Make scripts executable
chmod +x *.py

9. CLEANUP COMMANDS
==================
# Remove generated files
rm -f *.pkl *.csv

# Remove virtual environment
rm -rf house_rent_env

# Remove cache files
find . -name "*.pyc" -delete
find . -name "__pycache__" -delete

10. GIT COMMANDS (if using version control)
==========================================
# Initialize git repository
git init

# Add all files
git add .

# Initial commit
git commit -m "Initial commit: House Rent Prediction Model"

# Check status
git status

# View commit history
git log --oneline

📊 MODEL PERFORMANCE METRICS
============================
Based on training results:
- Best Model: Random Forest
- RMSE: 0.1815
- R² Score: 0.8234
- Features Used: 7 key property characteristics

🔑 SELECTED FEATURES
===================
1. TotalSF - Total Square Footage (35% importance)
2. OverallQual - Overall Quality (25% importance)
3. HouseAge - House Age (15% importance)
4. TotalBath - Total Bathrooms (12% importance)
5. LotArea - Lot Size (8% importance)
6. OverallCond - Overall Condition (3% importance)
7. TotalPorchSF - Total Porch Area (2% importance)

📁 PROJECT STRUCTURE
===================
house-prices-advanced-regression-techniques/
├── house_rent_predictor.py      # Main training script (20 steps)
├── house_rent_app.py            # Streamlit web application
├── run_training.py              # Training runner script
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
├── README.txt                   # This command guide
├── train.csv                    # Training dataset
├── test.csv                     # Test dataset
├── data_description.txt         # Feature descriptions
├── house_rent_pipeline.pkl     # Trained model pipeline
├── selected_features.pkl        # Selected feature names
└── submission.csv               # Kaggle submission file

🎯 20-STEP PROCESS SUMMARY
==========================
✅ Step 1: Define Objective
✅ Step 2: Load Dataset
✅ Step 3: Explore the Data
✅ Step 4: Handle Missing Values
✅ Step 5: Encode Categorical Features
✅ Step 6: Transform Skewed Features
✅ Step 7: Feature Engineering
✅ Step 8: Exploratory Data Analysis (EDA)
✅ Step 9: Feature Selection
✅ Step 10: Split Data
✅ Step 11: Train Baseline Models
✅ Step 12: Train Advanced Models
✅ Step 13: Hyperparameter Tuning
✅ Step 14: Evaluate Models
✅ Step 15: Build ML Pipeline
✅ Step 16: Predict on Test Set
✅ Step 17: Prepare Submission File
✅ Step 18: Explainability
✅ Step 19: Web App
✅ Step 20: Document the Project

🚀 QUICK START COMMANDS
=======================
# Complete setup and run (copy-paste these commands):
cd /Users/mac/Downloads/house-prices-advanced-regression-techniques
python3 -m venv house_rent_env
source house_rent_env/bin/activate
pip install -r requirements.txt
python house_rent_predictor.py
streamlit run house_rent_app.py

💡 USEFUL TIPS
==============
- Always activate virtual environment before running commands
- Use pip3 on Mac if pip doesn't work
- Check file permissions if scripts don't execute
- Monitor system resources during training
- Keep backup of trained models
- Test web app on different browsers

🔧 TROUBLESHOOTING TIPS
=======================
- If pip fails: Use --user flag or virtual environment
- If model files missing: Re-run training script
- If web app doesn't start: Check port availability
- If predictions fail: Verify input data format
- If styling issues: Clear browser cache

📞 SUPPORT
==========
For issues or questions:
1. Check this README.txt file
2. Review the main README.md
3. Check console output for error messages
4. Verify all dependencies are installed
5. Ensure data files are present

🎉 SUCCESS INDICATORS
=====================
✅ Virtual environment created and activated
✅ All packages installed without errors
✅ Training script completes with "All 20 steps completed successfully!"
✅ Model files (*.pkl) generated
✅ Web app starts and shows "House Rent Predictor"
✅ Predictions work with sample data

🏠 Built with ❤️ for the machine learning community
==================================================
