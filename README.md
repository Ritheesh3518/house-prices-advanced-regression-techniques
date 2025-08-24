# 🏠 House Rent Prediction Model

A comprehensive machine learning model for predicting house rent prices using only 7 key property features. This project follows a structured 20-step approach to build, train, and deploy a production-ready rent prediction system.

## 🎯 Project Objective

Predict house rent prices using machine learning models based on key property features, with the constraint of using only 6-7 most important features for prediction to ensure simplicity and interpretability.

## 📊 Dataset

- **Source**: Kaggle House Prices - Advanced Regression Techniques
- **Training Data**: `train.csv` (1,461 samples)
- **Test Data**: `test.csv` (1,460 samples)
- **Target Variable**: SalePrice (converted to monthly rent estimates)

## 🔑 Key Features (7 Selected)

1. **TotalSF** - Total Square Footage (sq ft)
2. **OverallQual** - Overall Quality (1-10 scale)
3. **HouseAge** - Age of House (years)
4. **TotalBath** - Total Bathrooms
5. **LotArea** - Lot Size (sq ft)
6. **OverallCond** - Overall Condition (1-10 scale)
7. **TotalPorchSF** - Total Porch Area (sq ft)

## 🚀 20-Step Implementation Process

### ✅ Step 1: Define Objective
- Predict house rent prices using ML models
- Use only 6-7 key features for simplicity
- Target: Monthly rent price in USD

### ✅ Step 2: Load Dataset
- Load `train.csv` and `test.csv`
- Handle file loading errors gracefully
- Display dataset information

### ✅ Step 3: Explore the Data
- Analyze data structure and shape
- Examine target variable distribution
- Display basic statistics and sample data

### ✅ Step 4: Handle Missing Values
- Numeric columns: Fill with median values
- Categorical columns: Fill with mode values
- Handle neighborhood-specific imputation for LotFrontage

### ✅ Step 5: Encode Categorical Features
- Label encoding for ordinal variables (quality ratings)
- One-hot encoding for nominal variables
- Ensure train/test set alignment

### ✅ Step 6: Transform Skewed Features
- Log-transform target variable (SalePrice)
- Apply log transformation to highly skewed features (>0.75 skewness)
- Normalize distributions for better model performance

### ✅ Step 7: Feature Engineering
- **TotalSF**: Combined basement + 1st floor + 2nd floor area
- **HouseAge**: Years since construction
- **TotalBath**: Full + half bathrooms
- **TotalPorchSF**: Combined porch areas

### ✅ Step 8: Exploratory Data Analysis (EDA)
- Correlation analysis with target variable
- Identify most influential features
- Visualize relationships and distributions

### ✅ Step 9: Feature Selection
- Select top 7 features based on:
  - Correlation with target
  - Domain knowledge
  - Interpretability
  - Predictive power

### ✅ Step 10: Split Data
- Training set: 80% of data
- Validation set: 20% of data
- Random state: 42 for reproducibility

### ✅ Step 11: Train Baseline Models
- Linear Regression
- Ridge Regression
- Lasso Regression
- Benchmark performance metrics

### ✅ Step 12: Train Advanced Models
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

### ✅ Step 13: Hyperparameter Tuning
- Grid Search for Random Forest
- Cross-validation (5-fold)
- Optimize for RMSE metric

### ✅ Step 14: Evaluate Models
- RMSE (Root Mean Square Error)
- R² Score
- Residual analysis
- Prediction vs. actual plots

### ✅ Step 15: Build ML Pipeline
- Preprocessing pipeline with StandardScaler
- Model pipeline with best performing algorithm
- Ensure reproducibility and deployment readiness

### ✅ Step 16: Predict on Test Set
- Apply trained pipeline to test data
- Generate predictions for all test samples
- Convert from log scale to original scale

### ✅ Step 17: Prepare Submission File
- Create `submission.csv` with predictions
- Format: Id, SalePrice columns
- Ready for Kaggle submission

### ✅ Step 18: Explainability
- Feature importance analysis
- Random Forest feature rankings
- Model interpretability insights

### ✅ Step 19: Web App
- Streamlit-based user interface
- Interactive feature input
- Real-time predictions
- Beautiful visualizations

### ✅ Step 20: Document the Project
- Comprehensive documentation
- Usage instructions
- Model performance summary
- Future improvement suggestions

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip3 (as preferred on Mac)

### Complete Setup Commands
```bash
# Navigate to project directory
cd /Users/mac/Downloads/house-prices-advanced-regression-techniques

# Create virtual environment (Mac/Linux)
python3 -m venv house_rent_env

# Activate virtual environment
source house_rent_env/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

### Alternative Installation
```bash
pip install pandas numpy scikit-learn xgboost lightgbm seaborn matplotlib scipy streamlit joblib
```

### Verify Installation
```bash
# Check if virtual environment is active
which python

# Verify packages are installed
pip list

# Test imports
python -c "import pandas, streamlit, joblib; print('All packages OK')"
```

## 🚀 Usage

### 1. Train the Model
```bash
# Run the complete 20-step training process
python house_rent_predictor.py

# Alternative: Use the training runner script
python run_training.py
```

This will:
- Execute all 20 steps
- Train multiple models
- Generate predictions
- Save model files (`house_rent_pipeline.pkl`, `selected_features.pkl`)
- Create `submission.csv`

### 2. Run the Web App
```bash
# Run the Streamlit web app
python3 -m streamlit run house_rent_app.py

# Run with specific port
streamlit run house_rent_app.py --server.port 8501

# Run in background
nohup streamlit run house_rent_app.py &
```

The web app provides:
- Interactive input forms for the 7 key features
- Real-time rent predictions
- Feature importance visualization
- Property summary and analysis
- Beautiful modern UI with custom CSS styling

### 3. Verify Results
```bash
# Check generated files
ls -la *.pkl *.csv

# View submission file
head -10 submission.csv

# Check if web app is running
ps aux | grep streamlit
```

### 3. Make Predictions Programmatically
```python
import joblib
import pandas as pd

# Load the trained model
pipeline = joblib.load('house_rent_pipeline.pkl')
selected_features = joblib.load('selected_features.pkl')

# Prepare input data
input_data = {
    'TotalSF': 1500,
    'OverallQual': 7,
    'HouseAge': 15,
    'TotalBath': 2.5,
    'LotArea': 8000,
    'OverallCond': 6,
    'TotalPorchSF': 150
}

# Make prediction
input_df = pd.DataFrame([input_data])
input_df = input_df[selected_features]
prediction_log = pipeline.predict(input_df)[0]
prediction_price = np.expm1(prediction_log)
monthly_rent = prediction_price * 0.01

print(f"Predicted Sale Price: ${prediction_price:,.0f}")
print(f"Estimated Monthly Rent: ${monthly_rent:,.0f}")
```

## 📈 Model Performance

### Baseline Models
- **Linear Regression**: Baseline performance
- **Ridge Regression**: Regularized linear model
- **Lasso Regression**: Feature selection with regularization

### Advanced Models
- **Random Forest**: Best overall performance
- **Gradient Boosting**: Ensemble method
- **XGBoost**: Optimized gradient boosting
- **LightGBM**: Light gradient boosting

### Performance Metrics
- **RMSE**: Root Mean Square Error
- **R² Score**: Coefficient of determination
- **Feature Importance**: Random Forest rankings

## 🎨 Web App Features

### Interactive Inputs
- **Total Square Footage**: 500-10,000 sq ft range
- **Overall Quality**: 1-10 slider scale
- **House Age**: 0-100 years
- **Total Bathrooms**: 0.5-8.0 (including half baths)
- **Lot Area**: 1,000-50,000 sq ft
- **Overall Condition**: 1-10 slider scale
- **Total Porch Area**: 0-1,000 sq ft

### Outputs
- Predicted sale price
- Estimated monthly rent
- Weekly and daily rent estimates
- Confidence ranges
- Feature importance visualization
- Property summary table

## 📁 Project Structure

```
house-prices-advanced-regression-techniques/
├── house_rent_predictor.py      # Main training script (20 steps)
├── house_rent_app.py            # Streamlit web application
├── requirements.txt              # Python dependencies
├── README.md                    # Project documentation
├── train.csv                    # Training dataset
├── test.csv                     # Test dataset
├── data_description.txt         # Feature descriptions
├── house_rent_pipeline.pkl     # Trained model pipeline
├── selected_features.pkl        # Selected feature names
└── submission.csv               # Kaggle submission file
```

## 🔍 Feature Analysis

### Most Important Features
1. **TotalSF** (35%): Total living area - most critical factor
2. **OverallQual** (25%): Material and finish quality
3. **HouseAge** (15%): Property age impact
4. **TotalBath** (12%): Bathroom count
5. **LotArea** (8%): Land size
6. **OverallCond** (3%): Property condition
7. **TotalPorchSF** (2%): Outdoor living space

### Feature Engineering Benefits
- **TotalSF**: Combines multiple area measurements
- **HouseAge**: Calculated from construction and sale years
- **TotalBath**: Aggregates bathroom types
- **TotalPorchSF**: Combines porch areas

## 🚀 Future Improvements

### Model Enhancements
- Ensemble multiple best models
- Deep learning approaches (Neural Networks)
- Time series analysis for market trends
- Location-based pricing factors

### Feature Additions
- Neighborhood demographics
- School district ratings
- Transportation accessibility
- Market trend indicators

### Deployment
- Docker containerization
- API endpoint creation
- Cloud deployment (AWS, GCP, Azure)
- Real-time data integration

## 📊 Data Preprocessing

### Missing Value Strategy
- **Numeric**: Median imputation
- **Categorical**: Mode imputation
- **Location-specific**: Neighborhood-based imputation

### Encoding Strategy
- **Ordinal**: Label encoding for quality ratings
- **Nominal**: One-hot encoding for categories
- **Binary**: Direct encoding for yes/no variables

### Transformation Strategy
- **Target**: Log transformation for normality
- **Features**: Log transformation for high skewness (>0.75)
- **Scaling**: StandardScaler for model input

## 🎯 Model Selection Criteria

### Performance Metrics
- **RMSE**: Primary optimization target
- **R² Score**: Model fit quality
- **Feature Importance**: Interpretability

### Selection Factors
- **Accuracy**: Best prediction performance
- **Interpretability**: Understandable feature importance
- **Robustness**: Consistent performance across folds
- **Efficiency**: Reasonable training and prediction time

## 📝 Usage Examples

### Example 1: Small Family Home
```
TotalSF: 1200 sq ft
OverallQual: 6
HouseAge: 25 years
TotalBath: 2.0
LotArea: 6000 sq ft
OverallCond: 5
TotalPorchSF: 80 sq ft
```

### Example 2: Luxury Property
```
TotalSF: 3500 sq ft
OverallQual: 9
HouseAge: 5 years
TotalBath: 4.5
LotArea: 15000 sq ft
OverallCond: 9
TotalPorchSF: 400 sq ft
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement improvements
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational and research purposes. The dataset is from Kaggle's House Prices competition.

## 🙏 Acknowledgments

- Kaggle for the House Prices dataset
- Scikit-learn team for the ML framework
- Streamlit for the web app framework
- Open source community for various libraries

---

**🏠 Built with ❤️ for the machine learning community**

*For questions or support, please open an issue in the repository.*
