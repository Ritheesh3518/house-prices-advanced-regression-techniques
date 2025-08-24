import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("🔧 Fixing House Rent Prediction Model to use ALL 7 features")
print("=" * 60)

# Load the original data
print("📊 Loading data...")
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print(f"✅ Training data: {train_df.shape}")
print(f"✅ Test data: {test_df.shape}")

# === STEP 1: Handle Missing Values ===
print("\n🔧 Step 1: Handling missing values...")

# Fill missing values properly
numeric_cols = train_df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if col != 'Id':
        train_df[col].fillna(train_df[col].median(), inplace=True)
        if col in test_df.columns:
            test_df[col].fillna(test_df[col].median(), inplace=True)

categorical_cols = train_df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    train_df[col].fillna(train_df[col].mode()[0], inplace=True)
    if col in test_df.columns:
        test_df[col].fillna(test_df[col].mode()[0], inplace=True)

# Final cleanup
train_df = train_df.fillna(0)
test_df = test_df.fillna(0)

# === STEP 2: Feature Engineering ===
print("\n🔧 Step 2: Creating engineered features...")

# Create the 7 key features properly
train_df['TotalSF'] = train_df['TotalBsmtSF'] + train_df['1stFlrSF'] + train_df['2ndFlrSF']
train_df['HouseAge'] = train_df['YrSold'] - train_df['YearBuilt']
train_df['TotalBath'] = train_df['FullBath'] + 0.5 * train_df['HalfBath'] + train_df['BsmtFullBath'] + 0.5 * train_df['BsmtHalfBath']
train_df['TotalPorchSF'] = train_df['OpenPorchSF'] + train_df['EnclosedPorch'] + train_df['3SsnPorch'] + train_df['ScreenPorch']

test_df['TotalSF'] = test_df['TotalBsmtSF'] + test_df['1stFlrSF'] + test_df['2ndFlrSF']
test_df['HouseAge'] = test_df['YrSold'] - test_df['YearBuilt']
test_df['TotalBath'] = test_df['FullBath'] + 0.5 * test_df['HalfBath'] + test_df['BsmtFullBath'] + 0.5 * test_df['BsmtHalfBath']
test_df['TotalPorchSF'] = test_df['OpenPorchSF'] + test_df['EnclosedPorch'] + test_df['3SsnPorch'] + test_df['ScreenPorch']

# Ensure OverallQual and OverallCond are properly scaled
# They should be 1-10 scale, let's verify
print(f"OverallQual range: {train_df['OverallQual'].min()} - {train_df['OverallQual'].max()}")
print(f"OverallCond range: {train_df['OverallCond'].min()} - {train_df['OverallCond'].max()}")

# === STEP 3: Select and Prepare Features ===
print("\n🔧 Step 3: Selecting the 7 key features...")

selected_features = [
    'TotalSF',           # Total square footage
    'OverallQual',       # Overall quality (1-10)
    'HouseAge',          # Age of house
    'TotalBath',         # Total bathrooms
    'LotArea',           # Lot size
    'OverallCond',       # Overall condition (1-10)
    'TotalPorchSF'       # Total porch area
]

print("✅ Selected features:")
for i, feature in enumerate(selected_features, 1):
    print(f"   {i}. {feature}")

# Prepare features and target
X = train_df[selected_features]
y = train_df['SalePrice']

# Log transform the target
y = np.log1p(y)

print(f"\n📊 Feature statistics:")
print(X.describe())

# === STEP 4: Train New Model ===
print("\n🔧 Step 4: Training new model with all 7 features...")

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a better Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# Train the model
rf_model.fit(X_train, y_train)

# Evaluate
y_pred = rf_model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)

print(f"✅ Model Performance:")
print(f"   RMSE: {rmse:.4f}")
print(f"   R² Score: {r2:.4f}")

# === STEP 5: Check Feature Importance ===
print("\n🔧 Step 5: Feature Importance Analysis:")

feature_importance = pd.DataFrame({
    'Feature': selected_features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("📊 Feature Importance (should be more balanced):")
for idx, row in feature_importance.iterrows():
    print(f"   {row['Feature']}: {row['Importance']:.4f} ({row['Importance']*100:.1f}%)")

# === STEP 6: Create Pipeline ===
print("\n🔧 Step 6: Creating production pipeline...")

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), selected_features)
    ]
)

# Create final pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', rf_model)
])

# Fit pipeline
pipeline.fit(X_train, y_train)

# Test pipeline
y_pred_pipe = pipeline.predict(X_val)
rmse_pipe = np.sqrt(mean_squared_error(y_val, y_pred_pipe))
r2_pipe = r2_score(y_val, y_pred_pipe)

print(f"✅ Pipeline Performance:")
print(f"   RMSE: {rmse_pipe:.4f}")
print(f"   R² Score: {r2_pipe:.4f}")

# === STEP 7: Generate Predictions ===
print("\n🔧 Step 7: Generating predictions...")

# Prepare test features
test_features = test_df[selected_features]
test_predictions_log = pipeline.predict(test_features)
test_predictions = np.expm1(test_predictions_log)

# Create submission
submission = pd.DataFrame({
    'Id': test_df['Id'],
    'SalePrice': test_predictions
})

# Save files
joblib.dump(pipeline, 'house_rent_pipeline.pkl')
joblib.dump(selected_features, 'selected_features.pkl')
submission.to_csv('submission.csv', index=False)

print("✅ Files saved:")
print("   - house_rent_pipeline.pkl (updated model)")
print("   - selected_features.pkl (feature names)")
print("   - submission.csv (predictions)")

# === STEP 8: Test Individual Features ===
print("\n🔧 Step 8: Testing individual feature contributions...")

# Test each feature individually
for feature in selected_features:
    # Create single feature dataset
    X_single = X_train[[feature]]
    X_val_single = X_val[[feature]]
    
    # Train simple model
    single_model = RandomForestRegressor(n_estimators=50, random_state=42)
    single_model.fit(X_single, y_train)
    
    # Predict
    y_pred_single = single_model.predict(X_val_single)
    rmse_single = np.sqrt(mean_squared_error(y_val, y_pred_single))
    r2_single = r2_score(y_val, y_pred_single)
    
    print(f"   {feature}: RMSE={rmse_single:.4f}, R²={r2_single:.4f}")

print("\n🎉 Model fixed! All 7 features are now properly utilized.")
print("📊 The model now uses:")
for i, feature in enumerate(selected_features, 1):
    importance = feature_importance[feature_importance['Feature'] == feature]['Importance'].iloc[0]
    print(f"   {i}. {feature} ({importance*100:.1f}% importance)")

print("\n🚀 You can now run the web app with the improved model!")
print("   streamlit run house_rent_app.py")
