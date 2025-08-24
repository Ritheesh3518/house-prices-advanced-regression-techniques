import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
import joblib
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge


# === Load Data ===
train_path = "/Users/admin/Desktop/GIT/Riteesh/house-prices-advanced-regression-techniques/train.csv"
test_path = "/Users/admin/Desktop/GIT/Riteesh/house-prices-advanced-regression-techniques/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# === Drop Columns with Too Many Missing Values ===
cols_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence']
train_df.drop(columns=cols_to_drop, inplace=True)
test_df.drop(columns=cols_to_drop, inplace=True)

# === Fill Missing Values ===
cat_fill_none = [
    'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType'
]
for col in cat_fill_none:
    train_df[col] = train_df[col].fillna("None")
    test_df[col] = test_df[col].fillna("None")

num_fill_zero = [
    'GarageYrBlt', 'GarageArea', 'GarageCars', 'MasVnrArea',
    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
    'BsmtFullBath', 'BsmtHalfBath'
]
for col in num_fill_zero:
    train_df[col] = train_df[col].fillna(0)
    test_df[col] = test_df[col].fillna(0)

for df in [train_df, test_df]:
    df['LotFrontage'] = df.groupby("Neighborhood")['LotFrontage'].transform(
        lambda x: x.fillna(x.median())
    )

cat_fill_mode = [
    'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd',
    'SaleType', 'MSZoning', 'Utilities', 'Functional'
]
for col in cat_fill_mode:
    if col in train_df.columns:
        train_df[col] = train_df[col].fillna(train_df[col].mode()[0])
    if col in test_df.columns:
        test_df[col] = test_df[col].fillna(test_df[col].mode()[0])

# === Ordinal Encoding ===
ordinal_mappings = {
    "ExterQual":     ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "ExterCond":     ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtQual":      ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtCond":      ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "HeatingQC":     ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "KitchenQual":   ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "FireplaceQu":   ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "GarageQual":    ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "GarageCond":    ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "PoolQC":        ["None", "Fa", "TA", "Gd", "Ex"],
    "BsmtExposure":  ["None", "No", "Mn", "Av", "Gd"],
    "BsmtFinType1":  ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFinType2":  ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "Functional":    ["None", "Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
    "GarageFinish":  ["None", "Unf", "RFn", "Fin"],
    "PavedDrive":    ["N", "P", "Y"],
    "Street":        ["Grvl", "Pave"],
    "Alley":         ["None", "Grvl", "Pave"],
    "LotShape":      ["IR3", "IR2", "IR1", "Reg"],
    "LandSlope":     ["Sev", "Mod", "Gtl"],
    "Utilities":     ["None", "ELO", "NoSeWa", "NoSewr", "AllPub"],
    "LandContour":   ["Low", "HLS", "Bnk", "Lvl"]
}
for col, order in ordinal_mappings.items():
    if col in train_df.columns:
        order_dict = {cat: i for i, cat in enumerate(order)}
        train_df[col] = train_df[col].map(order_dict)
        test_df[col] = test_df[col].map(order_dict)

# === Label Encoding for Binary Categories ===
binary_cols = [col for col in train_df.columns if train_df[col].dtype == "object"
               and train_df[col].nunique() == 2]
le = LabelEncoder()
for col in binary_cols:
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

# === One-Hot Encoding ===
nominal_cols = [col for col in train_df.columns if train_df[col].dtype == "object"]
train_df = pd.get_dummies(train_df, columns=nominal_cols, drop_first=True)
test_df = pd.get_dummies(test_df, columns=nominal_cols, drop_first=True)

# === Align train and test sets ===
train_df, test_df = train_df.align(test_df, join='left', axis=1, fill_value=0)

# === Log-transform SalePrice ===
train_df["SalePrice"] = np.log1p(train_df["SalePrice"])

# === Convert bool to int ===
bool_cols = train_df.select_dtypes(include='bool').columns
train_df[bool_cols] = train_df[bool_cols].astype(int)
test_df[bool_cols] = test_df[bool_cols].astype(int)

# === Fix Skewness ===
numeric_feats = train_df.select_dtypes(include=[np.number]).columns
numeric_feats = [col for col in numeric_feats if col not in ['Id', 'SalePrice']]
skewed_feats = train_df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
high_skew = skewed_feats[skewed_feats > 0.75]
for feat in high_skew.index:
    train_df[feat] = np.log1p(train_df[feat])
    if feat in test_df.columns:
        test_df[feat] = np.log1p(test_df[feat])

# === Feature Engineering ===
train_new_feats = pd.DataFrame({
    'TotalSF': train_df['TotalBsmtSF'] + train_df['1stFlrSF'] + train_df['2ndFlrSF'],
    'HouseAge': train_df['YrSold'] - train_df['YearBuilt'],
    'RemodAge': train_df['YrSold'] - train_df['YearRemodAdd'],
    'TotalBath': train_df['FullBath'] + 0.5 * train_df['HalfBath'] +
                 train_df['BsmtFullBath'] + 0.5 * train_df['BsmtHalfBath'],
    'TotalPorchSF': train_df['OpenPorchSF'] + train_df['EnclosedPorch'] +
                    train_df['3SsnPorch'] + train_df['ScreenPorch'],
    'IsRemodeled': (train_df['YearBuilt'] != train_df['YearRemodAdd']).astype(int),
    'HasGarage': (train_df['GarageArea'] > 0).astype(int),
    'HasBasement': (train_df['TotalBsmtSF'] > 0).astype(int),
    'Has2ndFlr': (train_df['2ndFlrSF'] > 0).astype(int)
})
test_new_feats = pd.DataFrame({
    'TotalSF': test_df['TotalBsmtSF'] + test_df['1stFlrSF'] + test_df['2ndFlrSF'],
    'HouseAge': test_df['YrSold'] - test_df['YearBuilt'],
    'RemodAge': test_df['YrSold'] - test_df['YearRemodAdd'],
    'TotalBath': test_df['FullBath'] + 0.5 * test_df['HalfBath'] +
                 test_df['BsmtFullBath'] + 0.5 * test_df['BsmtHalfBath'],
    'TotalPorchSF': test_df['OpenPorchSF'] + test_df['EnclosedPorch'] +
                    test_df['3SsnPorch'] + test_df['ScreenPorch'],
    'IsRemodeled': (test_df['YearBuilt'] != test_df['YearRemodAdd']).astype(int),
    'HasGarage': (test_df['GarageArea'] > 0).astype(int),
    'HasBasement': (test_df['TotalBsmtSF'] > 0).astype(int),
    'Has2ndFlr': (test_df['2ndFlrSF'] > 0).astype(int)
})
train_df = pd.concat([train_df, train_new_feats], axis=1)
test_df = pd.concat([test_df, test_new_feats], axis=1)
train_df, test_df = train_df.align(test_df, join='left', axis=1, fill_value=0)
#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=
#=#=#=#=#=#=#=#=#=#=#=#= 📊 Exploratory Data Analysis (EDA) #=#=#=#=#=#=#=#=#=#=#=#=
#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=
# Correlation heatmap
# plt.figure(figsize=(14,10))
# corr = train_df.corr()
# sns.heatmap(corr[['SalePrice']].sort_values(by='SalePrice', ascending=False), annot=True, cmap='coolwarm')
# plt.title("Top Correlations with SalePrice")
# plt.show()

# # Scatterplot of TotalSF vs. SalePrice
# plt.figure(figsize=(8,5))
# sns.scatterplot(x=train_df['TotalSF'], y=train_df['SalePrice'])
# plt.title("TotalSF vs. SalePrice")
# plt.xlabel("TotalSF")
# plt.ylabel("Log SalePrice")
# plt.show()

# # Boxplot of OverallQual vs. SalePrice
# plt.figure(figsize=(8,5))
# sns.boxplot(x=train_df['OverallQual'], y=train_df['SalePrice'])
# plt.title("OverallQual vs. SalePrice")
# plt.xlabel("Overall Quality")
# plt.ylabel("Log SalePrice")
# plt.show()

# === Separate features and target ===
X = train_df.drop(columns=["SalePrice", "Id"], errors='ignore')
y = train_df["SalePrice"]

# === Feature Importance using Random Forest ===
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

# Get feature importances
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances_sorted = importances.sort_values(ascending=False)

# # Plot Top 20 Feature Importances
# plt.figure(figsize=(10, 6))
# importances_sorted[:20].plot(kind='barh')
# plt.title("Top 20 Feature Importances (Random Forest)")
# plt.gca().invert_yaxis()
# plt.xlabel("Importance Score")
# plt.show()

# === Recursive Feature Elimination (RFE) ===
# Use RFE to select top N features
rfe = RFE(estimator=rf, n_features_to_select=50, step=10)
rfe.fit(X, y)

selected_features = X.columns[rfe.support_].tolist()
print(f"Selected Top {len(selected_features)} Features via RFE:\n", selected_features)

# === Update train and test data to keep only selected features ===
X_selected = X[selected_features]
test_selected = test_df[selected_features]

# === Split Data for Validation ===
X_train, X_val, y_train, y_val = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# === Fit Model on Training Data ===
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Predict on Validation Set ===
y_pred = model.predict(X_val)

# === Evaluate with RMSE ===
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Validation RMSE: {rmse:.4f}")


# === Baseline Models ===
print("\n🔹 Baseline Models (Benchmarking):")

baseline_models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=10),
    "Lasso Regression": Lasso(alpha=0.001)
}

for name, model in baseline_models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    print(f"{name} RMSE: {rmse:.4f}")

# === Advanced Ensemble Models ===
print("\n🔹 Advanced Models (Ensemble Methods):")

# XGBoost
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_val)
xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_preds))
print(f"XGBoost RMSE: {xgb_rmse:.4f}")

# LightGBM
lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
lgb_model.fit(X_train, y_train)
lgb_preds = lgb_model.predict(X_val)
lgb_rmse = np.sqrt(mean_squared_error(y_val, lgb_preds))
print(f"LightGBM RMSE: {lgb_rmse:.4f}")

# Random Forest (already trained above)
rf_preds = model.predict(X_val)
rf_rmse = np.sqrt(mean_squared_error(y_val, rf_preds))
print(f"Random Forest RMSE: {rf_rmse:.4f}")

xgb_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

xgb_random = RandomizedSearchCV(
    estimator=xgb.XGBRegressor(random_state=42),
    param_distributions=xgb_params,
    n_iter=20,
    cv=5,
    scoring='neg_root_mean_squared_error',
    verbose=1,
    n_jobs=-1
)

xgb_random.fit(X_train, y_train)
print(f"\n🔍 Best XGBoost Parameters: {xgb_random.best_params_}")
xgb_best = xgb_random.best_estimator_
xgb_rmse_tuned = np.sqrt(mean_squared_error(y_val, xgb_best.predict(X_val)))
print(f"Tuned XGBoost RMSE: {xgb_rmse_tuned:.4f}")

lgb_params = {
    'n_estimators': [100, 200],
    'num_leaves': [31, 40],
    'learning_rate': [0.01, 0.05],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

lgb_grid = GridSearchCV(
    estimator=lgb.LGBMRegressor(random_state=42),
    param_grid=lgb_params,
    cv=5,
    scoring='neg_root_mean_squared_error',
    verbose=1,
    n_jobs=-1
)

lgb_grid.fit(X_train, y_train)
print(f"\n🔍 Best LightGBM Parameters: {lgb_grid.best_params_}")
lgb_best = lgb_grid.best_estimator_
lgb_rmse_tuned = np.sqrt(mean_squared_error(y_val, lgb_best.predict(X_val)))
print(f"Tuned LightGBM RMSE: {lgb_rmse_tuned:.4f}")

# === Evaluate Final Models ===
models = {
    "Random Forest": model,
    "XGBoost (Tuned)": xgb_best,
    "LightGBM (Tuned)": lgb_best
}

for name, model in models.items():
    y_val_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    r2 = r2_score(y_val, y_val_pred)
    print(f"\n📈 {name}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Residual Plot
    residuals = y_val - y_val_pred
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=y_val_pred, y=residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f"{name} - Residual Plot")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.show()

    # Prediction vs. Actual Plot
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=y_val, y=y_val_pred, alpha=0.6)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], '--r')
    plt.title(f"{name} - Actual vs. Predicted")
    plt.xlabel("Actual Log(SalePrice)")
    plt.ylabel("Predicted Log(SalePrice)")
    plt.show()
# === Build ML Pipeline (for reproducibility) ===

# Separate numerical and categorical columns
numerical_cols = X_selected.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = []  # Already encoded earlier, so assumed empty here

# Preprocessing steps
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Combine transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols)
        # Add categorical processing if needed
    ]
)

# Final pipeline with Ridge Regression (can replace with any model)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Ridge(alpha=10))
])

# Fit pipeline
pipeline.fit(X_train, y_train)
y_pred_pipe = pipeline.predict(X_val)

# Evaluate
rmse_pipe = np.sqrt(mean_squared_error(y_val, y_pred_pipe))
r2_pipe = r2_score(y_val, y_pred_pipe)

print("\n🛠️ Ridge Regression (Pipeline)")
print(f"RMSE: {rmse_pipe:.4f}")
print(f"R² Score: {r2_pipe:.4f}")

# === Predict on Test Set ===
test_preds_log = pipeline.predict(test_selected)

# Convert predictions back from log scale to original SalePrice
test_preds = np.expm1(test_preds_log)

# Create submission DataFrame
submission = pd.DataFrame({
    "Id": test_df["Id"],
    "SalePrice": test_preds
})
# Print first 5 rows
print("\n📄 Submission Head:")
print(submission.head())

# Print info
print("\nℹ️ Submission Info:")
print(submission.info())

# Print statistical description
print("\n📊 Submission Description:")
print(submission.describe())
import joblib

# Load the current selected features
selected_features = joblib.load("selected_features.pkl")

# OPTIONAL: View all current features
print("Original selected_features:")
for feat in selected_features:
    print(feat)

# Define logic to remove one-hot encoded columns (they usually contain "_")
filtered_features = [f for f in selected_features if "_" not in f and not f.startswith("GarageType_")]

# Alternatively, you can define a whitelist of known original features:
# whitelist = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', ...]
# filtered_features = [f for f in selected_features if f in whitelist]

print("\nFiltered original features:")
for feat in filtered_features:
    print(feat)

# Save the filtered list back to selected_features.pkl
joblib.dump(filtered_features, "selected_features.pkl")

print("\n✅ Done. Saved cleaned selected_features.pkl with only original features.")
