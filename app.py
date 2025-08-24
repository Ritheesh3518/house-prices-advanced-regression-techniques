import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.compose import ColumnTransformer

warnings.filterwarnings("ignore", category=UserWarning)

# === Load model ===
pipeline = joblib.load("house_price_pipeline.pkl")

# === Extract expected model input features ===
def get_model_input_features(pipeline):
    for name, step in pipeline.named_steps.items():
        if isinstance(step, ColumnTransformer):
            return step.get_feature_names_out()
        elif hasattr(step, 'transformers_'):
            return step.get_feature_names_out()
    return []

selected_features = list(get_model_input_features(pipeline))

# === Streamlit App UI ===
st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
st.title("🏠 House Price Predictor")
st.markdown("Enter the values for each feature below to predict the **house price**.")

# === Human-readable Labels ===
feature_labels = {
    "MSSubClass": "Building Class",
    "LotFrontage": "Lot Frontage (ft)",
    "LotArea": "Lot Area (sq ft)",
    "OverallQual": "Overall Quality (1-10)",
    "OverallCond": "Overall Condition (1-10)",
    "YearBuilt": "Year Built",
    "YearRemodAdd": "Year Remodeled",
    "MasVnrArea": "Masonry Veneer Area (sq ft)",
    "ExterQual": "Exterior Quality",
    "ExterCond": "Exterior Condition",
    "BsmtQual": "Basement Quality",
    "BsmtFinType1": "Basement Finish Type",
    "BsmtExposure": "Basement Exposure",
    "BsmtFinSF1": "Finished Basement SF",
    "BsmtUnfSF": "Unfinished Basement SF",
    "TotalBsmtSF": "Total Basement SF",
    "HeatingQC": "Heating Quality",
    "CentralAir": "Has Central Air",
    "1stFlrSF": "1st Floor SF",
    "2ndFlrSF": "2nd Floor SF",
    "GrLivArea": "Above Ground Living Area",
    "BedroomAbvGr": "Bedrooms Above Ground",
    "KitchenQual": "Kitchen Quality",
    "TotRmsAbvGrd": "Total Rooms Above Ground",
    "Fireplaces": "Number of Fireplaces",
    "FireplaceQu": "Fireplace Quality",
    "GarageYrBlt": "Garage Year Built",
    "GarageFinish": "Garage Finish",
    "GarageCars": "Garage Capacity (Cars)",
    "GarageArea": "Garage Area (sq ft)",
    "GarageQual": "Garage Quality",
    "PavedDrive": "Has Paved Driveway",
    "WoodDeckSF": "Wood Deck Area (sq ft)",
    "OpenPorchSF": "Open Porch Area (sq ft)",
    "EnclosedPorch": "Enclosed Porch Area (sq ft)",
    "MoSold": "Month Sold",
    "YrSold": "Year Sold",
    "TotalSF": "Total Square Footage",
    "HouseAge": "House Age",
    "RemodAge": "Years Since Remodel",
    "TotalBath": "Total Bathrooms",
    "TotalPorchSF": "Total Porch Area (sq ft)",
    "MSZoning": "Zoning",
    "Neighborhood": "Neighborhood",
    "GarageType": "Garage Type",
    "SaleCondition": "Sale Condition",
    "Functional": "Home Functionality",
    "LotShape": "Lot Shape",
    "LandContour": "Land Contour"
}

# === Default values for input fields ===
default_values = {
    "MSSubClass": 20, "LotFrontage": 70.0, "LotArea": 8450, "OverallQual": 5,
    "OverallCond": 5, "YearBuilt": 1973, "YearRemodAdd": 1994, "MasVnrArea": 0.0,
    "ExterQual": "TA", "ExterCond": "TA", "BsmtQual": "TA", "BsmtFinType1": "GLQ",
    "BsmtExposure": "No", "BsmtFinSF1": 400, "BsmtUnfSF": 500, "TotalBsmtSF": 1000,
    "HeatingQC": "TA", "CentralAir": "Yes", "1stFlrSF": 1200, "2ndFlrSF": 300,
    "GrLivArea": 1500, "BedroomAbvGr": 3, "KitchenQual": "TA", "TotRmsAbvGrd": 6,
    "Fireplaces": 1, "FireplaceQu": "NA", "GarageYrBlt": 2003, "GarageFinish": "RFn",
    "GarageCars": 2, "GarageArea": 480, "GarageQual": "TA", "PavedDrive": "Y",
    "WoodDeckSF": 0, "OpenPorchSF": 60, "EnclosedPorch": 0, "MoSold": 6, "YrSold": 2007,
    "TotalSF": 1800, "HouseAge": 34, "RemodAge": 13, "TotalBath": 2.0,
    "TotalPorchSF": 60, "MSZoning": "RM", "Neighborhood": "IDOTRR",
    "GarageType": "Attchd", "SaleCondition": "Normal", "Functional": "Typ",
    "LotShape": "Reg", "LandContour": "Lvl"
}

# === Input options for categorical fields ===
qual_choices = ["Ex", "Gd", "TA", "Fa", "Po"]
yes_no = ["Yes", "No"]
paved_choices = ["Y", "N"]
garage_types = ["Attchd", "Detchd", "BuiltIn", "Basment", "2Types", "CarPort", "None"]
zoning_types = ["RL", "RM", "FV", "RH", "C (all)"]
neighborhoods = ["NAmes", "CollgCr", "OldTown", "Edwards", "Somerst", "IDOTRR"]
sale_conditions = ["Normal", "Abnorml", "Partial", "AdjLand", "Alloca", "Family"]
garage_finish_choices = ["Fin", "RFn", "Unf", "NA"]
bsmt_fin_choices = ["GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "NA"]
bsmt_exposure_choices = ["Gd", "Av", "Mn", "No", "NA"]
fireplace_qu_choices = ["Ex", "Gd", "TA", "Fa", "Po", "NA"]
functional_choices = ["Typ", "Min1", "Min2", "Mod", "Maj1", "Maj2", "Sev", "Sal"]
lot_shapes = ["Reg", "IR1", "IR2", "IR3"]
land_contours = ["Lvl", "Bnk", "HLS", "Low"]

# === Sidebar Inputs ===
st.sidebar.header("🏗️ Feature Input")
input_data = {}

for feature in default_values:
    label = feature_labels.get(feature, feature)
    default = default_values[feature]

    if feature in ["ExterQual", "ExterCond", "BsmtQual", "KitchenQual", "HeatingQC", "GarageQual"]:
        input_data[feature] = st.sidebar.selectbox(f"{label}", qual_choices, index=qual_choices.index(default))
    elif feature == "CentralAir":
        input_data[feature] = st.sidebar.selectbox(f"{label}", yes_no, index=yes_no.index(default))
    elif feature == "PavedDrive":
        input_data[feature] = st.sidebar.selectbox(f"{label}", paved_choices, index=paved_choices.index(default))
    elif feature == "GarageType":
        input_data[feature] = st.sidebar.selectbox(f"{label}", garage_types, index=garage_types.index(default))
    elif feature == "MSZoning":
        input_data[feature] = st.sidebar.selectbox(f"{label}", zoning_types, index=zoning_types.index(default))
    elif feature == "Neighborhood":
        input_data[feature] = st.sidebar.selectbox(f"{label}", neighborhoods, index=neighborhoods.index(default))
    elif feature == "SaleCondition":
        input_data[feature] = st.sidebar.selectbox(f"{label}", sale_conditions, index=sale_conditions.index(default))
    elif feature == "GarageFinish":
        input_data[feature] = st.sidebar.selectbox(f"{label}", garage_finish_choices, index=garage_finish_choices.index(default))
    elif feature == "BsmtFinType1":
        input_data[feature] = st.sidebar.selectbox(f"{label}", bsmt_fin_choices, index=bsmt_fin_choices.index(default))
    elif feature == "BsmtExposure":
        input_data[feature] = st.sidebar.selectbox(f"{label}", bsmt_exposure_choices, index=bsmt_exposure_choices.index(default))
    elif feature == "FireplaceQu":
        input_data[feature] = st.sidebar.selectbox(f"{label}", fireplace_qu_choices, index=fireplace_qu_choices.index(default))
    elif feature == "Functional":
        input_data[feature] = st.sidebar.selectbox(f"{label}", functional_choices, index=functional_choices.index(default))
    elif feature == "LotShape":
        input_data[feature] = st.sidebar.selectbox(f"{label}", lot_shapes, index=lot_shapes.index(default))
    elif feature == "LandContour":
        input_data[feature] = st.sidebar.selectbox(f"{label}", land_contours, index=land_contours.index(default))
    elif feature in ["OverallQual", "OverallCond"]:
        input_data[feature] = st.sidebar.slider(f"{label}", 1, 10, default)
    elif isinstance(default, int):
        input_data[feature] = st.sidebar.number_input(f"{label}", value=default, step=1)
    else:
        input_data[feature] = st.sidebar.number_input(f"{label}", value=float(default), step=10.0)

# === Predict Button ===
if st.button("🔮 Predict Sale Price"):
    input_df = pd.DataFrame([input_data])

    # One-hot encode input
    input_df_encoded = pd.get_dummies(input_df)

    # Add missing columns
    missing_cols = [col for col in selected_features if col not in input_df_encoded.columns]
    for col in missing_cols:
        input_df_encoded[col] = 0

    # Ensure correct column order
    input_df_encoded = input_df_encoded[selected_features]

    # Predict log sale price and convert
    prediction_log = pipeline.predict(input_df_encoded)[0]
    prediction_price = np.expm1(prediction_log)

    # Display result
    st.success(f"💰 Predicted House Price: **${prediction_price:,.2f}**")
    st.caption("Note: This price is based on a log-transformed model.")

    st.markdown("---")
    st.subheader("📋 Input Summary")
    st.dataframe(input_df.T.rename(columns={0: "Value"}))
