import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.compose import ColumnTransformer

warnings.filterwarnings("ignore", category=UserWarning)

# === Custom CSS Styling ===
def load_css():
    st.markdown("""
    <style>
    /* Main styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    /* Header styling */
    .main-header {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-header h1 {
        color: #2c3e50;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .main-header h3 {
        color: #7f8c8d;
        font-size: 1.2rem;
        font-weight: 400;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
    }
    
    /* Card styling */
    .prediction-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Feature importance styling */
    .importance-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Input styling */
    .stNumberInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 0.5rem;
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Success message styling */
    .success-message {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
    }
    
    /* Info box styling */
    .info-box {
        background: rgba(52, 152, 219, 0.1);
        border-left: 4px solid #3498db;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        color: #7f8c8d;
        margin-top: 3rem;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        .metric-value {
            font-size: 1.5rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# === Page Configuration ===
st.set_page_config(
    page_title="🏠 House Rent Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
load_css()

# === Load Model ===
@st.cache_resource
def load_model():
    try:
        pipeline = joblib.load("house_rent_pipeline.pkl")
        selected_features = joblib.load("selected_features.pkl")
        return pipeline, selected_features
    except FileNotFoundError:
        st.error("❌ Model files not found! Please run the training script first.")
        st.stop()

# === Main App ===
def main():
    # Header with custom styling
    st.markdown("""
    <div class="main-header">
        <h1>🏠 House Rent Predictor</h1>
        <h3>Predict monthly rent prices using only 7 key property features</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    try:
        pipeline, selected_features = load_model()
    except:
        st.error("❌ Failed to load model. Please ensure the training script has been run.")
        return
    
    # === Sidebar Inputs ===
    st.sidebar.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2>🏗️ Property Features</h2>
        <p>Enter the values for each feature below:</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature descriptions and input widgets
    feature_inputs = {}
    
    # 1. Total Square Footage
    total_sf = st.sidebar.number_input(
        "🏠 Total Square Footage (sq ft)",
        min_value=500,
        max_value=10000,
        value=1500,
        step=100,
        help="Total living area including basement, 1st floor, and 2nd floor"
    )
    feature_inputs['TotalSF'] = total_sf
    
    # 2. Overall Quality
    overall_qual = st.sidebar.slider(
        "⭐ Overall Quality",
        min_value=1,
        max_value=10,
        value=5,
        help="Overall material and finish quality (1=Poor, 10=Excellent)"
    )
    feature_inputs['OverallQual'] = overall_qual
    
    # 3. House Age
    house_age = st.sidebar.number_input(
        "📅 House Age (years)",
        min_value=0,
        max_value=100,
        value=20,
        step=1,
        help="Age of the house in years"
    )
    feature_inputs['HouseAge'] = house_age
    
    # 4. Total Bathrooms
    total_bath = st.sidebar.number_input(
        "🚿 Total Bathrooms",
        min_value=0.5,
        max_value=8.0,
        value=2.0,
        step=0.5,
        help="Total bathrooms (full + half bathrooms)"
    )
    feature_inputs['TotalBath'] = total_bath
    
    # 5. Lot Area
    lot_area = st.sidebar.number_input(
        "🌳 Lot Area (sq ft)",
        min_value=1000,
        max_value=50000,
        value=8000,
        step=500,
        help="Lot size in square feet"
    )
    feature_inputs['LotArea'] = lot_area
    
    # 6. Overall Condition
    overall_cond = st.sidebar.slider(
        "🔧 Overall Condition",
        min_value=1,
        max_value=10,
        value=5,
        help="Overall condition (1=Poor, 10=Excellent)"
    )
    feature_inputs['OverallCond'] = overall_cond
    
    # 7. Total Porch Area
    total_porch = st.sidebar.number_input(
        "🏡 Total Porch Area (sq ft)",
        min_value=0,
        max_value=1000,
        value=100,
        step=25,
        help="Total porch area including open, enclosed, and screen porches"
    )
    feature_inputs['TotalPorchSF'] = total_porch
    
    # === Prediction Button ===
    st.sidebar.markdown("---")
    predict_button = st.sidebar.button("🔮 Predict Rent Price", type="primary")
    
    # === Main Content Area ===
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="prediction-card">
            <h2>📊 Property Summary</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Display input summary
        summary_df = pd.DataFrame({
            'Feature': list(feature_inputs.keys()),
            'Value': list(feature_inputs.values())
        })
        
        # Add descriptions
        descriptions = {
            'TotalSF': 'Total Square Footage (sq ft)',
            'OverallQual': 'Overall Quality (1-10)',
            'HouseAge': 'House Age (years)',
            'TotalBath': 'Total Bathrooms',
            'LotArea': 'Lot Area (sq ft)',
            'OverallCond': 'Overall Condition (1-10)',
            'TotalPorchSF': 'Total Porch Area (sq ft)'
        }
        
        summary_df['Description'] = summary_df['Feature'].map(descriptions)
        summary_df = summary_df[['Feature', 'Description', 'Value']]
        
        st.dataframe(summary_df, use_container_width=True)
        
        # Feature importance visualization
        if predict_button:
            st.markdown("""
            <div class="importance-card">
                <h3>🎯 Feature Importance</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Create a simple bar chart of feature importance (updated from fixed model)
            importance_data = {
                'TotalSF': 0.423,
                'OverallQual': 0.413,
                'HouseAge': 0.057,
                'TotalBath': 0.030,
                'LotArea': 0.038,
                'OverallCond': 0.023,
                'TotalPorchSF': 0.016
            }
            
            importance_df = pd.DataFrame({
                'Feature': list(importance_data.keys()),
                'Importance': list(importance_data.values())
            }).sort_values('Importance', ascending=True)
            
            st.bar_chart(importance_df.set_index('Feature'))
    
    with col2:
        st.markdown("""
        <div class="prediction-card">
            <h2>💰 Rent Prediction</h2>
        </div>
        """, unsafe_allow_html=True)
        
        if predict_button:
            try:
                # Prepare input data
                input_df = pd.DataFrame([feature_inputs])
                
                # Ensure correct column order
                input_df = input_df[selected_features]
                
                # Make prediction
                prediction_log = pipeline.predict(input_df)[0]
                prediction_price = np.expm1(prediction_log)
                
                # Convert to monthly rent (assuming 1% of sale price as monthly rent)
                monthly_rent = prediction_price * 0.01
                
                # Display results with custom styling
                st.markdown(f"""
                <div class="success-message">
                    🏠 <strong>Predicted Sale Price:</strong> ${prediction_price:,.0f}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">${monthly_rent:,.0f}</div>
                    <div class="metric-label">Estimated Monthly Rent</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Additional metrics
                st.markdown("""
                <div style="margin: 1rem 0;">
                    <h4>📈 Additional Estimates:</h4>
                </div>
                """, unsafe_allow_html=True)
                
                weekly_rent = monthly_rent / 4.33
                daily_rent = monthly_rent / 30
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">${weekly_rent:,.0f}</div>
                        <div class="metric-label">Weekly Rent</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_b:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">${daily_rent:,.0f}</div>
                        <div class="metric-label">Daily Rent</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Confidence interval
                st.markdown("""
                <div style="margin: 1rem 0;">
                    <h4>🎯 Confidence Range:</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Simple confidence calculation based on model performance
                confidence_range = monthly_rent * 0.15  # ±15%
                st.markdown(f"""
                <div class="info-box">
                    <strong>Rent Range:</strong> ${monthly_rent - confidence_range:,.0f} - ${monthly_rent + confidence_range:,.0f}
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.info("Please ensure all features have valid values.")
        else:
            st.markdown("""
            <div class="info-box">
                👆 Click 'Predict Rent Price' to get started!
            </div>
            """, unsafe_allow_html=True)
    
    # === Model Information ===
    st.markdown("---")
    st.markdown("""
    <div class="prediction-card">
        <h2>ℹ️ Model Information</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown("""
        <div class="info-box">
            <h4>🔬 Model Details:</h4>
            <ul>
                <li><strong>Algorithm:</strong> Random Forest Regressor</li>
                <li><strong>Features:</strong> 7 key property characteristics</li>
                <li><strong>Training:</strong> Hyperparameter-tuned with cross-validation</li>
                <li><strong>Performance:</strong> Optimized for accuracy and interpretability</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col_info2:
        st.markdown("""
        <div class="info-box">
            <h4>📊 Feature Selection:</h4>
            <ul>
                <li>Selected based on correlation analysis</li>
                <li>Domain knowledge consideration</li>
                <li>Reduced complexity for better interpretability</li>
                <li>Focused on most impactful factors</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # === Usage Instructions ===
    st.markdown("---")
    st.markdown("""
    <div class="prediction-card">
        <h2>📖 How to Use</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <ol>
            <li><strong>Input Property Details:</strong> Use the sidebar to enter property characteristics</li>
            <li><strong>Click Predict:</strong> Press the 'Predict Rent Price' button</li>
            <li><strong>View Results:</strong> See predicted sale price and estimated monthly rent</li>
            <li><strong>Understand Features:</strong> Review feature importance and property summary</li>
        </ol>
        
        <h4>💡 Tips for Accurate Predictions:</h4>
        <ul>
            <li>Ensure all measurements are accurate</li>
            <li>Quality ratings should reflect actual property condition</li>
            <li>Age should be the actual age of the property</li>
            <li>Bathroom count includes both full and half bathrooms</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # === Footer ===
    st.markdown("""
    <div class="footer">
        🏠 House Rent Predictor | Built with Streamlit & Scikit-learn | 
        Based on Kaggle House Prices Dataset
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
