#!/usr/bin/env python3
"""
Simple script to run the house rent prediction training process.
This script will execute all 20 steps and save the trained model.
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'xgboost', 
        'lightgbm', 'seaborn', 'matplotlib', 'scipy', 
        'streamlit', 'joblib'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip3 install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_data_files():
    """Check if required data files exist."""
    required_files = ['train.csv', 'test.csv']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing data files: {', '.join(missing_files)}")
        print("Please ensure train.csv and test.csv are in the current directory")
        return False
    
    print("✅ All required data files found")
    return True

def run_training():
    """Run the main training script."""
    print("\n🚀 Starting House Rent Prediction Model Training...")
    print("=" * 60)
    
    try:
        # Run the training script
        result = subprocess.run([
            sys.executable, 'house_rent_predictor.py'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("\n✅ Training completed successfully!")
            print("\n📁 Generated files:")
            
            # Check for generated files
            generated_files = [
                'house_rent_pipeline.pkl',
                'selected_features.pkl',
                'submission.csv'
            ]
            
            for file in generated_files:
                if os.path.exists(file):
                    print(f"   ✅ {file}")
                else:
                    print(f"   ❌ {file} (not found)")
            
            print("\n🎉 Your house rent prediction model is ready!")
            print("\n📱 To run the web app:")
            print("   streamlit run house_rent_app.py")
            
        else:
            print(f"❌ Training failed with error code: {result.returncode}")
            print(f"Error output: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error running training script: {str(e)}")
        return False
    
    return True

def main():
    """Main function to orchestrate the training process."""
    print("🏠 House Rent Prediction Model - Training Runner")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check data files
    if not check_data_files():
        return
    
    # Run training
    if run_training():
        print("\n🎊 All done! Your model is ready for use.")
    else:
        print("\n💥 Training failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
