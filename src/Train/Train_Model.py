import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from pathlib import Path

# Set up paths
BASE_DIR = Path(__file__).parent.parent.parent  # Gets the App_Measurement directory
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "src" / "model"
SCALER_DIR = BASE_DIR / "src" / "scaler"
IMG_DIR = BASE_DIR / "img"


print("Starting model training...")

# 1. Train LOC Model (Project_3.py equivalent)
def train_loc_model():
    print("\n======= Training LOC Model (NASA93) =======")
    
    # Load the NASA93 dataset
    nasa93_path = DATASET_DIR / "nasa93.arff.csv"
    if not nasa93_path.exists():
        print(f"Error: NASA93 dataset not found at {nasa93_path}")
        return False
    
    nasa93 = pd.read_csv(nasa93_path, sep=';')
    print(f"Number of samples: {nasa93.shape[0]}")
    
    # Data preprocessing
    rating_map = {
        'vl': 0.5,   # Very Low
        'l': 0.7,    # Low
        'n': 1.0,    # Nominal
        'h': 1.15,   # High
        'vh': 1.4,   # Very High
        'xh': 1.65   # Extra High
    }
    
    # List of cost driver columns that need mapping
    cost_drivers = ['rely', 'data', 'cplx', 'time', 'stor', 'virt', 'turn', 
                    'acap', 'aexp', 'pcap', 'vexp', 'lexp', 'modp', 'tool', 'sced']
    
    # Create a copy of the dataset for preprocessing
    data = nasa93.copy()
    
    # Apply mapping to cost drivers
    for driver in cost_drivers:
        if driver in data.columns:
            data[driver] = data[driver].map(rating_map)
    
    # Store original mode for reference
    original_mode = data['mode'].copy()
    
    # Select features and target variable
    X = data.copy()
    
    # Remove columns that aren't needed for modeling
    columns_to_drop = ['recordnumber', 'projectname', 'cat2', 'forg', 'center', 'year', 'act_effort']
    X = X.drop(columns_to_drop, axis=1)
    
    # The target variable is the actual effort
    y = data['act_effort']
    
    # One-hot encode the 'mode' column
    X = pd.get_dummies(X, columns=['mode'], prefix='mode')
    
    # Check for missing values
    missing_values = X.isnull().sum()
    if missing_values.sum() > 0:
        X = X.dropna()
        y = y[X.index]
    else:
        print("No missing values found")
    
    # Log transform the target variable to normalize distribution
    y_log = np.log1p(y)  # log(1+x) handles zero values
    
    # Feature scaling for linear models only
    scaler = StandardScaler()
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
    X_scaled = X.copy()
    X_scaled[numeric_features] = scaler.fit_transform(X[numeric_features])
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_log, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    
    # Train RandomForest model
    print("Training Random Forest model...")
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Evaluate model
    rf_pred = rf_model.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_r2 = r2_score(y_test, rf_pred)
    
    print(f"Random Forest - MAE: {rf_mae:.2f}, RMSE: {rf_rmse:.2f}, R²: {rf_r2:.2f}")
    
    # Save the model and scaler
    joblib.dump(rf_model, MODEL_DIR / "trained_model_loc.pkl")
    joblib.dump(scaler, SCALER_DIR / "scaler_loc.pkl")
    print(f"Saved LOC model and scaler to disk")
    
    return True

# 2. Train UCP Model (Project_2.py equivalent)
def train_ucp_model():
    print("\n======= Training UCP Model =======")
    
    # Load the UCP dataset
    ucp_path = DATASET_DIR / "UCP_Dataset.csv"
    if not ucp_path.exists():
        print(f"Error: UCP dataset not found at {ucp_path}")
        return False
    
    ucp_data = pd.read_csv(ucp_path)
    print(f"Number of samples: {ucp_data.shape[0]}")
    
    # Data preprocessing
    data = ucp_data.copy()
    
    # Handle missing values
    for col in data.columns:
        if data[col].dtype == 'object':  # Categorical data
            fill_value = data[col].mode()[0]
        else:  # Numerical data
            fill_value = data[col].median()
        data.fillna({col: fill_value}, inplace=True)
    
    # Select features and target variable
    y = data['Real_Effort_Person_Hours']
    columns_to_drop = ['Project_No', 'Real_Effort_Person_Hours', 'Real_P20']
    X = data.drop(columns_to_drop, axis=1)
    
    # One-hot encode categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Log transform the target variable
    y_log = np.log1p(y)
    
    # Feature scaling
    scaler = StandardScaler()
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
    X_scaled = X.copy()
    X_scaled[numeric_features] = scaler.fit_transform(X[numeric_features])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_log, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    
    # Train RandomForest model
    print("Training Random Forest model...")
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Evaluate model
    rf_pred = rf_model.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_r2 = r2_score(y_test, rf_pred)
    
    print(f"Random Forest - MAE: {rf_mae:.2f}, RMSE: {rf_rmse:.2f}, R²: {rf_r2:.2f}")
    
    # Save the model and scaler
    joblib.dump(rf_model, MODEL_DIR / "trained_model_ucp.pkl")
    joblib.dump(scaler, SCALER_DIR / "scaler_ucp.pkl")
    print(f"Saved UCP model and scaler to disk")
    
    return True

# 3. Train FP Model (Project_4.py equivalent)
def train_fp_model():
    print("\n======= Training FP Model (China) =======")
    
    # Load the China dataset
    china_path = DATASET_DIR / "china.csv"
    if not china_path.exists():
        print(f"Error: China dataset not found at {china_path}")
        return False
    
    china_data = pd.read_csv(china_path)
    print(f"Number of samples: {china_data.shape[0]}")
    
    # Data preprocessing
    data = china_data.copy()
    
    # Check for missing values
    missing_values = data.isnull().sum()
    if missing_values.sum() > 0:
        print(f"Missing values found: {missing_values[missing_values > 0]}")
        data = data.dropna()
    else:
        print("No missing values found")
    
    # Select features and target variable
    X = data.drop(['ID', 'Effort', 'Added', 'Changed', 'Deleted', 'Resource', 'Dev.Type', 'Duration', 'N_effort'], axis=1)  # Remove ID and target
    y = data['Effort']  # The target variable is the actual effort
    
    # Check for features with zero variance (if any)
    zero_var_features = X.columns[X.var() == 0].tolist()
    if zero_var_features:
        print(f"Removing zero variance features: {zero_var_features}")
        X = X.drop(zero_var_features, axis=1)
    
    # Log transform the target variable to normalize distribution
    y_log = np.log1p(y)  # log(1+x) handles zero values
    
    # Feature scaling for linear models
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)  # Convert back to DataFrame
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_log, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    
    # Train RandomForest model
    print("Training Random Forest model...")
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Evaluate model
    rf_pred = rf_model.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_r2 = r2_score(y_test, rf_pred)
    
    print(f"Random Forest - MAE: {rf_mae:.2f}, RMSE: {rf_rmse:.2f}, R²: {rf_r2:.2f}")
    
    # Save the model and scaler
    joblib.dump(rf_model, MODEL_DIR / "trained_model_fp.pkl")
    joblib.dump(scaler, SCALER_DIR / "scaler_fp.pkl")
    print(f"Saved FP model and scaler to disk")
    
    return True

# Run all training functions
loc_success = train_loc_model()
ucp_success = train_ucp_model()
fp_success = train_fp_model()

if loc_success and ucp_success and fp_success:
    print("\n✅ All models trained and saved successfully!")
    print(f"Models saved to: {MODEL_DIR}")
    print(f"Scalers saved to: {SCALER_DIR}")
else:
    print("\n❌ Some models failed to train. Please check the error messages above.")

# Show saved files
print("\nSaved files:")
for dir_path, file_pattern in [(MODEL_DIR, "trained_model_*.pkl"), (SCALER_DIR, "scaler_*.pkl")]:
    for file in dir_path.glob(file_pattern):
        print(f"- {file}")