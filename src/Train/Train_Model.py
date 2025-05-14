import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
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
    
    nasa93_path = DATASET_DIR / "nasa93.arff.csv"
    if not nasa93_path.exists():
        print(f"Error: NASA93 dataset not found at {nasa93_path}")
        return False
    
    nasa93 = pd.read_csv(nasa93_path, sep=';')
    print(f"Number of samples: {nasa93.shape[0]}")
    
    rating_map = {'vl': 0.5, 'l': 0.7, 'n': 1.0, 'h': 1.15, 'vh': 1.4, 'xh': 1.65}
    cost_drivers = ['rely', 'data', 'cplx', 'time', 'stor', 'virt', 'turn', 
                    'acap', 'aexp', 'pcap', 'vexp', 'lexp', 'modp', 'tool', 'sced']
    
    data = nasa93.copy()
    for driver in cost_drivers:
        data[driver] = data[driver].map(rating_map)
    
    X = data.drop(['recordnumber', 'projectname', 'cat2', 'forg', 'center', 'year', 'act_effort', 'mode'], axis=1)
    y = data['act_effort']
    
    X['equivphyskloc'] = np.log1p(X['equivphyskloc'])  # Log-transform KLOC
    X['eaf'] = X[cost_drivers].prod(axis=1)  # Effort Adjustment Factor
    
    # X = pd.get_dummies(X, columns=['mode'], prefix='mode')
    
    if X.isnull().sum().sum() > 0:
        X = X.dropna()
        y = y[X.index]
    
    y_log = np.log1p(y)
    
    scaler = StandardScaler()
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
    X_scaled = X.copy()
    X_scaled[numeric_features] = scaler.fit_transform(X[numeric_features])
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_log, test_size=0.2, random_state=42)
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    
    gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    gb_model.fit(X_train, y_train)
    
    gb_pred = gb_model.fit(X_train, y_train)
    
    gb_pred = gb_model.predict(X_test)
    gb_pred_original = np.expm1(gb_pred)
    y_test_original = np.expm1(y_test)
    
    gb_mae = mean_absolute_error(y_test_original, gb_pred_original)
    gb_rmse = np.sqrt(mean_squared_error(y_test_original, gb_pred_original))
    gb_r2 = r2_score(y_test_original, gb_pred_original)
    
    print(f"Gradient Boosting (original scale) - MAE: {gb_mae:.2f}, RMSE: {gb_rmse:.2f}, R²: {gb_r2:.2f}")
    
    joblib.dump(gb_model, MODEL_DIR / "trained_model_loc.pkl")
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
    print(f"Features: {', '.join(ucp_data.columns)}")
    
    # Data preprocessing
    data = ucp_data.copy()
    
    # Handle missing values - following Project_2.py approach
    missing_values = data.isnull().sum()
    print("\nMissing values per column:")
    print(missing_values[missing_values > 0])
    
    # Fill missing values with appropriate method for each column
    for col in data.columns:
        if data[col].dtype == 'object':  # Categorical data
            fill_value = data[col].mode()[0]
        else:  # Numerical data
            fill_value = data[col].median()
        data.fillna({col: fill_value}, inplace=True)
    
    y = data['Real_Effort_Person_Hours']
    columns_to_drop = ['Project_No', 'Real_Effort_Person_Hours', 'Real_P20']
    X = data.drop(columns_to_drop, axis=1)

    # 2.3 One-hot encode categorical variables
    features = [
        'Simple Actors', 'Average Actors', 'Complex Actors',
        'UAW', 'Simple UC', 'Average UC', 'Complex UC', 
        'UUCW', 'TCF', 'ECF'
    ]
    
    # Get categorical columns that need one-hot encoding
    categorical_cols = ['Language', 'Methodology', 'ApplicationType'] 
    
    # Create X (features) and y (target)
    X = data[features + categorical_cols]
    y = data['Real_Effort_Person_Hours']

    # One-hot encode categorical variables
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)
    
    # Print feature names for debugging
    print(f"Features after encoding: {X.columns.tolist()}")
    print(f"Total number of features: {X.shape[1]}")

    # Log transform y
    y_log = np.log1p(y)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # Split data and train model
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_log, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    
    # Train Gradient Boosting model
    print("Training Gradient Boosting model...")
    gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    gb_model.fit(X_train, y_train)
    
    # Evaluate model
    gb_pred = gb_model.predict(X_test)
    gb_mae_log = mean_absolute_error(y_test, gb_pred)
    gb_rmse_log = np.sqrt(mean_squared_error(y_test, gb_pred))
    gb_r2_log = r2_score(y_test, gb_pred)
    
    # Transform back to original scale for interpretable metrics
    gb_pred_original = np.expm1(gb_pred)
    y_test_original = np.expm1(y_test)
    
    gb_mae = mean_absolute_error(y_test_original, gb_pred_original)
    gb_rmse = np.sqrt(mean_squared_error(y_test_original, gb_pred_original))
    gb_r2 = r2_score(y_test_original, gb_pred_original)
    
    print(f"Gradient Boosting (log scale) - MAE: {gb_mae_log:.2f}, RMSE: {gb_rmse_log:.2f}, R²: {gb_r2_log:.2f}")
    print(f"Gradient Boosting (original scale) - MAE: {gb_mae:.2f}, RMSE: {gb_rmse:.2f}, R²: {gb_r2:.2f}")
    
    # Save the model and scaler
    joblib.dump(gb_model, MODEL_DIR / "trained_model_ucp.pkl")
    joblib.dump(scaler, SCALER_DIR / "scaler_ucp.pkl")
    print(f"Saved UCP model and scaler to disk")
    
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, MODEL_DIR / "ucp_feature_names.pkl")

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
        columns_to_drop = ['ID', 'Effort', 'Added', 'Changed', 'Deleted', 
                      'Resource', 'Dev.Type', 'Duration', 'N_effort']
    
    # Only drop columns that exist in the DataFrame
    existing_columns = [col for col in columns_to_drop if col in data.columns]
    if existing_columns:
        print(f"Dropping columns: {existing_columns}")
        X = data.drop(existing_columns, axis=1)
    else:
        print("No columns to drop")
        X = data.copy()
    
    y = data['Effort']   # The target variable is the actual effort
    
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