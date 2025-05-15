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
from pathlib import Path

# Set up paths
BASE_DIR = Path(__file__).parent.parent.parent  # Gets the App_Measurement directory
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "src" / "model"
SCALER_DIR = BASE_DIR / "src" / "scaler"
IMG_DIR = BASE_DIR / "img"


print("Starting model training...")

def train_and_save_models(X_train, X_test, y_train, y_test, prefix, scaler):
    """Train and save all models for a specific measurement type (LOC/UCP/FP)"""
    
    models = {
        'linear': LinearRegression(),
        'decision_tree': DecisionTreeRegressor(max_depth=5, random_state=42),
        'random_forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name.replace('_', ' ').title()}...")
        model.fit(X_train, y_train)
        
        # Make predictions (still in log scale)
        y_pred = model.predict(X_test)
        
        # Calculate metrics in log scale
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Store both log scale and original scale results
        results[name] = {
            'model': model,
            'metrics': {
                'MAE (log)': mae,
                'RMSE (log)': rmse,
                'R²': r2,
                'MAE': np.mean(np.abs(np.expm1(y_test) - np.expm1(y_pred))),
                'RMSE': np.sqrt(np.mean((np.expm1(y_test) - np.expm1(y_pred)) ** 2))
            }
        }
        
        print(f"{name.replace('_', ' ').title()}:")
        print(f"  Log scale - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")
        
        # Save individual model
        model_path = MODEL_DIR / f"{prefix}_{name}_model.pkl"
        joblib.dump(model, model_path)
    
    # Save scaler
    scaler_path = SCALER_DIR / f"{prefix}_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    
    # Create and save comparison table
    comparison_df = pd.DataFrame({
        name: {
            'MAE (log)': metrics['metrics']['MAE (log)'],
            'RMSE (log)': metrics['metrics']['RMSE (log)'],
            'MAE': metrics['metrics']['MAE'],
            'RMSE': metrics['metrics']['RMSE'],
            'R²': metrics['metrics']['R²']
        }
        for name, metrics in results.items()
    }).T
    
    comparison_df.to_csv(IMG_DIR / f"{prefix}_model_comparison.csv")
    print(f"\nModel comparison for {prefix}:")
    print(comparison_df.round(2))
    
    return results
# 1. Train LOC Model (Project_3.py equivalent)
def train_loc_model():
    print("\n======= Training LOC Model (NASA93) =======")
    
    nasa93_path = DATASET_DIR / "nasa93_converted.xlsx"
    if not nasa93_path.exists():
        print(f"Error: NASA93 dataset not found at {nasa93_path}")
        return False
    
    nasa93 = pd.read_excel(nasa93_path)
    print(f"Number of samples: {nasa93.shape[0]}")
    
    cost_drivers = ['rely', 'data', 'cplx', 'time', 'stor', 'virt', 'turn', 
                    'acap', 'aexp', 'pcap', 'vexp', 'lexp', 'modp', 'tool', 'sced']
    
    data = nasa93.copy()
    
    X = data.drop(['recordnumber', 'projectname', 'cat2', 'forg', 'center', 'year', 'act_effort', 'mode'], axis=1)
    y = data['act_effort']
    
    X['equivphyskloc'] = np.log1p(X['equivphyskloc'])  # Log-transform KLOC
    X['eaf'] = X[cost_drivers].prod(axis=1)  # Effort Adjustment Factor
    
    if X.isnull().sum().sum() > 0:
        X = X.dropna()
        y = y[X.index]
    
    y_log = np.log1p(y)
    
    # Scale features
    scaler = StandardScaler()
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
    X_scaled = X.copy()
    X_scaled[numeric_features] = scaler.fit_transform(X[numeric_features])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_log, test_size=0.2, random_state=42
    )
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    
    # Train and save all models
    results = train_and_save_models(X_train, X_test, y_train, y_test, 'loc', scaler)
    
    # Save feature names for later use
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, MODEL_DIR / "loc_feature_names.pkl")
    
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
    
    results = train_and_save_models(X_train, X_test, y_train, y_test, 'ucp', scaler)
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
    
    results = train_and_save_models(X_train, X_test, y_train, y_test, 'fp', scaler)
    
    return True

# Run all training functions
loc_success = train_loc_model()
ucp_success = train_ucp_model()
fp_success = train_fp_model()

if loc_success and ucp_success and fp_success:
    print("\n✅ All models trained and saved successfully!")
    print(f"\nSaved files in {MODEL_DIR}:")
    for file in MODEL_DIR.glob("*.pkl"):
        print(f"- {file.name}")
    
    print(f"\nSaved files in {SCALER_DIR}:")
    for file in SCALER_DIR.glob("*.pkl"):
        print(f"- {file.name}")
    
    print(f"\nComparison tables saved in {IMG_DIR}:")
    for file in IMG_DIR.glob("*_model_comparison.csv"):
        print(f"- {file.name}")
else:
    print("\n❌ Some models failed to train. Please check the error messages above.")
