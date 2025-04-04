import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load and explore the NASA93 dataset
nasa93 = pd.read_csv('nasa93.arff.csv', sep=';')
print(f"Number of samples: {nasa93.shape[0]}")
print(f"Features: {', '.join(nasa93.columns)}")

# 2. Data preprocessing
# 2.1 Map categorical values to numerical values
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

# Store original mode for COCOMO II
original_mode = data['mode'].copy()

# 2.2 Select features and target variable
X = data.copy()

# Remove columns that aren't needed for modeling
columns_to_drop = ['recordnumber', 'projectname', 'cat2', 'forg', 'center', 'year', 'act_effort']
X = X.drop(columns_to_drop, axis=1)

# The target variable is the actual effort
y = data['act_effort']

# 2.3 One-hot encode the 'mode' column
X = pd.get_dummies(X, columns=['mode'], prefix='mode')

# 2.4 Check for missing values
missing_values = X.isnull().sum()
if missing_values.sum() > 0:
    X = X.dropna()
    y = y[X.index]
else:
    print("No missing values found")

# 2.5 Log transform the target variable to normalize distribution
y_log = np.log1p(y)  # log(1+x) handles zero values

# 2.6 Feature scaling for linear models only
scaler = StandardScaler()
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
X_scaled = X.copy()
X_scaled[numeric_features] = scaler.fit_transform(X[numeric_features])

# 3. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_log, test_size=0.2, random_state=42
)

# Save original indices for later
test_indices = y_test.index

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# 4. Train machine learning models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
}

# Train all models
for name, model in models.items():
    model.fit(X_train, y_train)

# 5. Implement improved COCOMO II model
def calculate_cocomo_ii_effort(row, original_data):
    # COCOMO II Post-Architecture model
    a = 2.94  # Constant
    b = 0.91  # Size exponent
    
    # Get project size in KLOC (already in KLOC in NASA93)
    kloc = row['equivphyskloc']
    
    # Calculate Effort Multiplier (EM) based on cost drivers
    em_factors = ['rely', 'data', 'cplx', 'time', 'stor', 'virt', 'turn', 
                  'acap', 'aexp', 'pcap', 'vexp', 'lexp', 'modp', 'tool', 'sced']
    em = 1.0
    for factor in em_factors:
        if factor in row and not pd.isna(row[factor]):
            em *= row[factor]
    
    # Get mode directly from original data
    index = row.name
    mode = original_data.loc[index, 'mode'].lower() if 'mode' in original_data.columns else ''
    
    # Handle project mode
    if mode == 'embedded':
        complexity_adjustment = 1.2
    elif mode == 'organic':
        complexity_adjustment = 0.8
    else:  # semidetached or other values
        complexity_adjustment = 1.0
    
    # Calculate effort: Effort = a × (KLOC)^b × EM × complexity_adjustment
    effort = a * (kloc ** b) * em * complexity_adjustment
    
    # Return log-transformed effort to match target scale
    return np.log1p(effort)

# 6. Make predictions and evaluate models
# Get predictions from ML models
predictions = {}
for name, model in models.items():
    predictions[name] = model.predict(X_test)

# Calculate COCOMO II predictions (in log scale to match target)
cocomo_preds = []
for idx in test_indices:
    pred = calculate_cocomo_ii_effort(data.loc[idx], nasa93)
    cocomo_preds.append(pred)
predictions['COCOMO II'] = np.array(cocomo_preds)

# 6.1 Evaluation function
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'RMSE': rmse, 'R²': r2}

# 6.2 Evaluate all models
results = {}
for name, preds in predictions.items():
    results[name] = evaluate_model(y_test, preds)
    print(f"--- {name} ---")
    print(f"MAE: {results[name]['MAE']:.2f}")
    print(f"RMSE: {results[name]['RMSE']:.2f}")
    print(f"R²: {results[name]['R²']:.2f}")

# 6.3 Create a DataFrame with predictions for display
results_df = pd.DataFrame({'Actual': y_test})
for name, preds in predictions.items():
    results_df[name] = preds

print("\nPredictions for first 10 test samples (log scale):")
print(results_df.head(10))

# 7. Visualize results
plt.figure(figsize=(15, 12))

# 7.1. Scatter plot comparing actual vs predicted values
plt.subplot(2, 2, 1)
for name, preds in predictions.items():
    plt.scatter(y_test, preds, label=name, alpha=0.5)
    
plt_max = np.percentile(y_test, 95)  # Cap at 95th percentile
plt.plot([0, plt_max], [0, plt_max], 'k--', lw=2)  # Perfect prediction line
plt.xlabel('Actual Effort (log scale)')
plt.ylabel('Predicted Effort (log scale)')
plt.title('Actual vs Predicted Effort')
plt.xlim(0, plt_max)
plt.ylim(0, plt_max)
plt.legend()
plt.grid(True)

# 7.2. Error comparison across models
plt.subplot(2, 2, 2)
mae_values = [results[model]['MAE'] for model in results]
rmse_values = [results[model]['RMSE'] for model in results]

x = np.arange(len(results))
width = 0.35

plt.bar(x - width/2, mae_values, width, label='MAE')
plt.bar(x + width/2, rmse_values, width, label='RMSE')
plt.xticks(x, results.keys(), rotation=45)
plt.ylabel('Error (log scale)')
plt.title('MAE and RMSE Comparison')
plt.legend()
plt.grid(True)

# 7.3. R² comparison
plt.subplot(2, 2, 3)
r2_values = [results[model]['R²'] for model in results]
plt.bar(list(results.keys()), r2_values, color='lightgreen')
plt.ylabel('R² Score')
plt.title('R² Comparison Across Models')
plt.xticks(rotation=45)
plt.axhline(y=0, color='r', linestyle='-')
plt.grid(True)

# 7.4 Original vs Log scale for actual effort
plt.subplot(2, 2, 4)
plt.hist(data['act_effort'], bins=20, alpha=0.5, label='Original scale')
plt.hist(y_log, bins=20, alpha=0.5, label='Log scale')
plt.xlabel('Effort (person-months)')
plt.ylabel('Frequency')
plt.title('Distribution of Effort: Original vs Log scale')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('nasa93_effort_prediction_results_improved.png')

# 8. Find best model and provide conclusions
best_model = max(results.items(), key=lambda x: x[1]['R²'])[0]
print(f"\nBest performing model based on R²: {best_model}")
print(f"MAE: {results[best_model]['MAE']:.2f}")
print(f"RMSE: {results[best_model]['RMSE']:.2f}")
print(f"R²: {results[best_model]['R²']:.2f}")

