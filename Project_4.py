import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load and explore the China dataset
china_data = pd.read_csv('dataset/china.csv')
print(f"Number of samples: {china_data.shape[0]}")
print(f"Features: {', '.join(china_data.columns)}")

# 2. Data preprocessing
# Create a copy of the dataset for preprocessing
data = china_data.copy()

# 2.1 Check for missing values
missing_values = data.isnull().sum()
if missing_values.sum() > 0:
    print(f"Missing values found: {missing_values[missing_values > 0]}")
    data = data.dropna()
else:
    print("No missing values found")

# 2.2 Select features and target variable
X = data.drop(['ID', 'Effort'], axis=1)  # Remove ID and target
y = data['Effort']  # The target variable is the actual effort

# 2.3 Check for features with zero variance (if any)
zero_var_features = X.columns[X.var() == 0].tolist()
if zero_var_features:
    print(f"Removing zero variance features: {zero_var_features}")
    X = X.drop(zero_var_features, axis=1)

# 2.4 Log transform the target variable to normalize distribution
y_log = np.log1p(y)  # log(1+x) handles zero values

# 2.5 Feature scaling for linear models
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)  # Convert back to DataFrame

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

# 6. Make predictions and evaluate models
# Get predictions from ML models
predictions = {}
for name, model in models.items():
    predictions[name] = model.predict(X_test)

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
plt.figure(figsize=(15, 10))  # Reduced height since we're removing one plot

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

plt.tight_layout()
plt.savefig('img/china_effort_prediction_results.png')

# 8. Feature importance for the best ML model
best_model = max(results.items(), key=lambda x: x[1]['R²'])[0]
print(f"\nBest performing model based on R²: {best_model}")
print(f"MAE: {results[best_model]['MAE']:.2f}")
print(f"RMSE: {results[best_model]['RMSE']:.2f}")
print(f"R²: {results[best_model]['R²']:.2f}")

