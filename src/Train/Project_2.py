import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import joblib

# 1. Load and explore the dataset
ucp_data = pd.read_csv('././dataset/UCP_Dataset.csv')
print(f"Number of samples: {ucp_data.shape[0]}")
print(f"Features: {', '.join(ucp_data.columns)}")

# 2. Data preprocessing
data = ucp_data.copy()

# 2.1 Handle missing values
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

# 2.2 Select features and target variable
y = data['Real_Effort_Person_Hours']
columns_to_drop = ['Project_No', 'Real_Effort_Person_Hours', 'Real_P20']
X = data.drop(columns_to_drop, axis=1)

# 2.3 One-hot encode categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# 2.4 Log transform the target variable
y_log = np.log1p(y)

# 2.5 Feature scaling
scaler = StandardScaler()
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
X_scaled = X.copy()
X_scaled[numeric_features] = scaler.fit_transform(X[numeric_features])

# 3. Use cross-validation to assess model performance
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
}

# 4. Evaluate models with 5-fold cross-validation
cv_scores = {}
for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y_log, cv=5, scoring='r2')
    cv_scores[name] = scores.mean()
    print(f"{name} CV R²: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")

# 5. Train and predict on a single split for visualization
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_log, test_size=0.2, random_state=42)
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

predictions = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions[name] = model.predict(X_test)

# 6. Evaluation function
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'RMSE': rmse, 'R²': r2}

# 6.1 Evaluate all models on the test set
results = {}
for name, preds in predictions.items():
    results[name] = evaluate_model(y_test, preds)
    print(f"--- {name} ---")
    print(f"MAE: {results[name]['MAE']:.2f}")
    print(f"RMSE: {results[name]['RMSE']:.2f}")
    print(f"R²: {results[name]['R²']:.2f}")

# 6.2 Create a DataFrame with predictions for display
results_df = pd.DataFrame({'Actual': y_test})
for name, preds in predictions.items():
    results_df[name] = preds
print("\nPredictions for first 10 test samples (log scale):")
print(results_df.head(10))

# 7. Visualize results with improved readability
plt.figure(figsize=(18, 15))

# 7.1. Scatter plot comparing actual vs predicted values
plt.subplot(2, 2, 1)
for name, preds in predictions.items():
    plt.scatter(y_test, preds, label=name, alpha=0.6, s=100)  # Increased marker size and transparency
plt.plot([0, np.max(y_test)], [0, np.max(y_test)], 'k--', lw=2)  # Dynamic max limit
plt.xlabel('Actual Effort (log scale)')
plt.ylabel('Predicted Effort (log scale)')
plt.title('Actual vs Predicted Effort')
plt.xlim(0, np.max(y_test) * 1.1)
plt.ylim(0, np.max(y_test) * 1.1)
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
plt.savefig('././img/UCP_effort_prediction.png')

# 8. Find best model based on test set R²
best_model = max(results.items(), key=lambda x: x[1]['R²'])[0]
print(f"\nBest performing model based on R²: {best_model}")
print(f"MAE: {results[best_model]['MAE']:.2f}")
print(f"RMSE: {results[best_model]['RMSE']:.2f}")
print(f"R²: {results[best_model]['R²']:.2f}")


best_model_object = models[best_model]  # Lấy object model tốt nhất
joblib.dump(best_model_object, './src/model/trained_model_ucp.pkl')
joblib.dump(scaler, './src/scaler/scaler_ucp.pkl') 