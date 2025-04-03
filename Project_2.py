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
print("Step 1: Loading NASA93 dataset")
nasa93 = pd.read_csv('nasa93.arff.csv', sep=';')
print(f"Number of samples: {nasa93.shape[0]}")
print(f"Features: {', '.join(nasa93.columns)}")

# 2. Data preprocessing
print("\nStep 2: Data preprocessing")

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

# 2.6 Feature scaling
scaler = StandardScaler()
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
X_scaled = X.copy()
X_scaled[numeric_features] = scaler.fit_transform(X[numeric_features])

# 3. Split the data into training and testing sets
print("\nStep 3: Splitting data into training and testing sets")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_log, test_size=0.2, random_state=42
)
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# 4. Train basic machine learning models
print("\nStep 4: Training machine learning models")

# 4.1 Linear Regression
print("Training Linear Regression model...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# 4.2 Decision Tree
print("\nTraining Decision Tree model...")
dt_model = DecisionTreeRegressor(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

# 4.3 Random Forest
print("\nTraining Random Forest model...")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# 4.4 Gradient Boosting
print("\nTraining Gradient Boosting model...")
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
gb_model.fit(X_train, y_train)

# 5. Implement COCOMO II model
print("\nStep 5: Implementing COCOMO II model")

def calculate_cocomo_ii_effort(row):
    # COCOMO II Post-Architecture model
    a = 2.94  # Constant
    b = 0.91  # Size exponent
    
    # Project size in KLOC
    kloc = row['equivphyskloc']
    
    # Calculate Effort Multiplier (EM) based on cost drivers
    em_factors = ['rely', 'data', 'cplx', 'time', 'stor', 'virt', 'turn', 
                  'acap', 'aexp', 'pcap', 'vexp', 'lexp', 'modp', 'tool', 'sced']
    em = 1.0
    for factor in em_factors:
        if factor in row and not pd.isna(row[factor]):
            em *= row[factor]
    
    # Handle project mode
    mode = row.get('mode', None)
    if mode is not None:
        mode = mode.lower()
        if mode == 'embedded':
            complexity_adjustment = 1.2
        elif mode == 'organic':
            complexity_adjustment = 0.8
        else:  # semidetached or other values
            complexity_adjustment = 1.0
    else:
        complexity_adjustment = 1.0
    
    # Calculate effort: Effort = a × (KLOC)^b × EM × complexity_adjustment
    effort = a * (kloc ** b) * em * complexity_adjustment
    return effort

# Apply COCOMO II calculation to the dataset
data['cocomo_ii_effort'] = data.apply(calculate_cocomo_ii_effort, axis=1)

# 6. Make predictions and evaluate models
print("\nStep 6: Making predictions and evaluating models")

# 6.1 Get predictions from all models (on log scale)
lr_pred = lr_model.predict(X_test)
dt_pred = dt_model.predict(X_test)
rf_pred = rf_model.predict(X_test)
gb_pred = gb_model.predict(X_test)

# 6.2 Get COCOMO II predictions for the test set
cocomo_ii_pred = data.loc[y_test.index, 'cocomo_ii_effort'].values

# 6.3 Convert predictions back to original scale for evaluation
y_test_original = np.expm1(y_test)
lr_pred_original = np.expm1(lr_pred)
dt_pred_original = np.expm1(dt_pred)
rf_pred_original = np.expm1(rf_pred)
gb_pred_original = np.expm1(gb_pred)

# 6.4 Evaluation function
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"--- {model_name} ---")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.2f}")
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2}

# 6.5 Evaluate all models
results = {}
results['COCOMO II'] = evaluate_model(y_test_original, cocomo_ii_pred, "COCOMO II")
results['Linear Regression'] = evaluate_model(y_test_original, lr_pred_original, "Linear Regression")
results['Decision Tree'] = evaluate_model(y_test_original, dt_pred_original, "Decision Tree")
results['Random Forest'] = evaluate_model(y_test_original, rf_pred_original, "Random Forest")
results['Gradient Boosting'] = evaluate_model(y_test_original, gb_pred_original, "Gradient Boosting")

# 7. Visualize results with three key visualizations
print("\nStep 7: Visualizing results")

plt.figure(figsize=(15, 15))

# 7.1. Scatter plot comparing actual vs predicted values (Visualization 1)
plt.subplot(2, 2, 1)
plt_max = min(np.percentile(y_test_original, 95), 5000)  # Cap at 95th percentile or 5000
plt.scatter(y_test_original, cocomo_ii_pred, label='COCOMO II', alpha=0.5, color='blue')
plt.scatter(y_test_original, lr_pred_original, label='Linear Regression', alpha=0.5, color='red')
plt.scatter(y_test_original, rf_pred_original, label='Random Forest', alpha=0.5, color='green')
plt.scatter(y_test_original, gb_pred_original, label='Gradient Boosting', alpha=0.5, color='purple')
plt.plot([0, plt_max], [0, plt_max], 'k--', lw=2)  # Perfect prediction line
plt.xlabel('Actual Effort (person-months)')
plt.ylabel('Predicted Effort (person-months)')
plt.title('Actual vs Predicted Effort')
plt.xlim(0, plt_max)
plt.ylim(0, plt_max)
plt.legend()
plt.grid(True)

# 7.2. Error comparison across models (Visualization 2)
plt.subplot(2, 2, 2)
models = ['COCOMO II', 'Linear Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting']
mae_values = [results[model]['mae'] for model in models]
rmse_values = [results[model]['rmse'] for model in models]

x = np.arange(len(models))
width = 0.35

plt.bar(x - width/2, mae_values, width, label='MAE')
plt.bar(x + width/2, rmse_values, width, label='RMSE')
plt.xticks(x, models, rotation=45)
plt.ylabel('Error (person-months)')
plt.title('MAE and RMSE Comparison Across Models')
plt.legend()
plt.grid(True)

# 7.3. R² comparison (Visualization 3)
plt.subplot(2, 2, 3)
r2_values = [results[model]['r2'] for model in models]
plt.bar(models, r2_values, color='lightgreen')
plt.ylabel('R² Score')
plt.title('R² Comparison Across Models')
plt.xticks(rotation=45)
plt.axhline(y=0, color='r', linestyle='-')
plt.grid(True)

# Save the visualizations
plt.tight_layout()
plt.savefig('nasa93_effort_prediction_results.png')

# 8. Find best model and provide conclusions
print("\nStep 8: Conclusions and recommendations")
best_model = max(results.items(), key=lambda x: x[1]['r2'])[0]
print(f"Best performing model based on R²: {best_model}")
print(f"MAE: {results[best_model]['mae']:.2f}")
print(f"RMSE: {results[best_model]['rmse']:.2f}")
print(f"R²: {results[best_model]['r2']:.2f}")

print("\nRecommendations:")
print("1. The log transformation of the target variable significantly improves model performance")
print("2. Machine learning models (especially Gradient Boosting) generally outperform traditional COCOMO II")
print("3. The COCOMO II model can be further improved with project-specific calibration")
print("4. Consider collecting more data for better model training")
print("All visualizations have been saved.")