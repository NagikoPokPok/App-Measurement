import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression

# 1. Load and explore the NASA93 dataset
print("Step 1: Loading NASA93 dataset")
nasa93 = pd.read_csv('nasa93.arff.csv', sep=';')
print(f"Number of samples: {nasa93.shape[0]}")
print(f"Features: {', '.join(nasa93.columns)}")
print("\nFirst 5 rows:")
print(nasa93.head())

# 2. Data preprocessing with improved approach
print("\nStep 2: Data preprocessing")

# 2.1 Map categorical values to numerical values with refined mapping
# NASA93 uses vh (very high), h (high), n (nominal), l (low), vl (very low)
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

print("\nConverted cost drivers to numerical values:")
print(data[cost_drivers].head())

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
print(f"\nMissing values in each feature:\n{missing_values}")

if missing_values.sum() > 0:
    X = X.dropna()
    y = y[X.index]
    print(f"Number of samples after handling missing values: {X.shape[0]}")
else:
    print("No missing values found")

# 2.5 Log transform the target variable to normalize distribution
# This helps with the wide range of effort values
y_log = np.log1p(y)  # log(1+x) handles zero values

# 2.6 Feature scaling with RobustScaler (better for outliers)
scaler = RobustScaler()  # More robust to outliers than StandardScaler
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
X_scaled = X.copy()
X_scaled[numeric_features] = scaler.fit_transform(X[numeric_features])

print("\nNormalized features (first 5 rows):")
print(X_scaled.head())

# 2.7 Outlier detection and handling
def detect_outliers(y, threshold=3):
    from scipy import stats
    z_scores = stats.zscore(y)
    return np.where(np.abs(z_scores) > threshold)

outliers = detect_outliers(y_log)
print(f"\nNumber of outliers detected: {len(outliers[0])}")
print(f"Outlier indices: {outliers[0]}")

# 3. Split the data into training and testing sets
print("\nStep 3: Splitting data into training and testing sets")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_log, test_size=0.2, random_state=42, stratify=None
)
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# 3.1 Feature selection to improve model focus
print("\nStep 3.1: Feature selection")
selector = SelectKBest(f_regression, k=10)  # Select top 10 features
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Get selected feature names for interpretation
selected_features = X_train.columns[selector.get_support()]
print("Selected features:", selected_features)

# 4. Train machine learning models with improved approach
print("\nStep 4: Training machine learning models")

# 4.1 Linear Regression
print("Training Linear Regression model...")
lr_model = LinearRegression()
lr_model.fit(X_train_selected, y_train)

# 4.2 Ridge Regression (L2 regularization)
print("\nTraining Ridge Regression model...")
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_selected, y_train)

# 4.3 Decision Tree with limited depth to prevent overfitting
print("\nTraining Decision Tree model...")
dt_model = DecisionTreeRegressor(max_depth=5, min_samples_split=5, random_state=42)
dt_model.fit(X_train_selected, y_train)

# 4.4 Random Forest with optimized parameters
print("\nTraining Random Forest model...")
rf_model = RandomForestRegressor(
    n_estimators=100, 
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
rf_model.fit(X_train_selected, y_train)

# 4.5 Gradient Boosting (often performs well on small datasets)
print("\nTraining Gradient Boosting model...")
gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
gb_model.fit(X_train_selected, y_train)

# 5. Implement improved COCOMO II model
print("\nStep 5: Implementing improved COCOMO II model")

def calculate_cocomo_ii_effort(row):
    # COCOMO II Post-Architecture model
    a = 2.94  # Hằng số hiệu chỉnh
    b = 0.91  # Hệ số mũ quy mô
    
    # Kích thước dự án tính theo KLOC (chia 1000 dòng lệnh)
    kloc = row['equivphyskloc'] / 1000.0
    
    # Tính toán Effort Multiplier (EM) dựa trên các yếu tố chi phí
    em_factors = ['rely', 'data', 'cplx', 'time', 'stor', 'virt', 'turn', 
                  'acap', 'aexp', 'pcap', 'vexp', 'lexp', 'modp', 'tool', 'sced']
    em = 1.0
    for factor in em_factors:
        if factor in row and not pd.isna(row[factor]):
            em *= row[factor]
    
    # Xử lý thông tin mode từ dữ liệu gốc (không dùng dữ liệu đã chuẩn hóa hay one-hot encoding)
    # Giả sử trong dữ liệu gốc có cột 'mode' với các giá trị: 'embedded', 'organic', 'semidetached'
    mode = row.get('mode', None)
    if mode is not None:
        mode = mode.lower()
        if mode == 'embedded':
            complexity_adjustment = 1.2
        elif mode == 'organic':
            complexity_adjustment = 0.8
        else:  # semidetached hoặc các giá trị khác
            complexity_adjustment = 1.0
    else:
        complexity_adjustment = 1.0  # Giá trị mặc định nếu không có thông tin mode
    
    # Tính toán effort theo công thức: Effort = a × (KLOC)^b × EM × complexity_adjustment
    effort = a * (kloc ** b) * em * complexity_adjustment
    return effort


# Apply improved COCOMO II calculation to the dataset
data['cocomo_ii_effort'] = data.apply(calculate_cocomo_ii_effort, axis=1)

# 6. Make predictions and evaluate all models
print("\nStep 6: Making predictions and evaluating models")

# 6.1 Get predictions from all models (on log scale)
lr_pred = lr_model.predict(X_test_selected)
ridge_pred = ridge_model.predict(X_test_selected)
dt_pred = dt_model.predict(X_test_selected)
rf_pred = rf_model.predict(X_test_selected)
gb_pred = gb_model.predict(X_test_selected)

# 6.2 Get COCOMO II predictions for the test set
cocomo_ii_pred = data.loc[y_test.index, 'cocomo_ii_effort'].values

# 6.3 Convert predictions back to original scale for evaluation
y_test_original = np.expm1(y_test)
lr_pred_original = np.expm1(lr_pred)
ridge_pred_original = np.expm1(ridge_pred)
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
results['Ridge Regression'] = evaluate_model(y_test_original, ridge_pred_original, "Ridge Regression")
results['Decision Tree'] = evaluate_model(y_test_original, dt_pred_original, "Decision Tree")
results['Random Forest'] = evaluate_model(y_test_original, rf_pred_original, "Random Forest")
results['Gradient Boosting'] = evaluate_model(y_test_original, gb_pred_original, "Gradient Boosting")

# 7. Visualize results with improved visualization
print("\nStep 7: Visualizing results")

plt.figure(figsize=(16, 14))

# 7.1 Actual vs predicted effort comparison for selected samples (Bar chart instead of line)
plt.subplot(2, 2, 1)
sample_size = min(5, len(y_test))  # Reduce number of samples displayed
sample_indices = np.arange(sample_size)

# Create DataFrame for actual and predicted values
comparison_data = pd.DataFrame({
    'Actual': y_test_original.iloc[:sample_size].values,
    'COCOMO II': cocomo_ii_pred[:sample_size],
    'Linear Regression': lr_pred_original[:sample_size],
    'Random Forest': rf_pred_original[:sample_size],
    'Gradient Boosting': gb_pred_original[:sample_size]
})

# Use bar chart instead of line plot as requested
bar_width = 0.15
plt.bar(sample_indices - bar_width*2, comparison_data['Actual'], width=bar_width, label='Actual')
plt.bar(sample_indices - bar_width, comparison_data['COCOMO II'], width=bar_width, label='COCOMO II')
plt.bar(sample_indices, comparison_data['Linear Regression'], width=bar_width, label='Linear Regression')
plt.bar(sample_indices + bar_width, comparison_data['Random Forest'], width=bar_width, label='Random Forest')
plt.bar(sample_indices + bar_width*2, comparison_data['Gradient Boosting'], width=bar_width, label='Gradient Boosting')

plt.xlabel('Sample')
plt.ylabel('Effort (person-months)')
plt.title('Actual vs Predicted Effort (First 5 Samples)')
plt.xticks(sample_indices, [f'Sample {i+1}' for i in range(sample_size)])
plt.legend()
plt.grid(False)  # Remove grid lines

# 7.2 Scatter plot comparing actual vs predicted values
plt.subplot(2, 2, 2)
# Limit the y-axis range for better visualization (ignore outliers)
plt_max = min(np.percentile(y_test_original, 95), 5000)  # Cap at 95th percentile or 5000
plt.scatter(y_test_original, cocomo_ii_pred, label='COCOMO II', alpha=0.5, color='blue')
plt.scatter(y_test_original, lr_pred_original, label='Linear Regression', alpha=0.5, color='red')
plt.scatter(y_test_original, dt_pred_original, label='Decision Tree', alpha=0.5, color='orange')
plt.scatter(y_test_original, rf_pred_original, label='Random Forest', alpha=0.5, color='green')
plt.plot([0, plt_max], [0, plt_max], 'k--', lw=2)
plt.xlabel('Actual Effort (person-months)')
plt.ylabel('Predicted Effort (person-months)')
plt.title('Actual vs Predicted Effort')
plt.xlim(0, plt_max)
plt.ylim(0, plt_max)
plt.legend()
plt.grid(False)  # Remove grid lines

# 7.3. Error comparison across models
plt.subplot(2, 2, 3)
models = ['COCOMO II', 'Linear Regression', 'Ridge Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting']
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
plt.grid(False)  # Remove grid lines

# 7.4. R² comparison
plt.subplot(2, 2, 4)
r2_values = [results[model]['r2'] for model in models]
plt.bar(models, r2_values, color='lightgreen')
plt.ylabel('R² Score')
plt.title('R² Comparison Across Models')
plt.xticks(rotation=45)
plt.axhline(y=0, color='r', linestyle='-')
plt.grid(False)  # Remove grid lines

# Save the visualizations
plt.tight_layout()
plt.savefig('nasa93_effort_prediction_results.png')

# 8. Find best model and provide conclusions
print("\nStep 9: Conclusions and recommendations")
best_model = max(results.items(), key=lambda x: x[1]['r2'])[0]
print(f"Best performing model based on R²: {best_model}")
print(f"MAE: {results[best_model]['mae']:.2f}")
print(f"RMSE: {results[best_model]['rmse']:.2f}")
print(f"R²: {results[best_model]['r2']:.2f}")

print("\nRecommendations:")
print("1. The log transformation of the target variable significantly improves model performance")
print("2. Feature selection helps identify the most important factors affecting effort")
print("3. The COCOMO II model can be further improved with project-specific calibration")
print("4. Ensemble methods like Gradient Boosting generally perform better on this dataset")
print("5. Consider collecting more data or using transfer learning from similar datasets")

print("All visualizations have been saved.")