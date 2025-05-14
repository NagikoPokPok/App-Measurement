import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

@st.cache
def load_and_process_data():
    ucp_data = pd.read_csv('dataset/UCP_Dataset.csv')
    data = ucp_data.copy()
    for col in data.columns:
        if data[col].dtype == 'object':
            fill_value = data[col].mode()[0]
        else:
            fill_value = data[col].median()
        data.fillna({col: fill_value}, inplace=True)

    y = data['Real_Effort_Person_Hours']
    columns_to_drop = ['Project_No', 'Real_Effort_Person_Hours', 'Real_P20']
    X = data.drop(columns_to_drop, axis=1)
    categorical_cols = X.select_dtypes(include=['object']).columns
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    scaler = StandardScaler()
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
    X_scaled = X.copy()
    X_scaled[numeric_features] = scaler.fit_transform(X[numeric_features])
    return X_scaled, y, X.columns

def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'RMSE': rmse, 'R²': r2}

def app():
    st.title("Software Effort Estimation")

    X_scaled, y, all_columns = load_and_process_data()

    # Model selection with 4 options
    model_choice = st.sidebar.selectbox(
        'Select Model',
        ['Linear Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting']
    )

    # Model mapping
    if model_choice == 'Linear Regression':
        model = LinearRegression()
    elif model_choice == 'Decision Tree':
        model = DecisionTreeRegressor(max_depth=5, random_state=42)
    elif model_choice == 'Random Forest':
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    else:
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    st.sidebar.header("Enter Project Data")
    input_data = {}
    for col in all_columns:
        input_data[col] = st.sidebar.number_input(f'{col}', value=0.0, format="%.4f")

    input_df = pd.DataFrame([input_data])
    input_df_encoded = pd.get_dummies(input_df, columns=input_df.select_dtypes(include=['object']).columns, drop_first=True)
    missing_cols = set(all_columns) - set(input_df_encoded.columns)
    for col in missing_cols:
        input_df_encoded[col] = 0
    input_df_encoded = input_df_encoded[all_columns]

    scaler = StandardScaler()
    scaler.fit(X_scaled)
    input_scaled = scaler.transform(input_df_encoded)

    prediction = model.predict(input_scaled)

    st.write(f"Predicted Effort (in Person Hours): {prediction[0]:.2f}")

    predictions = model.predict(X_test)
    results = evaluate_model(y_test, predictions)
    st.write(f"Model Performance:")
    st.write(f"MAE: {results['MAE']:.2f}")
    st.write(f"RMSE: {results['RMSE']:.2f}")
    st.write(f"R²: {results['R²']:.2f}")

if __name__ == "__main__":
    app()