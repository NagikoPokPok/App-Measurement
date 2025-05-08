import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load('trained_model.pkl')

# Constants
COST_PER_HOUR = 50
HOURS_PER_MONTH = 160

def create_effort_input_form(method):
    input_data = {}
    
    if method == 'LOC':
        col1, col2 = st.columns(2)
        
        with col1:
            input_data['loc'] = st.number_input('Lines of Code (LOC)', min_value=0)
            input_data['rely'] = st.number_input('Required Software Reliability', min_value=0.0)
            input_data['data'] = st.number_input('Database Size', min_value=0.0)
            input_data['cplx'] = st.number_input('Product Complexity', min_value=0.0)
            input_data['time'] = st.number_input('Time Constraint', min_value=0.0)
            input_data['stor'] = st.number_input('Main Storage Constraint', min_value=0.0)
            input_data['virt'] = st.number_input('Virtual Machine Volatility', min_value=0.0)
            input_data['turn'] = st.number_input('Computer Turnaround Time', min_value=0.0)
        
        with col2:
            input_data['acap'] = st.number_input('Analyst Capability', min_value=0.0)
            input_data['aexp'] = st.number_input('Applications Experience', min_value=0.0)
            input_data['pcap'] = st.number_input('Programmer Capability', min_value=0.0)
            input_data['vexp'] = st.number_input('Virtual Machine Experience', min_value=0.0)
            input_data['lexp'] = st.number_input('Programming Language Experience', min_value=0.0)
            input_data['modp'] = st.number_input('Modern Programming Practices', min_value=0.0)
            input_data['tool'] = st.number_input('Use of Software Tools', min_value=0.0)
            input_data['sced'] = st.number_input('Required Development Schedule', min_value=0.0)
            
        input_data['actual'] = st.number_input('Actual Effort', min_value=0.0)
    
    elif method == 'FP':
        col1, col2, col3 = st.columns(3)
        with col1:
            input_data['Simple Actors'] = st.number_input('Simple Actors', min_value=0)
        with col2:
            input_data['Average Actors'] = st.number_input('Average Actors', min_value=0)
        with col3:
            input_data['Complex Actors'] = st.number_input('Complex Actors', min_value=0)
            
        input_data['TCF'] = st.number_input('Technical Complexity Factor', min_value=0.0)
        input_data['ECF'] = st.number_input('Environmental Complexity Factor', min_value=0.0)
        
    elif method == 'UCP':
        input_data['UAW'] = st.number_input('Actor Weight Total', min_value=0)
        input_data['UUCW'] = st.number_input('Use Case Weight Total', min_value=0)
        input_data['TCF'] = st.number_input('Technical Complexity Factor', min_value=0.0)
        input_data['ECF'] = st.number_input('Environmental Complexity Factor', min_value=0.0)
    
    return pd.DataFrame([input_data])


model = joblib.load('trained_model.pkl')
scaler = joblib.load('scaler.pkl')

def predict_effort(input_data):
    # Preprocessing the input data: Scaling
    scaled_input = scaler.transform(input_data)
    
    # Predict using the loaded model
    predicted_effort = model.predict(scaled_input)
    return predicted_effort

def estimate_project_metrics(effort):
    hours = effort * HOURS_PER_MONTH
    cost = hours * COST_PER_HOUR
    months = hours / HOURS_PER_MONTH
    return hours, cost, months

def display_results(predicted_effort, actual_effort):
    st.subheader("Estimation Results")
    
    # Calculate metrics
    hours, cost, months = estimate_project_metrics(predicted_effort)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Estimated Hours", f"{hours:,.0f}")
    with col2:
        st.metric("Estimated Cost", f"${cost:,.2f}")
    with col3:
        st.metric("Estimated Months", f"{months:.1f}")
    
    # Display comparison if actual effort is provided
    if actual_effort > 0:
        mae = mean_absolute_error([actual_effort], [predicted_effort])
        rmse = np.sqrt(mean_squared_error([actual_effort], [predicted_effort]))
        r2 = r2_score([actual_effort], [predicted_effort])
        
        st.subheader("Model Performance")
        st.write(f"Mean Absolute Error: {mae:.2f}")
        st.write(f"Root Mean Squared Error: {rmse:.2f}")
        st.write(f"R² Score: {r2:.2f}")

def main():
    st.title("Software Project Effort Estimation Tool")
    
    # Method selection
    method = st.selectbox("Select Estimation Method", ["LOC", "Function Points", "Use Case Points"])
    
    input_method = st.radio("Choose Input Method", ["Upload File", "Manual Input"])
    
    if input_method == "Manual Input":
        data = create_effort_input_form(method)
        
        if st.button("Calculate Effort"):
            predicted_effort = predict_effort(data)
            
            actual_effort = data['actual'].iloc[0]
            display_results(predicted_effort, actual_effort)

if __name__ == "__main__":
    main()
