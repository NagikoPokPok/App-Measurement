import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load the trained models
model_loc = joblib.load('./src/model/trained_model_loc.pkl')
model_ucp = joblib.load('./src/model/trained_model_ucp.pkl')
model_fp = joblib.load('./src/model/trained_model_fp.pkl')

scaler_loc = joblib.load('./src/scaler/scaler_loc.pkl')  # Assuming separate scalers for each model
scaler_ucp = joblib.load('./src/scaler/scaler_ucp.pkl')
scaler_fp = joblib.load('./src/scaler/scaler_fp.pkl')

# Constants for cost calculation
COST_PER_HOUR = 50
HOURS_PER_MONTH = 160

# Function to create the input form for LOC, FP, and UCP
def create_effort_input_form(method):
    input_data = {}
    
    if method == 'LOC':
        input_data['loc'] = st.sidebar.number_input('Lines of Code (LOC)', min_value=0)
        input_data['rely'] = st.sidebar.number_input('Required Software Reliability', min_value=0.0)
        input_data['data'] = st.sidebar.number_input('Database Size', min_value=0.0)
        input_data['cplx'] = st.sidebar.number_input('Product Complexity', min_value=0.0)
        input_data['time'] = st.sidebar.number_input('Time Constraint', min_value=0.0)
        input_data['stor'] = st.sidebar.number_input('Main Storage Constraint', min_value=0.0)
        input_data['virt'] = st.sidebar.number_input('Virtual Machine Volatility', min_value=0.0)
        input_data['turn'] = st.sidebar.number_input('Computer Turnaround Time', min_value=0.0)
        input_data['acap'] = st.sidebar.number_input('Analyst Capability', min_value=0.0)
        input_data['aexp'] = st.sidebar.number_input('Applications Experience', min_value=0.0)
        input_data['pcap'] = st.sidebar.number_input('Programmer Capability', min_value=0.0)
        input_data['vexp'] = st.sidebar.number_input('Virtual Machine Experience', min_value=0.0)
        input_data['lexp'] = st.sidebar.number_input('Programming Language Experience', min_value=0.0)
        input_data['modp'] = st.sidebar.number_input('Modern Programming Practices', min_value=0.0)
        input_data['tool'] = st.sidebar.number_input('Use of Software Tools', min_value=0.0)
        input_data['sced'] = st.sidebar.number_input('Required Development Schedule', min_value=0.0)
    
    elif method == 'FP':
        input_data['Simple Actors'] = st.sidebar.number_input('Simple Actors', min_value=0)
        input_data['Average Actors'] = st.sidebar.number_input('Average Actors', min_value=0)
        input_data['Complex Actors'] = st.sidebar.number_input('Complex Actors', min_value=0)
        input_data['TCF'] = st.sidebar.number_input('Technical Complexity Factor', min_value=0.0)
        input_data['ECF'] = st.sidebar.number_input('Environmental Complexity Factor', min_value=0.0)
        
    elif method == 'UCP':
        input_data['UAW'] = st.sidebar.number_input('Actor Weight Total', min_value=0)
        input_data['UUCW'] = st.sidebar.number_input('Use Case Weight Total', min_value=0)
        input_data['TCF'] = st.sidebar.number_input('Technical Complexity Factor', min_value=0.0)
        input_data['ECF'] = st.sidebar.number_input('Environmental Complexity Factor', min_value=0.0)
    
    return input_data

# Function to predict the effort using the selected model
def predict_effort(input_data, method):
    if method == 'LOC':
        scaled_input = scaler_loc.transform(np.array(list(input_data.values())).reshape(1, -1))
        prediction = model_loc.predict(scaled_input)
    elif method == 'FP':
        scaled_input = scaler_fp.transform(np.array(list(input_data.values())).reshape(1, -1))
        prediction = model_fp.predict(scaled_input)
    elif method == 'UCP':
        scaled_input = scaler_ucp.transform(np.array(list(input_data.values())).reshape(1, -1))
        prediction = model_ucp.predict(scaled_input)
    
    return prediction[0]

# Function to estimate project metrics (effort, cost, and time)
def estimate_project_metrics(effort):
    hours = effort * HOURS_PER_MONTH
    cost = hours * COST_PER_HOUR
    months = hours / HOURS_PER_MONTH
    return hours, cost, months

# Display results
def display_results(pred_effort):
    hours, cost, months = estimate_project_metrics(pred_effort)
    st.write(f"### Estimated Effort (in Person Hours): {pred_effort:.2f}")
    st.write(f"Estimated Hours: {hours:.0f}")
    st.write(f"Estimated Cost: ${cost:.2f}")
    st.write(f"Estimated Months: {months:.1f}")

# Main function for the Streamlit interface
def main():
    st.title("Software Project Effort Estimation Tool")

    # Select model in sidebar
    method = st.sidebar.selectbox("Select Estimation Method", ["LOC", "FP", "UCP"])
    
    # Generate the input form based on the selected estimation method
    data = create_effort_input_form(method)
    
    # Layout: inputs on the left and results on the right
    col1, col2 = st.columns(2)

    with col1:
        # When button is clicked, calculate and display results
        if st.button("Calculate Effort"):
            pred_effort = predict_effort(data, method)
            with col2:
                display_results(pred_effort)

if __name__ == "__main__":
    main()