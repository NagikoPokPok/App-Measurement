import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from pathlib import Path

# Set page config as the first Streamlit command
st.set_page_config(page_title="Software Project Effort Estimation Tool", layout="wide")

# Dynamic model discovery function
def find_model_files():
    """Search for model and scaler files in common locations."""
    script_path = Path(__file__).resolve()
    possible_base_dirs = [
        script_path.parent.parent,              # If Tool is at root level
        script_path.parent,                     # If tool.py is in root
        script_path.parent.parent.parent        # Original assumption
    ]
    
    # Try to locate model and scaler directories
    for base_dir in possible_base_dirs:
        # Look for model directory in common locations
        model_paths = [
            base_dir / "src" / "model",
            base_dir / "model",
            base_dir.parent / "src" / "model"
        ]
        
        # Look for scaler directory in common locations
        scaler_paths = [
            base_dir / "src" / "scaler",
            base_dir / "scaler",
            base_dir.parent / "src" / "scaler"
        ]
        
        # Check each model path
        for model_dir in model_paths:
            if any((model_dir / f"trained_model_{m}.pkl").exists() for m in ["loc", "ucp", "fp"]):
                # Found a model file, now find matching scaler dir
                for scaler_dir in scaler_paths:
                    if any((scaler_dir / f"scaler_{m}.pkl").exists() for m in ["loc", "ucp", "fp"]):
                        return model_dir, scaler_dir
    
    # If no models found, return default paths
    return script_path.parent.parent / "src" / "model", script_path.parent.parent / "src" / "scaler"

# Find model and scaler directories
MODEL_DIR, SCALER_DIR = find_model_files()

# Display paths in sidebar for debugging
st.sidebar.text(f"Models: {MODEL_DIR}")
st.sidebar.text(f"Scalers: {SCALER_DIR}")

# Function to check if model files exist
def check_model_files():
    missing_files = []
    model_files = [
        MODEL_DIR / "trained_model_loc.pkl",
        MODEL_DIR / "trained_model_ucp.pkl",
        MODEL_DIR / "trained_model_fp.pkl",
        SCALER_DIR / "scaler_loc.pkl",
        SCALER_DIR / "scaler_ucp.pkl",
        SCALER_DIR / "scaler_fp.pkl"
    ]
    
    for file in model_files:
        if not file.exists():
            missing_files.append(str(file))
    
    return missing_files

# Check for missing files
missing_files = check_model_files()
if missing_files:
    st.error("Missing model files. Please run the training scripts first.")
    st.write("Missing files:")
    for file in missing_files:
        st.write(f"- {file}")
    st.stop()

# Load the trained models
try:
    model_loc = joblib.load(MODEL_DIR / "trained_model_loc.pkl")
    model_ucp = joblib.load(MODEL_DIR / "trained_model_ucp.pkl")
    model_fp = joblib.load(MODEL_DIR / "trained_model_fp.pkl")

    scaler_loc = joblib.load(SCALER_DIR / "scaler_loc.pkl")
    scaler_ucp = joblib.load(SCALER_DIR / "scaler_ucp.pkl")
    scaler_fp = joblib.load(SCALER_DIR / "scaler_fp.pkl")
    
    st.sidebar.success("Models loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Constants for cost calculation
COST_PER_HOUR = 50
HOURS_PER_MONTH = 160

# Documentation for the methods
METHOD_INFO = {
    'LOC': """
    ## Lines of Code (LOC) Estimation
    
    This method estimates effort based on the number of lines of code and various COCOMO cost drivers.
    
    ### Input Parameters:
    - **Lines of Code (LOC)**: The estimated number of lines of code for the project
    - **Required Software Reliability (RELY)**: How critical system failures are (0.75 to 1.40)
    - **Database Size (DATA)**: The ratio of database size to program size (0.94 to 1.16)
    - **Product Complexity (CPLX)**: The complexity of the system's functions (0.70 to 1.65)
    - **Time Constraint (TIME)**: Execution time constraint (1.00 to 1.66)
    - **Main Storage Constraint (STOR)**: Main storage constraint (1.00 to 1.56)
    - **Virtual Machine Volatility (VIRT)**: Hardware/software platform changes (0.87 to 1.30)
    - **Computer Turnaround Time (TURN)**: Development computer response time (0.87 to 1.15)
    """,
    
    'FP': """
    ## Function Points (FP) Estimation
    
    Function Points measure software size based on the functionality provided.
    
    ### Input Parameters:
    - **Input**: Number of user inputs
    - **Output**: Number of user outputs
    - **Enquiry**: Number of user Enquiries
    - **File**: Number of files
    - **Interface**: Number of external interfaces
    - **Complexity adjustments**: 14 technical factors (0-5 scale)
    """,
    
    'UCP': """
    ## Use Case Points (UCP) Estimation
    
    Use Case Points estimate effort based on the complexity of use cases.
    
    ### Input Parameters:
    - **Simple Actors**: Number of simple actors (weight: 1)
    - **Average Actors**: Number of average actors (weight: 2)
    - **Complex Actors**: Number of complex actors (weight: 3)
    - **Simple Use Cases**: Number of simple use cases (weight: 5)
    - **Average Use Cases**: Number of average use cases (weight: 10)
    - **Complex Use Cases**: Number of complex use cases (weight: 15)
    - **Technical Factors**: 13 technical factors (0-5 scale)
    - **Environmental Factors**: 8 environmental factors (0-5 scale)
    """
}

# Function to create the input form for LOC, FP, and UCP
def create_effort_input_form(method):
    input_data = {}
    
    if method == 'LOC':
        col1, col2 = st.sidebar.columns(2)
        with col1:
            input_data['kloc'] = st.number_input('KLOC (thousands of LOC)', min_value=0.1, value=10.0, step=0.1, help="Estimated thousand lines of code") / 1000
        
        st.sidebar.subheader("Cost Drivers:")
        st.sidebar.markdown("#### Product Attributes")
        cost_driver_cols1 = st.sidebar.columns(2)
        
        with cost_driver_cols1[0]:
            input_data['rely'] = st.slider('Required Reliability', 0.75, 1.40, 1.0, 0.05, help="How critical system failures are")
            input_data['data'] = st.slider('Database Size', 0.94, 1.16, 1.0, 0.01, help="Ratio of database size to program size")
            input_data['cplx'] = st.slider('Product Complexity', 0.70, 1.65, 1.0, 0.05, help="Complexity of the system's functions")
        
        with cost_driver_cols1[1]:
            input_data['time'] = st.slider('Time Constraint', 1.00, 1.66, 1.0, 0.01, help="Execution time constraint")
            input_data['stor'] = st.slider('Storage Constraint', 1.00, 1.56, 1.0, 0.01, help="Main storage constraint")
            input_data['virt'] = st.slider('Platform Volatility', 0.87, 1.30, 1.0, 0.01, help="Hardware/software platform changes")
            input_data['turn'] = st.slider('Turnaround Time', 0.87, 1.15, 1.0, 0.01, help="Development computer response time")
        
        st.sidebar.markdown("#### Personnel Attributes")
        cost_driver_cols2 = st.sidebar.columns(2)
        
        with cost_driver_cols2[0]:
            input_data['acap'] = st.slider('Analyst Capability', 0.71, 1.46, 1.0, 0.01, help="Analyst team capability")
            input_data['aexp'] = st.slider('Analyst Experience', 0.82, 1.29, 1.0, 0.01, help="Analyst experience with application")
            input_data['pcap'] = st.slider('Programmer Capability', 0.70, 1.42, 1.0, 0.01, help="Programmer team capability")
            input_data['vexp'] = st.slider('VM Experience', 0.90, 1.21, 1.0, 0.01, help="Virtual machine experience")
        
        with cost_driver_cols2[1]:
            input_data['lexp'] = st.slider('Language Experience', 0.95, 1.14, 1.0, 0.01, help="Programming language experience")
            input_data['modp'] = st.slider('Modern Practices', 0.82, 1.24, 1.0, 0.01, help="Use of modern programming practices")
            input_data['tool'] = st.slider('Software Tools', 0.83, 1.24, 1.0, 0.01, help="Use of software tools")
            input_data['sced'] = st.slider('Schedule Constraint', 1.00, 1.23, 1.0, 0.01, help="Required development schedule")
        
        # Add mode selection (will be one-hot encoded)
        st.sidebar.markdown("#### Project Mode")
        mode_options = ['Organic', 'Semi-detached', 'Embedded']
        selected_mode = st.sidebar.radio("Development Mode", mode_options, index=0)
        
        # Create one-hot encoding manually
        if selected_mode == 'Organic':
            input_data['mode_e'] = 0
            input_data['mode_o'] = 1
            input_data['mode_s'] = 0
        elif selected_mode == 'Semi-detached':
            input_data['mode_e'] = 0
            input_data['mode_o'] = 0
            input_data['mode_s'] = 1
        else:  # Embedded
            input_data['mode_e'] = 1
            input_data['mode_o'] = 0
            input_data['mode_s'] = 0
    
    elif method == 'FP':
        # Các trường cho FP
        st.sidebar.subheader("Function Point Components:")
        col1, col2 = st.sidebar.columns(2)

        with col1:
            input_data['AFP'] = st.number_input('AFP (Adjusted Function Points)', min_value=0.0, value=1.0, help="Adjusted Function Points")
            input_data['Input'] = st.number_input('Input Count', min_value=0, value=30, help="Number of user inputs")
            input_data['Output'] = st.number_input('Output Count', min_value=0, value=25, help="Number of user outputs")
            input_data['Enquiry'] = st.number_input('Enquiry Count', min_value=0, value=15, help="Number of user enquiries")

        with col2:
            input_data['File'] = st.number_input('File Count', min_value=0, value=10, help="Number of files")
            input_data['Interface'] = st.number_input('Interface Count', min_value=0, value=5, help="Number of external interfaces")
        
        # Các trường năng suất cần nhập: PDR_AFP, PDR_UFP, NPDR_AFP, NPDU_UFP
        st.sidebar.subheader("Productivity Factors:")
        input_data['PDR_AFP'] = st.number_input('PDR_AFP (Productivity Derived from AFP)', min_value=0.0, value=1.0, help="Productivity derived from Adjusted Function Points")
        input_data['PDR_UFP'] = st.number_input('PDR_UFP (Productivity Derived from UFP)', min_value=0.0, value=1.0, help="Productivity derived from Unadjusted Function Points")
        input_data['NPDR_AFP'] = st.number_input('NPDR_AFP (Non-Productivity Derived from AFP)', min_value=0.0, value=0.5, help="Non-productivity derived from Adjusted Function Points")
        input_data['NPDU_UFP'] = st.number_input('NPDU_UFP (Non-Productivity Derived from UFP)', min_value=0.0, value=0.5, help="Non-productivity derived from Unadjusted Function Points")
    
    elif method == 'UCP':
        st.sidebar.subheader("Actors:")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            simple_actors = st.number_input('Simple Actors', min_value=0, value=3, help="Count of simple actors")
            average_actors = st.number_input('Average Actors', min_value=0, value=4, help="Count of average actors")
        
        with col2:
            complex_actors = st.number_input('Complex Actors', min_value=0, value=2, help="Count of complex actors")
        
        st.sidebar.subheader("Use Cases:")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            simple_uc = st.number_input('Simple Use Cases', min_value=0, value=6, help="Count of simple use cases")
            average_uc = st.number_input('Average Use Cases', min_value=0, value=8, help="Count of average use cases")
        
        with col2:
            complex_uc = st.number_input('Complex Use Cases', min_value=0, value=4, help="Count of complex use cases")
        
        # Calculate UAW and UUCW
        input_data['UAW'] = simple_actors * 1 + average_actors * 2 + complex_actors * 3
        input_data['UUCW'] = simple_uc * 5 + average_uc * 10 + complex_uc * 15
        
        # Add the rest of the UCP model features
        st.sidebar.subheader("Technical Factors:")
        tech_factors = [
            "T1_Distributed_System",
            "T2_Response_Objectives",
            "T3_End_User_Efficiency",
            "T4_Complex_Processing",
            "T5_Reusable_Code",
            "T6_Easy_to_Install",
            "T7_Easy_to_Use",
            "T8_Portable",
            "T9_Easy_to_Change",
            "T10_Concurrent",
            "T11_Security_Features",
            "T12_Access_for_Third_Parties",
            "T13_Special_Training_Required"
        ]
        
        # Display technical factors as slider inputs
        for i, factor in enumerate(tech_factors):
            input_data[factor] = st.sidebar.slider(f'T{i+1}: {factor.replace("_", " ")}', 0, 5, 3, 1, 
                                             help=f"Rate from 0 (no influence) to 5 (strong influence)")
        
        st.sidebar.subheader("Environmental Factors:")
        env_factors = [
            "E1_Familiar_with_RUP",
            "E2_Application_Experience",
            "E3_OO_Programming_Experience",
            "E4_Lead_Analyst_Capability",
            "E5_Motivation",
            "E6_Stable_Requirements",
            "E7_Part_Time_Workers",
            "E8_Difficult_Programming_Language"
        ]
        
        # Display environmental factors as slider inputs
        for i, factor in enumerate(env_factors):
            input_data[factor] = st.sidebar.slider(f'E{i+1}: {factor.replace("_", " ")}', 0, 5, 3, 1, 
                                             help=f"Rate from 0 (no influence) to 5 (strong influence)")
    
    return input_data

# Function to prepare features for model prediction
def prepare_features(input_data, method):
    if method == 'LOC':
        # Directly use the input data since it's already in the expected format
        features = pd.DataFrame([input_data])
        return features
    
    elif method == 'FP':
        
        feature_order = [
            'AFP', 'Input', 'Output', 'Enquiry', 'File', 'Interface', 
            'PDR_AFP', 'PDR_UFP', 'NPDR_AFP', 'NPDU_UFP'
        ]
        
        # Tạo một dictionary với các giá trị đã nhập theo thứ tự trường đã huấn luyện
        china_features = {key: input_data[key] for key in feature_order}

        return pd.DataFrame([china_features])

        return pd.DataFrame([china_features])
    
    elif method == 'UCP':
        # Start with the UAW and UUCW
        ucp_features = {
            'UAW': input_data['UAW'],
            'UUCW': input_data['UUCW']
        }
        
        # Calculate TCF (Technical Complexity Factor)
        tech_factors = {
            'T1_Distributed_System': 2.0,
            'T2_Response_Objectives': 1.0,
            'T3_End_User_Efficiency': 1.0,
            'T4_Complex_Processing': 1.0,
            'T5_Reusable_Code': 1.0,
            'T6_Easy_to_Install': 0.5,
            'T7_Easy_to_Use': 0.5,
            'T8_Portable': 2.0,
            'T9_Easy_to_Change': 1.0,
            'T10_Concurrent': 1.0,
            'T11_Security_Features': 1.0,
            'T12_Access_for_Third_Parties': 1.0,
            'T13_Special_Training_Required': 1.0
        }
        
        tcf_sum = sum(tech_factors[factor] * input_data[factor] for factor in tech_factors)
        tcf = 0.6 + (0.01 * tcf_sum)
        
        # Calculate ECF (Environmental Complexity Factor)
        env_factors = {
            'E1_Familiar_with_RUP': 1.5,
            'E2_Application_Experience': 0.5,
            'E3_OO_Programming_Experience': 1.0,
            'E4_Lead_Analyst_Capability': 0.5,
            'E5_Motivation': 1.0,
            'E6_Stable_Requirements': 2.0,
            'E7_Part_Time_Workers': -1.0,
            'E8_Difficult_Programming_Language': -1.0
        }
        
        ecf_sum = sum(env_factors[factor] * input_data[factor] for factor in env_factors)
        ecf = 1.4 + (-0.03 * ecf_sum)
        
        # Calculate UCP (Use Case Points)
        ucp = input_data['UAW'] + input_data['UUCW']
        ucp_features['UCP'] = ucp
        ucp_features['TCF'] = tcf
        ucp_features['ECF'] = ecf
        ucp_features['Real_P20'] = 20  # Productivity factor, typical constant
        
        # Add all technical and environmental factors to match training data
        for factor in list(tech_factors.keys()) + list(env_factors.keys()):
            ucp_features[factor] = input_data[factor]
        
        return pd.DataFrame([ucp_features])

# Function to predict the effort using the selected model
def predict_effort(input_data, method):
    try:
        # Prepare the features based on the method
        features_df = prepare_features(input_data, method)
        
        if method == 'LOC':
            # Ensure column order matches training data
            scaled_input = scaler_loc.transform(features_df)
            prediction = model_loc.predict(scaled_input)
        elif method == 'FP':
            # Ensure column order matches training data
            scaled_input = scaler_fp.transform(features_df)
            prediction = model_fp.predict(scaled_input)
        elif method == 'UCP':
            # Ensure column order matches training data
            scaled_input = scaler_ucp.transform(features_df)
            prediction = model_ucp.predict(scaled_input)
        
        # Convert from log scale back to original scale
        return np.expm1(prediction[0])
    except Exception as e:
        st.error(f"Prediction error: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

# Function to estimate project metrics (effort, cost, and time)
def estimate_project_metrics(effort):
    hours = effort
    cost = hours * COST_PER_HOUR
    months = hours / HOURS_PER_MONTH
    return hours, cost, months

# Display results
def display_results(pred_effort, method):
    hours, cost, months = estimate_project_metrics(pred_effort)
    
    # Create metrics for key values
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Effort (Hours)", f"{hours:.0f}")
    col2.metric("Cost (USD)", f"${cost:,.2f}")
    col3.metric("Duration (Months)", f"{months:.1f}")
    col4.metric("Team Size (for 1 month)", f"{months:.1f} people")
    
    # Create a detailed breakdown
    st.subheader("Project Details")
    details = {
        "Metric": ["Estimation Method", "Estimated Effort", "Hourly Rate", "Total Cost", "Project Duration", "Required Team Size (1 month)"],
        "Value": [method, f"{hours:.0f} hours", f"${COST_PER_HOUR:.2f}/hour", f"${cost:,.2f}", f"{months:.1f} months", f"{months:.1f} people"]
    }
    st.table(pd.DataFrame(details))
    
    # Create a visualization
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Cost Breakdown Pie Chart
    labels = ['Development', 'Testing', 'Management']
    sizes = [cost * 0.7, cost * 0.2, cost * 0.1]  # 70% development, 20% testing, 10% management
    ax[0].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax[0].set_title('Cost Breakdown')
    
    # Timeline Bar Chart
    phases = ['Requirements', 'Design', 'Development', 'Testing', 'Deployment']
    phase_times = [months * 0.1, months * 0.2, months * 0.4, months * 0.2, months * 0.1]  # As fractions of total time
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#C2C2F0']
    
    ax[1].barh(phases, phase_times, color=colors)
    ax[1].set_xlabel('Months')
    ax[1].set_title('Project Timeline')
    
    plt.tight_layout()
    st.pyplot(fig)

# Main function for the Streamlit interface
def main():
    # Title and description
    st.title("Software Project Effort Estimation Tool")
    st.markdown("""
    This tool helps you estimate the effort required for software development projects using different estimation methods:
    - **Lines of Code (LOC)**: Based on COCOMO model
    - **Function Points (FP)**: Based on function point analysis
    - **Use Case Points (UCP)**: Based on use case complexity
    
    Select an estimation method from the sidebar to get started.
    """)
    
    # Sidebar configuration
    st.sidebar.title("Estimation Settings")
    
    # Select model in sidebar
    method = st.sidebar.selectbox("Select Estimation Method", ["LOC", "FP", "UCP"])
    
    # Show method information
    with st.expander("About this estimation method", expanded=False):
        st.markdown(METHOD_INFO[method])
    
    # Generate the input form based on the selected estimation method
    data = create_effort_input_form(method)
    
    # Calculate button
    if st.sidebar.button("Calculate Effort", type="primary"):
        with st.spinner('Calculating...'):
            pred_effort = predict_effort(data, method)
            
            if pred_effort is not None:
                st.success(f"Estimation completed using {method} method!")
                display_results(pred_effort, method)
            else:
                st.error("Estimation failed. Please check your inputs and try again.")
    
    # Export button
    if 'pred_effort' in locals():
        if st.download_button(
            label="Export Results",
            data="",  # Would implement actual export functionality here
            file_name="effort_estimation.csv",
            mime="text/csv",
        ):
            st.success("Results exported successfully!")

if __name__ == "__main__":
    main()