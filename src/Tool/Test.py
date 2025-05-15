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
            # Add KLOC input field
            input_data['equivphyskloc'] = (st.number_input('KLOC (K Lines of Code)', min_value=0.1, value=50.0, step=0.1, help="Estimated thousands of lines of code"))

        
        st.sidebar.subheader("Cost Drivers:")
        st.sidebar.markdown("#### Product Attributes")
        cost_driver_cols1 = st.sidebar.columns(2)
        
        with cost_driver_cols1[0]:
            input_data['rely'] = st.number_input('Required Reliability', 0.75, 1.40, 1.0, 0.05, help="How critical system failures are")
            input_data['data'] = st.number_input('Database Size', 0.94, 1.16, 1.0, 0.01, help="Ratio of database size to program size")
            input_data['cplx'] = st.number_input('Product Complexity', 0.70, 1.65, 1.0, 0.05, help="Complexity of the system's functions")
        
        with cost_driver_cols1[1]:
            input_data['time'] = st.number_input('Time Constraint', 1.00, 1.66, 1.0, 0.01, help="Execution time constraint")
            input_data['stor'] = st.number_input('Storage Constraint', 1.00, 1.56, 1.0, 0.01, help="Main storage constraint")
            input_data['virt'] = st.number_input('Platform Volatility', 0.87, 1.30, 1.0, 0.01, help="Hardware/software platform changes")
            input_data['turn'] = st.number_input('Turnaround Time', 0.87, 1.15, 1.0, 0.01, help="Development computer response time")
        
        st.sidebar.markdown("#### Personnel Attributes")
        cost_driver_cols2 = st.sidebar.columns(2)
        
        with cost_driver_cols2[0]:
            input_data['acap'] = st.number_input('Analyst Capability', 0.71, 1.46, 1.0, 0.01, help="Analyst team capability")
            input_data['aexp'] = st.number_input('Analyst Experience', 0.82, 1.29, 1.0, 0.01, help="Analyst experience with application")
            input_data['pcap'] = st.number_input('Programmer Capability', 0.70, 1.42, 1.0, 0.01, help="Programmer team capability")
            input_data['vexp'] = st.number_input('VM Experience', 0.90, 1.21, 1.0, 0.01, help="Virtual machine experience")
        
        with cost_driver_cols2[1]:
            input_data['lexp'] = st.number_input('Language Experience', 0.95, 1.14, 1.0, 0.01, help="Programming language experience")
            input_data['modp'] = st.number_input('Modern Practices', 0.82, 1.24, 1.0, 0.01, help="Use of modern programming practices")
            input_data['tool'] = st.number_input('Software Tools', 0.83, 1.24, 1.0, 0.01, help="Use of software tools")
            input_data['sced'] = st.number_input('Schedule Constraint', 1.00, 1.23, 1.0, 0.01, help="Required development schedule")
            
        # Calculate EAF (Effort Adjustment Factor)
        input_data['eaf'] = (input_data['rely'] * input_data['data'] * input_data['cplx'] * 
                             input_data['time'] * input_data['stor'] * input_data['virt'] * 
                             input_data['turn'] * input_data['acap'] * input_data['aexp'] * 
                             input_data['pcap'] * input_data['vexp'] * input_data['lexp'] * 
                             input_data['modp'] * input_data['tool'] * input_data['sced'])
    
    elif method == 'FP':
        # Function Point components
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
        
        # Productivity factors
        st.sidebar.subheader("Productivity Factors:")
        input_data['PDR_AFP'] = st.number_input('PDR_AFP (Productivity Derived from AFP)', min_value=0.0, value=1.0, help="Productivity derived from Adjusted Function Points")
        input_data['PDR_UFP'] = st.number_input('PDR_UFP (Productivity Derived from UFP)', min_value=0.0, value=1.0, help="Productivity derived from Unadjusted Function Points")
        input_data['NPDR_AFP'] = st.number_input('NPDR_AFP (Non-Productivity Derived from AFP)', min_value=0.0, value=0.5, help="Non-productivity derived from Adjusted Function Points")
        input_data['NPDU_UFP'] = st.number_input('NPDU_UFP (Non-Productivity Derived from UFP)', min_value=0.0, value=0.5, help="Non-productivity derived from Unadjusted Function Points")
    
    elif method == 'UCP':
        # Collect actor counts
        st.sidebar.subheader("Actors:")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            simple_actors = st.number_input('Simple Actors', min_value=0, value=3, help="Count of simple actors")
            average_actors = st.number_input('Average Actors', min_value=0, value=4, help="Count of average actors")
        
        with col2:
            complex_actors = st.number_input('Complex Actors', min_value=0, value=2, help="Count of complex actors")
        
        # Collect use case counts
        st.sidebar.subheader("Use Cases:")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            simple_uc = st.number_input('Simple Use Cases', min_value=0, value=6, help="Count of simple use cases")
            average_uc = st.number_input('Average Use Cases', min_value=0, value=8, help="Count of average use cases")
        
        with col2:
            complex_uc = st.number_input('Complex Use Cases', min_value=0, value=4, help="Count of complex use cases")
        
        # Technical and Environmental factors
        st.sidebar.subheader("Complexity Factors:")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            tcf = st.slider('Technical Complexity Factor (TCF)', 0.6, 1.3, 1.0, 0.01, 
                          help="Technical complexity factor (T1-T13)")
        
        with col2:
            ecf = st.slider('Environmental Complexity Factor (ECF)', 0.6, 1.3, 1.0, 0.01,
                          help="Environmental complexity factor (E1-E8)")
            
        # Save raw inputs for model
        input_data['Simple Actors'] = simple_actors
        input_data['Average Actors'] = average_actors  
        input_data['Complex Actors'] = complex_actors
        input_data['Simple UC'] = simple_uc
        input_data['Average UC'] = average_uc
        input_data['Complex UC'] = complex_uc
        
        # Calculate UAW and UUCW
        input_data['UAW'] = simple_actors * 1 + average_actors * 2 + complex_actors * 3
        input_data['UUCW'] = simple_uc * 5 + average_uc * 10 + complex_uc * 15
        input_data['TCF'] = tcf
        input_data['ECF'] = ecf
        
        # Calculate UCP and add productivity factor (default is 20 person-hours per UCP)
        input_data['UCP'] = (input_data['UAW'] + input_data['UUCW']) * tcf * ecf
        input_data['Real_P20'] = 20
        
        # Development environment selection
        st.sidebar.subheader("Development Environment:")
        language = st.sidebar.selectbox('Programming Language', 
                                      ['Java', 'C#', 'C++', 'Visual Basic', 'Other'], 
                                      index=0)
        methodology = st.sidebar.selectbox('Methodology',
                                         ['Waterfall', 'Agile', 'Incremental', 'Rapid Application Development', 'Other'],
                                         index=0)
        app_type = st.sidebar.selectbox('Application Type',
                                      ['Business Application', 'Real-Time Application', 'Mathematically-Intensive Application', 'Other'],
                                      index=0)
        
        # Save these values
        input_data['Language'] = language
        input_data['Methodology'] = methodology
        input_data['ApplicationType'] = app_type

    return input_data

# Function to prepare features for model prediction
def prepare_features(input_data, method):
    if method == 'LOC':
        # Create a dataframe with the input data in the correct format
        features = {
            'rely': input_data['rely'],
            'data': input_data['data'],
            'cplx': input_data['cplx'],
            'time': input_data['time'],
            'stor': input_data['stor'],
            'virt': input_data['virt'],
            'turn': input_data['turn'],
            'acap': input_data['acap'],
            'aexp': input_data['aexp'],
            'pcap': input_data['pcap'],
            'vexp': input_data['vexp'],
            'lexp': input_data['lexp'],
            'modp': input_data['modp'],
            'tool': input_data['tool'],
            'sced': input_data['sced'],
            'equivphyskloc': input_data['equivphyskloc'],
            'eaf': input_data['eaf'],
            # 'mode_embedded': input_data['mode_embedded'],
            # 'mode_organic': input_data['mode_organic'],
            # 'mode_semidetached': input_data['mode_semidetached']
        }
        return pd.DataFrame([features])
    
    elif method == 'FP':
        # Create a dataframe with the input data for FP
        feature_order = [
            'AFP', 'Input', 'Output', 'Enquiry', 'File', 'Interface', 
            'PDR_AFP', 'PDR_UFP', 'NPDR_AFP', 'NPDU_UFP'
        ]
        
        # Create features dictionary with the values in the correct order
        features = {key: input_data[key] for key in feature_order}
        return pd.DataFrame([features])
    
    elif method == 'UCP':
        # Create a DataFrame with the base features
        features = {
            'Simple Actors': input_data['Simple Actors'],
            'Average Actors': input_data['Average Actors'],
            'Complex Actors': input_data['Complex Actors'],
            'UAW': input_data['UAW'],
            'Simple UC': input_data['Simple UC'],
            'Average UC': input_data['Average UC'],
            'Complex UC': input_data['Complex UC'],
            'UUCW': input_data['UUCW'],
            'TCF': input_data['TCF'],
            'ECF': input_data['ECF'],
            'UCP': input_data['UCP'],
            'Real_P20': input_data['Real_P20']
        }
        
        df = pd.DataFrame([features])
        
        # Get the expected columns from the scaler
        expected_columns = scaler_ucp.feature_names_in_.tolist() if hasattr(scaler_ucp, 'feature_names_in_') else []
        
        # Create empty DataFrame with all expected columns (initialized with zeros)
        if expected_columns:
            full_df = pd.DataFrame(0, index=[0], columns=expected_columns)
            
            # Copy values from original df to full_df for columns that exist in both
            for col in df.columns:
                if col in full_df.columns:
                    full_df[col] = df[col]
            
            # Handle categorical variables that were one-hot encoded during training
            # Language encoding
            language_map = {
                'Java': 'Language_Java',
                'C#': 'Language_C#', 
                'C++': 'Language_C++',
                'Visual Basic': 'Language_Visual Basic',
                'Other': 'Language_Other'
            }
            if input_data['Language'] in language_map and language_map[input_data['Language']] in full_df.columns:
                full_df[language_map[input_data['Language']]] = 1
            
            # Methodology encoding
            methodology_map = {
                'Waterfall': 'Methodology_Waterfall',
                'Agile': 'Methodology_Agile',
                'Incremental': 'Methodology_Incremental',
                'Rapid Application Development': 'Methodology_Rapid Application Development',
                'Other': 'Methodology_Other'
            }
            if input_data['Methodology'] in methodology_map and methodology_map[input_data['Methodology']] in full_df.columns:
                full_df[methodology_map[input_data['Methodology']]] = 1
            
            # Application Type encoding
            app_type_map = {
                'Business Application': 'ApplicationType_Business Application',
                'Real-Time Application': 'ApplicationType_Real-Time Application',
                'Mathematically-Intensive Application': 'ApplicationType_Mathematically-Intensive Application',
                'Other': 'ApplicationType_Other'
            }
            if input_data['ApplicationType'] in app_type_map and app_type_map[input_data['ApplicationType']] in full_df.columns:
                full_df[app_type_map[input_data['ApplicationType']]] = 1
            
            # Debug info
            st.sidebar.markdown("### Debug Info")
            st.sidebar.markdown(f"Expected columns: {len(expected_columns)}")
            st.sidebar.markdown(f"Current columns: {len(full_df.columns)}")
            
            return full_df
        else:
            # Fallback if we can't get feature names
            st.warning("Could not determine expected features from the model. Prediction may not be accurate.")
            
            # Create comprehensive dummy variables for categorical columns
            df = pd.get_dummies(df, columns=['Language', 'Methodology', 'ApplicationType'], prefix=['Language', 'Methodology', 'ApplicationType'])
            
            return df
# Function to predict the effort using the selected model
def predict_effort(input_data, method):
    try:
        # Prepare the features based on the method
        features_df = prepare_features(input_data, method)
        
        if method == 'LOC':
            # For LOC model
            # Log transform KLOC as done in training
            features_df['equivphyskloc'] = np.log1p(features_df['equivphyskloc'])
            
            # Scale the features
            scaled_features = scaler_loc.transform(features_df)
            
            # Make prediction
            log_prediction = model_loc.predict(scaled_features)
            
            # Convert from log scale back to original scale
            prediction = np.expm1(log_prediction[0])
            
        elif method == 'FP':
            # For FP model
            # Scale the features
            scaled_features = scaler_fp.transform(features_df)
            
            # Make prediction 
            log_prediction = model_fp.predict(scaled_features)
            
            # Convert from log scale back to original scale
            prediction = np.expm1(log_prediction[0])
            
        elif method == 'UCP':
            # For UCP model
            # The challenge with UCP is ensuring the columns match exactly
            
            # First we need to get the expected columns from the scaler
            expected_columns = scaler_ucp.feature_names_in_.tolist() if hasattr(scaler_ucp, 'feature_names_in_') else None
            
            if expected_columns:
                st.sidebar.markdown(f"Expected columns: {expected_columns}")
                
                # Fill missing columns with 0
                for col in expected_columns:
                    if col not in features_df.columns:
                        features_df[col] = 0
                
                # Select only the expected columns in the right order
                features_df = features_df[expected_columns]
            
            # Scale the features
            scaled_features = scaler_ucp.transform(features_df)
            
            # Make prediction
            log_prediction = model_ucp.predict(scaled_features)
            
            # Convert from log scale back to original scale
            prediction = np.expm1(log_prediction[0])
            
        else:
            st.error(f"Unknown method: {method}")
            return None
        
        return prediction
        
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