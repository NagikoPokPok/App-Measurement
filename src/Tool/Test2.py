import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile
import pandas as pd
from pathlib import Path
import math 

# Set page config as the first Streamlit command
st.set_page_config(page_title="Software Project Effort Estimation Tool", layout="wide")

# Dynamic model discovery function
def find_model_files():
    """Search for model, scaler and img files in common locations."""
    script_path = Path(__file__).resolve()
    base_dir = script_path.parent.parent.parent  # App Measurement directory
    
    # Define correct paths relative to base directory
    model_dir = base_dir / "src" / "model"
    scaler_dir = base_dir / "src" / "scaler"
    img_dir = base_dir / "img"
    
    # Create directories if they don't exist
    model_dir.mkdir(parents=True, exist_ok=True)
    scaler_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if required files exist
    missing_files = []
    for method in ["loc", "ucp", "fp"]:
        required_files = [
            (model_dir / f"{method}_linear_model.pkl", "Model"),
            (model_dir / f"{method}_decision_tree_model.pkl", "Model"),
            (model_dir / f"{method}_random_forest_model.pkl", "Model"), 
            (model_dir / f"{method}_gradient_boosting_model.pkl", "Model"),
            (scaler_dir / f"{method}_scaler.pkl", "Scaler"),
            (img_dir / f"{method}_model_comparison.csv", "Comparison")
        ]
        
        for file_path, file_type in required_files:
            if not file_path.exists():
                missing_files.append(f"{file_type}: {file_path.name}")
    
    if missing_files:
        st.warning(f"""
        Some required files are missing. Please run Train_Model.py first.
        
        Missing files:
        {chr(10).join('- ' + f for f in missing_files)}
        
        Expected locations:
        - Models: {model_dir}
        - Scalers: {scaler_dir}
        - Comparisons: {img_dir}
        """)
    
    return model_dir, scaler_dir, img_dir
# Find model, scaler and img directories
MODEL_DIR, SCALER_DIR, IMG_DIR = find_model_files()

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
        # 🟢 Bước 1: Giá trị mặc định là 0
        default_values = {
            'equivphyskloc': 1.0,
            'rely': 1.0,
            'data': 1.0,
            'cplx': 1.0,
            'time': 1.0,
            'stor': 1.0,
            'virt': 1.0,
            'turn': 1.0,
            'acap': 1.0,
            'aexp': 1.0,
            'pcap': 1.0,
            'vexp': 1.0,
            'lexp': 1.0,
            'modp': 1.0,
            'tool': 1.0,
            'sced': 1.0
        }

        # 🟢 Bước 2: Hiển thị file uploader
        st.sidebar.markdown("### (Tùy chọn) Tải dữ liệu từ file Excel")
        uploaded_file = st.sidebar.file_uploader("Chọn file Excel", type=["xlsx"])

        # 🟢 Bước 3: Nếu có file, đọc và cập nhật default_values
        if uploaded_file is not None:
            try:
                df = pd.read_excel(uploaded_file)
                expected_cols = {'equivphyskloc', 'rely', 'data', 'cplx', 'time', 'stor', 'virt', 'turn', 'acap', 'aexp', 'pcap', 'vexp', 'lexp', 'modp', 'tool', 'sced'}
                if not df.empty and expected_cols.issubset(df.columns):
                    default_values['equivphyskloc'] = float(df['equivphyskloc'].iloc[0])
                    default_values['rely'] = float(df['rely'].iloc[0])
                    default_values['data'] = float(df['data'].iloc[0])
                    default_values['cplx'] = float(df['cplx'].iloc[0])
                    default_values['time'] = float(df['time'].iloc[0])
                    default_values['stor'] = float(df['stor'].iloc[0])
                    default_values['virt'] = float(df['virt'].iloc[0])
                    default_values['turn'] = float(df['turn'].iloc[0])
                    default_values['acap'] = float(df['acap'].iloc[0])
                    default_values['aexp'] = float(df['aexp'].iloc[0])
                    default_values['pcap'] = float(df['pcap'].iloc[0])
                    default_values['vexp'] = float(df['vexp'].iloc[0])
                    default_values['lexp'] = float(df['lexp'].iloc[0])
                    default_values['modp'] = float(df['modp'].iloc[0])
                    default_values['tool'] = float(df['tool'].iloc[0])
                    default_values['sced'] = float(df['sced'].iloc[0])

            except Exception as e:
                st.sidebar.error(f"Lỗi khi đọc file: {e}")


        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            # Add KLOC input field
            input_data['equivphyskloc'] = st.number_input('KLOC (K Lines of Code)', min_value=0.1, value=default_values['equivphyskloc'], step=0.1, help="Estimated thousands of lines of code")

            # Add mode selection
            mode = st.selectbox('Development Mode', ['Organic', 'Semi-detached', 'Embedded'], help="Software development mode")
            # Encode mode as dummy variables
            if mode == 'Organic':
                input_data['mode_embedded'] = 0
                input_data['mode_organic'] = 1
                input_data['mode_semidetached'] = 0
            elif mode == 'Semi-detached':
                input_data['mode_embedded'] = 0
                input_data['mode_organic'] = 0
                input_data['mode_semidetached'] = 1
            else:  # Embedded
                input_data['mode_embedded'] = 1
                input_data['mode_organic'] = 0
                input_data['mode_semidetached'] = 0
             
        st.sidebar.subheader("Cost Drivers:")
        st.sidebar.markdown("#### Product Attributes")
        cost_driver_cols1 = st.sidebar.columns(2)
        
        with cost_driver_cols1[0]:
            input_data['rely'] = st.number_input('Required Reliability', min_value=0.0, max_value=1.40, value=default_values['rely'], help="How critical system failures are")
            input_data['data'] = st.number_input('Database Size', min_value=0.94, max_value=1.16, value=default_values['data'], help="Ratio of database size to program size")
            input_data['cplx'] = st.number_input('Product Complexity', min_value=0.70, max_value=1.65, value=default_values['cplx'], help="Complexity of the system's functions")
        
        with cost_driver_cols1[1]:
            
            input_data['time'] = st.number_input('Time Constraint', min_value=1.00, max_value=1.66, value=default_values['time'], help="Execution time constraint")
            input_data['stor'] = st.number_input('Storage Constraint', min_value=1.00, max_value=1.56, value=default_values['stor'], help="Main storage constraint")
            input_data['virt'] = st.number_input('Virtual Machine Volatility', min_value=0.87, max_value=1.30, value=default_values['virt'], help="Hardware/software platform changes")
            input_data['turn'] = st.number_input('Turnaround Time', min_value=0.87, max_value=1.15, value=default_values['turn'], help="Development computer response time")
        
        st.sidebar.markdown("#### Personnel Attributes")
        cost_driver_cols2 = st.sidebar.columns(2)
        
        with cost_driver_cols2[0]:
            input_data['acap'] = st.number_input('Analyst Capability', 0.71, 1.46, value=default_values['acap'], help="Analyst team capability")
            input_data['aexp'] = st.number_input('Analyst Experience', 0.82, 1.29, value=default_values['aexp'], help="Analyst experience with application")
            input_data['pcap'] = st.number_input('Programmer Capability', 0.70, 1.42, value=default_values['pcap'], help="Programmer team capability")
            input_data['vexp'] = st.number_input('VM Experience', 0.90, 1.21, value=default_values['vexp'], help="Virtual machine experience")
        
        with cost_driver_cols2[1]:
            input_data['lexp'] = st.number_input('Language Experience', 0.95, 1.14, value=default_values['lexp'], help="Programming language experience")
            input_data['modp'] = st.number_input('Modern Practices', 0.82, 1.24, value=default_values['modp'], help="Use of modern programming practices")
            input_data['tool'] = st.number_input('Software Tools', 0.83, 1.24, value=default_values['tool'], help="Use of software tools")
            input_data['sced'] = st.number_input('Schedule Constraint', 1.00, 1.23, value=default_values['sced'], help="Required development schedule")
            
        # Calculate EAF (Effort Adjustment Factor)
        input_data['eaf'] = (input_data['rely'] * input_data['data'] * input_data['cplx'] * 
                             input_data['time'] * input_data['stor'] * input_data['virt'] * 
                             input_data['turn'] * input_data['acap'] * input_data['aexp'] * 
                             input_data['pcap'] * input_data['vexp'] * input_data['lexp'] * 
                             input_data['modp'] * input_data['tool'] * input_data['sced'])
    
    elif method == 'FP':
        # 🟢 Bước 1: Giá trị mặc định là 0
        default_values = {
            'AFP': 0.0,
            'EI': 0,
            'EO': 0,
            'EQ': 0,
            'ELF': 0,
            'IFL': 0,
            'PDR_AFP': 0.0,
            'PDR_UFP': 0.0,
            'NPDR_AFP': 0.0,
            'NPDU_UFP': 0.0
            
        }

        # 🟢 Bước 2: Hiển thị file uploader
        st.sidebar.markdown("### (Tùy chọn) Tải dữ liệu từ file Excel")
        uploaded_file = st.sidebar.file_uploader("Chọn file Excel", type=["xlsx"])

        # 🟢 Bước 3: Nếu có file, đọc và cập nhật default_values
        if uploaded_file is not None:
            try:
                df = pd.read_excel(uploaded_file)
                expected_cols = {'AFP', 'EI', 'EO', 'EQ', 'ELF', 'IFL', 'PDR_AFP', 'PDR_UFP', 'NPDR_AFP', 'NPDU_UFP'}
                if not df.empty and expected_cols.issubset(df.columns):
                    default_values['AFP'] = float(df['AFP'].iloc[0])
                    default_values['EI'] = int(df['EI'].iloc[0])
                    default_values['EO'] = int(df['EO'].iloc[0])
                    default_values['EQ'] = int(df['EQ'].iloc[0])
                    default_values['ELF'] = int(df['ELF'].iloc[0])
                    default_values['IFL'] = int(df['IFL'].iloc[0])
                    default_values['PDR_AFP'] = float(df['PDR_AFP'].iloc[0])
                    default_values['PDR_UFP'] = float(df['PDR_UFP'].iloc[0])
                    default_values['NPDR_AFP'] = float(df['NPDR_AFP'].iloc[0])
                    default_values['NPDU_UFP'] = float(df['NPDU_UFP'].iloc[0])
            except Exception as e:
                st.sidebar.error(f"Lỗi khi đọc file: {e}")

        # 🟢 Bước 4: Hiển thị các input với giá trị đã có (0 hoặc từ file)
        st.sidebar.subheader("Function Point Components:")
        col1, col2 = st.sidebar.columns(2)

        with col1:
            input_data['AFP'] = st.number_input('AFP (Adjusted Function Points)', min_value=0.0, value=default_values['AFP'], help="Adjusted Function Points")
            input_data['Input'] = st.number_input('Input Count (EI)', min_value=0, value=default_values['EI'], help="Number of user inputs")
            input_data['Output'] = st.number_input('Output Count (EO)', min_value=0, value=default_values['EO'], help="Number of user outputs")
            input_data['Enquiry'] = st.number_input('Enquiry Count (EQ)', min_value=0, value=default_values['EQ'], help="Number of user enquiries")

        with col2:
            input_data['File'] = st.number_input('File Count (ELF)', min_value=0, value=default_values['ELF'], help="Number of files")
            input_data['Interface'] = st.number_input('Interface Count (IFL)', min_value=0, value=default_values['IFL'], help="Number of external interfaces")
        
        # Productivity factors
        st.sidebar.subheader("Productivity Factors:")
        input_data['PDR_AFP'] = st.sidebar.number_input('PDR_AFP (Productivity Derived from AFP)', min_value=0.0, value=default_values['PDR_AFP'], help="Productivity derived from Adjusted Function Points")
        input_data['PDR_UFP'] = st.sidebar.number_input('PDR_UFP (Productivity Derived from UFP)', min_value=0.0, value=default_values['PDR_UFP'], help="Productivity derived from Unadjusted Function Points")
        input_data['NPDR_AFP'] = st.sidebar.number_input('NPDR_AFP (Non-Productivity Derived from AFP)', min_value=0.0, value=default_values['NPDR_AFP'], help="Non-productivity derived from Adjusted Function Points")
        input_data['NPDU_UFP'] = st.sidebar.number_input('NPDU_UFP (Non-Productivity Derived from UFP)', min_value=0.0, value=default_values['NPDU_UFP'], help="Non-productivity derived from Unadjusted Function Points")

    
    elif method == 'UCP':
        # 🟢 Bước 1: Giá trị mặc định
        default_values = {
            'Simple Actors': 0,
            'Average Actors': 0,
            'Complex Actors': 0,
            'Simple UC': 0,
            'Average UC': 0,
            'Complex UC': 0,
            'TCF': 1.0,
            'ECF': 1.0,
            'Language': 'Java',
            'Methodology': 'Waterfall',
            'ApplicationType': 'Business Application'
        }

        # 🟢 Bước 2: Hiển thị file uploader
        st.sidebar.markdown("### (Tùy chọn) Tải dữ liệu từ file Excel")
        uploaded_file = st.sidebar.file_uploader("Chọn file Excel", type=["xlsx"])

        if uploaded_file is not None:
            try:
                df = pd.read_excel(uploaded_file)
                expected_cols = set(default_values.keys())
                if not df.empty and expected_cols.issubset(df.columns):
                    for key in default_values:
                        default_values[key] = df[key].iloc[0]
            except Exception as e:
                st.sidebar.error(f"Lỗi khi đọc file: {e}")

        # 🟢 Bước 3: Hiển thị các input trong sidebar
        input_data = {}

        st.sidebar.subheader("Actors:")
        input_data['Simple Actors'] = st.sidebar.number_input(
            'Simple Actors', min_value=0, value=int(default_values['Simple Actors']), help="Count of simple actors")
        input_data['Average Actors'] = st.sidebar.number_input(
            'Average Actors', min_value=0, value=int(default_values['Average Actors']), help="Count of average actors")
        input_data['Complex Actors'] = st.sidebar.number_input(
            'Complex Actors', min_value=0, value=int(default_values['Complex Actors']), help="Count of complex actors")

        st.sidebar.subheader("Use Cases:")
        input_data['Simple UC'] = st.sidebar.number_input(
            'Simple Use Cases', min_value=0, value=int(default_values['Simple UC']), help="Count of simple use cases")
        input_data['Average UC'] = st.sidebar.number_input(
            'Average Use Cases', min_value=0, value=int(default_values['Average UC']), help="Count of average use cases")
        input_data['Complex UC'] = st.sidebar.number_input(
            'Complex Use Cases', min_value=0, value=int(default_values['Complex UC']), help="Count of complex use cases")

        st.sidebar.subheader("Complexity Factors:")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            input_data['TCF'] = st.number_input(
                'Technical Complexity Factor (TCF)',
                min_value=0.0, max_value=5.0, value=float(default_values['TCF']),
                step=0.1, format="%.5f", help="Technical complexity factor"
            )
        with col2:
            input_data['ECF'] = st.number_input(
                'Environmental Complexity Factor (ECF)',
                min_value=0.0, max_value=5.0, value=float(default_values['ECF']),
                step=0.1, format="%.5f", help="Environmental complexity factor"
            )

        # 🟢 Bước 4: Tính toán UAW, UUCW, UCP
        input_data['UAW'] = (
            input_data['Simple Actors'] * 1 +
            input_data['Average Actors'] * 2 +
            input_data['Complex Actors'] * 3
        )
        input_data['UUCW'] = (
            input_data['Simple UC'] * 5 +
            input_data['Average UC'] * 10 +
            input_data['Complex UC'] * 15
        )
        input_data['UCP'] = (input_data['UAW'] + input_data['UUCW']) * input_data['TCF'] * input_data['ECF']
        input_data['Real_P20'] = 20  # default

        # 🟢 Bước 5: Các lựa chọn dạng dropdown (selectbox)
        st.sidebar.subheader("Development Environment:")

        language_options = ['Java', 'C#', 'C++', 'Visual Basic', 'Other']
        methodology_options = ['Waterfall', 'Agile', 'Incremental', 'Rapid Application Development', 'Other']
        app_type_options = ['Business Application', 'Real-Time Application', 'Mathematically-Intensive Application', 'Other']

        # Lấy index tương ứng từ giá trị file Excel hoặc dùng index 0 mặc định
        language_index = language_options.index(default_values['Language']) if default_values['Language'] in language_options else 0
        methodology_index = methodology_options.index(default_values['Methodology']) if default_values['Methodology'] in methodology_options else 0
        app_type_index = app_type_options.index(default_values['ApplicationType']) if default_values['ApplicationType'] in app_type_options else 0

        input_data['Language'] = st.sidebar.selectbox('Programming Language', language_options, index=language_index)
        input_data['Methodology'] = st.sidebar.selectbox('Methodology', methodology_options, index=methodology_index)
        input_data['ApplicationType'] = st.sidebar.selectbox('Application Type', app_type_options, index=app_type_index)

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
        # Load feature names used during training
        feature_names = joblib.load(MODEL_DIR / "ucp_feature_names.pkl")
        
        # Create base features DataFrame
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
            'ECF': input_data['ECF']
        }
        
        # Create DataFrame with all expected columns (initialized with 0)
        df = pd.DataFrame(0, index=[0], columns=feature_names)
        
        # Fill in base features
        for col in features:
            if col in df.columns:
                df[col] = features[col]
        
        # Set categorical features based on input
        if f"Language_{input_data['Language']}" in df.columns:
            df[f"Language_{input_data['Language']}"] = 1
            
        if f"Methodology_{input_data['Methodology']}" in df.columns:
            df[f"Methodology_{input_data['Methodology']}"] = 1
            
        if f"ApplicationType_{input_data['ApplicationType']}" in df.columns:
            df[f"ApplicationType_{input_data['ApplicationType']}"] = 1
            
        return df
# Function to predict the effort using the selected model
def predict_all_models(input_data, method):
    """Predict effort using all models for the given method"""
    try:
        models = {
            'linear': joblib.load(MODEL_DIR / f"{method.lower()}_linear_model.pkl"),
            'decision_tree': joblib.load(MODEL_DIR / f"{method.lower()}_decision_tree_model.pkl"),
            'random_forest': joblib.load(MODEL_DIR / f"{method.lower()}_random_forest_model.pkl"),
            'gradient_boosting': joblib.load(MODEL_DIR / f"{method.lower()}_gradient_boosting_model.pkl")
        }
    except FileNotFoundError:
        st.error(f"""
        Models for {method} method not found. Please ensure you have:
        1. Run the training script (Train_Model.py) first
        2. Models are saved in the correct location: {MODEL_DIR}
        3. All required model files exist:
           - {method.lower()}_linear_model.pkl
           - {method.lower()}_decision_tree_model.pkl
           - {method.lower()}_random_forest_model.pkl
           - {method.lower()}_gradient_boosting_model.pkl
        """)
        return None
    
    try:
        scaler = joblib.load(SCALER_DIR / f"{method.lower()}_scaler.pkl")
    except FileNotFoundError:
        st.error(f"Scaler for {method} method not found. Please run the training script first.")
        return None

    features_df = prepare_features(input_data, method)
    
    predictions = {}
    for name, model in models.items():
        # Scale features
        scaled_features = scaler.transform(features_df)
        
        # Make prediction (in log scale)
        log_prediction = model.predict(scaled_features)
        
        # Convert back to original scale
        prediction = np.expm1(log_prediction[0])
        predictions[name] = prediction
    
    return predictions
def compare_models(predictions, method):
    """Compare and visualize all model predictions"""
    # Load comparison metrics from training
    try:
        comparison_df = pd.read_csv(IMG_DIR / f"{method.lower()}_model_comparison.csv", index_col=0)
        
        # Define weights for different metrics
        weights = {
            'MAE': 0.3,
            'RMSE': 0.3,
            'R²': 0.4
        }
        
        # Normalize metrics (lower is better for MAE and RMSE, higher is better for R²)
        normalized_metrics = pd.DataFrame()
        
        # Normalize MAE and RMSE (lower is better)
        for metric in ['MAE', 'RMSE']:
            max_val = comparison_df[metric].max()
            min_val = comparison_df[metric].min()
            normalized_metrics[metric] = 1 - ((comparison_df[metric] - min_val) / (max_val - min_val))
        
        # Normalize R² (higher is better)
        normalized_metrics['R²'] = (comparison_df['R²'] - comparison_df['R²'].min()) / (comparison_df['R²'].max() - comparison_df['R²'].min())
        
        # Calculate weighted score
        weighted_scores = pd.Series(index=comparison_df.index)
        for model in comparison_df.index:
            score = sum(normalized_metrics.loc[model, metric] * weight 
                       for metric, weight in weights.items())
            weighted_scores[model] = score
        
        # Find best model based on weighted score
        best_model = weighted_scores.idxmax()
        best_prediction = predictions[best_model]
        
        # Create visualization
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Bar plot of predictions
        plt.subplot(2, 2, 1)
        plt.bar(predictions.keys(), predictions.values())
        plt.title('Effort Predictions by Model')
        plt.xticks(rotation=45)
        plt.ylabel('Predicted Effort (hours)')
        
        # 2. Model metrics comparison
        plt.subplot(2, 2, 2)
        comparison_df[['MAE', 'RMSE']].plot(kind='bar', ax=plt.gca())
        plt.title('Error Metrics Comparison')
        plt.xticks(rotation=45)
        plt.ylabel('Error')
        
        # 3. R² scores
        plt.subplot(2, 2, 3)
        comparison_df['R²'].plot(kind='bar', color='green')
        plt.title('R² Score Comparison')
        plt.xticks(rotation=45)
        plt.ylabel('R² Score')
        
        # 4. Overall model scores
        plt.subplot(2, 2, 4)
        weighted_scores.plot(kind='bar', color='purple')
        plt.title('Overall Model Scores')
        plt.xticks(rotation=45)
        plt.ylabel('Weighted Score')
        
        plt.tight_layout()
        
        # Print detailed comparison
        print("\nModel Comparison Details:")
        comparison_details = pd.DataFrame({
            'MAE': comparison_df['MAE'],
            'RMSE': comparison_df['RMSE'],
            'R²': comparison_df['R²'],
            'Overall Score': weighted_scores
        })
        print(comparison_details.round(4))
        
        return fig, best_model, best_prediction
        
    except Exception as e:
        st.error(f"Error in model comparison: {e}")
        # Fall back to simple R² comparison if there's an error
        return None, None, None
    
# Function to estimate project metrics (effort, cost, and time)
def estimate_project_metrics(effort):
    hours = effort
    cost = hours * COST_PER_HOUR
    months = hours / HOURS_PER_MONTH
    return hours, cost, months

# Function to convert decimal months to months and days format
def convert_to_months_days(months):
    """Convert decimal months to months and days format"""
    full_months = int(months)  # Phần nguyên là số tháng
    remaining_days = int((months - full_months) * 30)  # Phần thập phân * 30 = số ngày
    
    if full_months == 0:
        if remaining_days == 0:
            return "0 days"
        return f"{remaining_days} days"
    elif remaining_days == 0:
        return f"{full_months} months"
    else:
        return f"{full_months} months {remaining_days} days"


# Display results
def display_results(pred_effort, method):
    # Get predictions from all models
    predictions = predict_all_models(pred_effort, method)
    
    # Compare models and get the best one
    comparison_fig, best_model, best_prediction = compare_models(predictions, method)
    
    # Display all predictions
    st.subheader("Model Predictions")
    
    # Create DataFrame with predictions and reset index to start from 1
    pred_df = pd.DataFrame({
        'No': range(1, len(predictions) + 1),  # Add No column starting from 1
        'Model': predictions.keys(),
        'Predicted Effort (hours)': predictions.values()
    })
    
    # Reset index to remove the default index
    pred_df.set_index('No', inplace=True)
    
    # Display the DataFrame
    st.dataframe(pred_df)
    
    # Display model comparison plot
    st.subheader("Model Comparison")
    st.pyplot(comparison_fig)
    
    # Display best model info
    st.success(f"Best performing model: {best_model.replace('_', ' ').title()}")
    
    # Calculate project metrics using the best model's prediction
    st.subheader("Project Metrics (Based on Best Model)")
    hours, cost, months = estimate_project_metrics(best_prediction)
    
    # Display metrics using columns
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Effort (Hours)", f"{hours:.0f}")
    col2.metric("Cost (USD)", f"${cost:,.2f}")
    col3.metric("Duration", convert_to_months_days(months))
    col4.metric("Team Size", f"{math.ceil(months)} people")
    
    # Create detailed breakdown
    st.subheader("Project Details")
    
    # Calculate phase durations
    requirements_duration = months * 0.1
    design_duration = months * 0.2
    development_duration = months * 0.4
    testing_duration = months * 0.2
    deployment_duration = months * 0.1
    
    details = {
        "Metric": [
            "Estimation Method", 
            "Estimated Effort", 
            "Hourly Rate", 
            "Total Cost", 
            "Project Duration",
            "Team Size",
            "\nPhase Breakdown:",
            "Requirements Phase",
            "Design Phase",
            "Development Phase",
            "Testing Phase",
            "Deployment Phase"
        ],
        "Value": [
            method,
            f"{hours:.0f} hours",
            f"${COST_PER_HOUR:.2f}/hour",
            f"${cost:,.2f}",
            convert_to_months_days(months),
            f"{math.ceil(months)} people",
            "",
            f"{convert_to_months_days(requirements_duration)} ({requirements_duration/months*100:.0f}%)",
            f"{convert_to_months_days(design_duration)} ({design_duration/months*100:.0f}%)",
            f"{convert_to_months_days(development_duration)} ({development_duration/months*100:.0f}%)",
            f"{convert_to_months_days(testing_duration)} ({testing_duration/months*100:.0f}%)",
            f"{convert_to_months_days(deployment_duration)} ({deployment_duration/months*100:.0f}%)"
        ]
    }
    st.table(pd.DataFrame(details))
    
    # Create visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Cost Breakdown Pie Chart
    labels = ['Development', 'Testing', 'Management']
    sizes = [cost * 0.7, cost * 0.2, cost * 0.1]
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Cost Breakdown')
    
    # Timeline Bar Chart
    phases = ['Requirements', 'Design', 'Development', 'Testing', 'Deployment']
    phase_times = [requirements_duration, design_duration, development_duration, 
                  testing_duration, deployment_duration]
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#C2C2F0']
    
    # Create horizontal bar chart
    bars = ax2.barh(phases, phase_times, color=colors)
    ax2.set_xlabel('Duration (months)')
    ax2.set_title('Project Timeline')
    
    # Add duration labels on the bars with just months and percentage
    for i, bar in enumerate(bars):
        width = bar.get_width()
        percentage = phase_times[i]/months*100
        ax2.text(width/2, bar.get_y() + bar.get_height()/2,
                f'{phase_times[i]:.1f}m ({percentage:.0f}%)',
                ha='center', va='center')
    
    plt.tight_layout()
    st.pyplot(fig)

# Export PDF report
def export_pdf_report(input_data, pred_effort, method):
    # Create PDF object
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Software Project Effort Estimation Report', 0, 1, 'C')
    pdf.ln(10)
    
    # Input Parameters Table
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Input Parameters', 0, 1, 'L')
    
    # Table header
    pdf.set_font('Arial', 'B', 12)
    col_width = [100, 90]  # Width for parameter and value columns
    
    # Table headers with borders
    pdf.cell(col_width[0], 10, 'Parameter', 1, 0, 'C')
    pdf.cell(col_width[1], 10, 'Value', 1, 1, 'C')
    
    # Table content
    pdf.set_font('Arial', '', 11)
    if method == 'LOC':
        params = [
            ('KLOC', f"{input_data['equivphyskloc']:.1f}"),
            ('Development Mode', "Organic" if input_data['mode_organic'] else 
                               "Semi-detached" if input_data['mode_semidetached'] else 
                               "Embedded"),
            ('Required Reliability', f"{input_data['rely']:.2f}"),
            ('Database Size', f"{input_data['data']:.2f}"),
            ('Product Complexity', f"{input_data['cplx']:.2f}"),
            ('Time Constraint', f"{input_data['time']:.2f}"),
            ('Storage Constraint', f"{input_data['stor']:.2f}"),
            ('Platform Volatility', f"{input_data['virt']:.2f}"),
            ('Turnaround Time', f"{input_data['turn']:.2f}"),
            ('Analyst Capability', f"{input_data['acap']:.2f}"),
            ('Analyst Experience', f"{input_data['aexp']:.2f}"),
            ('Programmer Capability', f"{input_data['pcap']:.2f}"),
            ('VM Experience', f"{input_data['vexp']:.2f}"),
            ('Language Experience', f"{input_data['lexp']:.2f}"),
            ('Modern Practices', f"{input_data['modp']:.2f}"),
            ('Software Tools', f"{input_data['tool']:.2f}"),
            ('Schedule Constraint', f"{input_data['sced']:.2f}")
        ]
    
    elif method == 'FP':
        params = [
            ('AFP', f"{input_data['AFP']:.1f}"),
            ('Input Count', str(input_data['Input'])),
            ('Output Count', str(input_data['Output'])),
            ('Enquiry Count', str(input_data['Enquiry'])),
            ('File Count', str(input_data['File'])),
            ('Interface Count', str(input_data['Interface'])),
            ('PDR_AFP', f"{input_data['PDR_AFP']:.2f}"),
            ('PDR_UFP', f"{input_data['PDR_UFP']:.2f}"),
            ('NPDR_AFP', f"{input_data['NPDR_AFP']:.2f}"),
            ('NPDU_UFP', f"{input_data['NPDU_UFP']:.2f}")
        ]
    
    elif method == 'UCP':
        params = [
            ('Simple Actors', str(input_data['Simple Actors'])),
            ('Average Actors', str(input_data['Average Actors'])),
            ('Complex Actors', str(input_data['Complex Actors'])),
            ('Simple Use Cases', str(input_data['Simple UC'])),
            ('Average Use Cases', str(input_data['Average UC'])),
            ('Complex Use Cases', str(input_data['Complex UC'])),
            ('TCF', f"{input_data['TCF']:.5f}"),
            ('ECF', f"{input_data['ECF']:.5f}"),
            ('Language', input_data['Language']),
            ('Methodology', input_data['Methodology']),
            ('Application Type', input_data['ApplicationType'])
        ]

    # Print table rows with borders
    for param, value in params:
        pdf.cell(col_width[0], 10, param, 1, 0, 'L')
        pdf.cell(col_width[1], 10, str(value), 1, 1, 'L')
    
    # Results section
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Estimation Results', 0, 1, 'L')
    
    # Calculate metrics
    hours, cost, months = estimate_project_metrics(pred_effort)
    
    # Results table
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(col_width[0], 10, 'Metric', 1, 0, 'C')
    pdf.cell(col_width[1], 10, 'Value', 1, 1, 'C')
    
    pdf.set_font('Arial', '', 11)
    results = [
        ('Total Effort', f"{hours:.0f} hours"),
        ('Total Cost', f"${cost:,.2f}"),
        ('Project Duration', convert_to_months_days(months)),
        ('Team Size', f"{math.ceil(months)} people"),
    ]

    for metric, value in results:
        pdf.cell(col_width[0], 10, metric, 1, 0, 'L')
        pdf.cell(col_width[1], 10, value, 1, 1, 'L')
    
    # Phase breakdown table
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Phase Breakdown', 0, 1, 'L')
    
    pdf.cell(col_width[0], 10, 'Phase', 1, 0, 'C')
    pdf.cell(col_width[1], 10, 'Duration', 1, 1, 'C')
    
    pdf.set_font('Arial', '', 11)
    phases = {
        'Requirements': months * 0.1,
        'Design': months * 0.2,
        'Development': months * 0.4,
        'Testing': months * 0.2,
        'Deployment': months * 0.1
    }
    
    for phase, duration in phases.items():
        percentage = (duration/months) * 100
        pdf.cell(col_width[0], 10, phase, 1, 0, 'L')
        pdf.cell(col_width[1], 10, f"{convert_to_months_days(duration)} ({percentage:.0f}%)", 1, 1, 'L')
    
    # Cost breakdown table
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Cost Breakdown', 0, 1, 'L')
    
    pdf.cell(col_width[0], 10, 'Category', 1, 0, 'C')
    pdf.cell(col_width[1], 10, 'Amount', 1, 1, 'C')
    
    pdf.set_font('Arial', '', 11)
    costs = [
        ('Development', (cost * 0.7, 70)),
        ('Testing', (cost * 0.2, 20)),
        ('Management', (cost * 0.1, 10))
    ]
    
    for category, (amount, percent) in costs:
        pdf.cell(col_width[0], 10, category, 1, 0, 'L')
        pdf.cell(col_width[1], 10, f"${amount:,.2f} ({percent}%)", 1, 1, 'L')
    
    # Add charts
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
        # Create figure with charts
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Cost Breakdown Pie Chart
        labels = ['Development', 'Testing', 'Management']
        sizes = [cost * 0.7, cost * 0.2, cost * 0.1]
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Cost Breakdown')
        
        # Timeline Bar Chart
        phase_names = list(phases.keys())
        phase_durations = list(phases.values())
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#C2C2F0']
        
        bars = ax2.barh(phase_names, phase_durations, color=colors)
        ax2.set_xlabel('Duration (months)')
        ax2.set_title('Project Timeline')
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            percentage = phase_durations[i]/months*100
            ax2.text(width/2, bar.get_y() + bar.get_height()/2,
                    f'{phase_durations[i]:.1f}m ({percentage:.0f}%)',
                    ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig(tmp_file.name)
        plt.close()
        
        # Add charts to PDF
        pdf.add_page()
        pdf.image(tmp_file.name, x=10, y=30, w=190)
    
    return pdf

# Main function
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
            # Get predictions from all models
            predictions = predict_all_models(data, method)
            
            # Compare models and get the best prediction
            comparison_fig, best_model, best_prediction = compare_models(predictions, method)
            
            # Display results
            display_results(data, method)
            st.success(f"Estimation completed using {method} method!")
            
            # Generate PDF report using best prediction
            pdf = export_pdf_report(data, best_prediction, method)
            
            # Save PDF to bytes
            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            
            # Download button for PDF
            st.download_button(
                label="Export Results (PDF)",
                data=pdf_bytes,
                file_name=f"effort_estimation_{method.lower()}.pdf",
                mime="application/pdf"
            )

if __name__ == "__main__":
    main()