import pandas as pd
import numpy as np

# Load the original NASA93 dataset
nasa93 = pd.read_csv('dataset/nasa93.arff.csv', sep=';')

# Define the mapping for cost drivers
rating_map = {
    'vl': 0.5,   # Very Low
    'l': 0.7,    # Low
    'n': 1.0,    # Nominal
    'h': 1.15,   # High
    'vh': 1.4,   # Very High
    'xh': 1.65   # Extra High
}

# Select relevant columns and preprocess
cost_drivers = ['rely', 'data', 'cplx', 'time', 'stor', 'virt', 'turn', 
                'acap', 'aexp', 'pcap', 'vexp', 'lexp', 'modp', 'tool', 'sced']

# Map categorical values to numerical values
data = nasa93.copy()
for driver in cost_drivers:
    if driver in data.columns:
        data[driver] = data[driver].map(rating_map)

# Define the COCOMO II calculation function
def calculate_cocomo_ii_effort(row):
    a = 2.94  # Constant
    b = 0.91  # Size exponent
    kloc = row['equivphyskloc']  # Project size in KLOC
    
    # Calculate Effort Multiplier (EM) based on cost drivers
    em_factors = ['rely', 'data', 'cplx', 'time', 'stor', 'virt', 'turn', 
                  'acap', 'aexp', 'pcap', 'vexp', 'lexp', 'modp', 'tool', 'sced']
    em = 1.0
    for factor in em_factors:
        if factor in row and not pd.isna(row[factor]):
            em *= row[factor]
    
    # Handle project mode (complexity adjustment)
    mode = row['mode'].lower() if 'mode' in row else ''
    if mode == 'embedded':
        complexity_adjustment = 1.2
    elif mode == 'organic':
        complexity_adjustment = 0.8
    else:  # semidetached or other values
        complexity_adjustment = 1.0
    
    # Calculate effort
    effort = a * (kloc ** b) * em * complexity_adjustment
    return np.log1p(effort)

# Function to generate synthetic data with realistic variation
def generate_synthetic_data_with_realistic_values(original_data, num_rows):
    synthetic_data = []
    
    # Get the min and max values for each cost driver from the original data
    cost_driver_ranges = {}
    for driver in cost_drivers:
        min_value = original_data[driver].min()
        max_value = original_data[driver].max()
        cost_driver_ranges[driver] = (min_value, max_value)
    
    for _ in range(num_rows):
        # Randomly select a row from the original data
        row = original_data.sample(1).iloc[0]
        
        # Add realistic variation to cost drivers
        for driver in cost_drivers:
            if driver in row:
                min_value, max_value = cost_driver_ranges[driver]
                row[driver] = np.random.uniform(min_value, max_value)  # Random value within the range

        # Randomly vary 'equivphyskloc' (size of the project)
        row['equivphyskloc'] = np.random.uniform(row['equivphyskloc'] * 0.95, row['equivphyskloc'] * 1.05)
        
        # Calculate the 'act_effort' for the synthetic row based on COCOMO II
        row['act_effort'] = calculate_cocomo_ii_effort(row)
        
        # Add this synthetic row to the list
        synthetic_data.append(row)
    
    # Convert the list of rows into a DataFrame
    synthetic_df = pd.DataFrame(synthetic_data)
    
    return synthetic_df

# Generate the synthetic data to reach 10,000 rows
num_new_rows = 10000 - len(data)
synthetic_data = generate_synthetic_data_with_realistic_values(data, num_new_rows)

# Combine the original data with the new synthetic data
augmented_data = pd.concat([data, synthetic_data])

# Convert 'act_effort' to integers by rounding and removing decimal places
augmented_data['act_effort'] = augmented_data['act_effort'].round().astype(int)

# Save the updated dataset to a new file
augmented_data = augmented_data.drop_duplicates()
augmented_data.to_csv('dataset/augmented_nasa93_10000_realistic.csv', index=False)

print(f"Data successfully generated. The dataset now contains {len(augmented_data)} rows.")
