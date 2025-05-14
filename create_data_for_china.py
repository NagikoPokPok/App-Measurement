import pandas as pd
import numpy as np

# Hàm sinh dữ liệu giả với tính toán Effort
def generate_synthetic_data_with_effort(df, num_samples):
    df_no_id = df.drop("ID", axis=1)
    synthetic_data = {}

    # Sinh dữ liệu giả cho mỗi cột
    for col in df_no_id.columns:
        mean = df_no_id[col].mean()
        std = df_no_id[col].std()
        synthetic_col = np.random.normal(loc=mean, scale=std, size=num_samples)
        synthetic_data[col] = np.maximum(0, synthetic_col.round().astype(int))

    # Cập nhật công thức tính Effort
    if 'N_effort' in df_no_id.columns and 'AFP' in df_no_id.columns and 'Input' in df_no_id.columns and 'Output' in df_no_id.columns:
        # Công thức tính Effort giả định
        synthetic_effort = synthetic_data['N_effort'] * (synthetic_data['AFP'] + synthetic_data['Input'] + synthetic_data['Output'])
        synthetic_data['Effort'] = synthetic_effort.astype(int)

    # Tạo DataFrame từ dữ liệu giả
    synthetic_df = pd.DataFrame(synthetic_data)
    synthetic_df.insert(0, "ID", range(df["ID"].max() + 1, df["ID"].max() + 1 + num_samples))
    
    return synthetic_df

# Hàm chính để sinh và lưu dữ liệu vào file CSV
def generate_and_save_data(df, total_samples=10000, output_path="china_combined_10000.csv"):
    num_to_generate = total_samples - len(df)
    synthetic_df = generate_synthetic_data_with_effort(df, num_to_generate)

    # Gộp dữ liệu gốc và dữ liệu giả
    combined_df = pd.concat([df, synthetic_df], ignore_index=True)

    # Lưu dữ liệu vào file CSV
    combined_df.to_csv(output_path, index=False)
    print(f"Dữ liệu đã được lưu vào: {output_path}")

    # Trả về kết quả dưới dạng DataFrame
    return combined_df

# Đọc dữ liệu gốc
df_original = pd.read_csv('dataset/china.csv')

# Sinh dữ liệu và lưu vào file
combined_data = generate_and_save_data(df_original, total_samples=10000, output_path="dataset/china_combined_10000_2.csv")
