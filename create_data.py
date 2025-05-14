import pandas as pd
import numpy as np

# Đọc dữ liệu gốc
df_original = pd.read_csv("dataset/china.csv")

# Hàm sinh dữ liệu giả
def generate_synthetic_data(df, num_samples):
    df_no_id = df.drop("ID", axis=1)
    synthetic_data = {}

    for col in df_no_id.columns:
        mean = df_no_id[col].mean()
        std = df_no_id[col].std()
        synthetic_col = np.random.normal(loc=mean, scale=std, size=num_samples)
        synthetic_data[col] = np.maximum(0, synthetic_col.round().astype(int))

    synthetic_df = pd.DataFrame(synthetic_data)
    synthetic_df.insert(0, "ID", range(df["ID"].max() + 1, df["ID"].max() + 1 + num_samples))
    return synthetic_df

# Sinh thêm 9,500 dòng để tổng cộng là 10,000 dòng
num_to_generate = 10000 - len(df_original)
synthetic_df = generate_synthetic_data(df_original, num_to_generate)

# Gộp dữ liệu gốc và mới
combined_df = pd.concat([df_original, synthetic_df], ignore_index=True)

# Xuất ra file CSV
combined_df.to_csv("dataset/china_combined_10000.csv", index=False)

print(f"Tổng số dòng: {len(combined_df)}")
print(combined_df.head())
