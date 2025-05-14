import pandas as pd
import numpy as np

# === BƯỚC 1: Đọc dữ liệu gốc ===
df_ucp = pd.read_csv("dataset/UCP_Dataset.csv")

# === BƯỚC 2: Tính Productivity Factor trung bình từ dữ liệu gốc ===
df_ucp["UCP_Calculated"] = (df_ucp["UAW"] + df_ucp["UUCW"]) * df_ucp["TCF"] * df_ucp["ECF"]
df_ucp["Productivity_Factor"] = df_ucp["Real_Effort_Person_Hours"] / df_ucp["UCP_Calculated"]
pf_mean = df_ucp["Productivity_Factor"].mean()

# === BƯỚC 3: Lấy danh sách duy nhất các giá trị chuỗi để sinh dữ liệu giả hợp lệ ===
languages = df_ucp["Language"].dropna().unique()
methodologies = df_ucp["Methodology"].dropna().unique()
application_types = df_ucp["ApplicationType"].dropna().unique()

# === BƯỚC 4: Sinh dữ liệu mới với số nguyên và các trường chuỗi ===
def generate_ucp_synthetic_data(df, num_samples, pf):
    cols = ["UAW", "UUCW", "TCF", "ECF"]
    synthetic = {}

    for col in cols:
        mean = df[col].mean()
        std = df[col].std()
        synthetic[col] = np.maximum(0, np.random.normal(mean, std, num_samples).round().astype(int))

    synthetic_df = pd.DataFrame(synthetic)

    # Tính effort (số nguyên)
    synthetic_df["Real_Effort_Person_Hours"] = (
        (synthetic_df["UAW"] + synthetic_df["UUCW"]) *
        synthetic_df["TCF"] *
        synthetic_df["ECF"] *
        pf
    ).round().astype(int)

    # Sinh các cột chuỗi
    synthetic_df["Language"] = np.random.choice(languages, num_samples)
    synthetic_df["Methodology"] = np.random.choice(methodologies, num_samples)
    synthetic_df["ApplicationType"] = np.random.choice(application_types, num_samples)

    # Sinh cột Real_P20 với giá trị tương tự từ dữ liệu gốc
    synthetic_df["Real_P20"] = np.random.choice(df_ucp["Real_P20"].dropna(), num_samples)

    # Bổ sung cột còn thiếu để khớp với df_ucp
    missing_cols = [col for col in df_ucp.columns if col not in synthetic_df.columns]
    for col in missing_cols:
        if df_ucp[col].dtype == object:
            synthetic_df[col] = ""
        else:
            synthetic_df[col] = 0

    # Đảm bảo cột đúng thứ tự
    synthetic_df = synthetic_df[df_ucp.columns]
    
    return synthetic_df

# === BƯỚC 5: Sinh thêm để đủ 10,000 dòng ===
num_existing = len(df_ucp)
num_to_generate = 10000 - num_existing
df_synthetic = generate_ucp_synthetic_data(df_ucp, num_to_generate, pf_mean)

# === BƯỚC 6: Xóa cột Project_No nếu đã tồn tại trong dữ liệu gốc và sinh thêm ===
if 'Project_No' in df_synthetic.columns:
    df_synthetic = df_synthetic.drop(columns=['Project_No'])

# === BƯỚC 7: Gộp lại dữ liệu gốc và dữ liệu sinh thêm ===
df_combined = pd.concat([df_ucp, df_synthetic], ignore_index=True)

# === BƯỚC 8: Xóa cột Project_No nếu tồn tại trong df_combined và thêm lại từ 1 đến 10.000 ===
if 'Project_No' in df_combined.columns:
    df_combined = df_combined.drop(columns=['Project_No'])

# Đánh lại Project_No từ 1 đến 10.000
df_combined.insert(0, "Project_No", range(1, len(df_combined) + 1))

# === BƯỚC 9: Xóa cột UCP_Calculated và Productivity_Factor ===
df_combined = df_combined.drop(columns=["UCP_Calculated", "Productivity_Factor"])

# === BƯỚC 10: Lưu ra file CSV ===
df_combined.to_csv("dataset/UCP_Combined_10000.csv", index=False)

print("✅ File hoàn chỉnh đã được tạo tại: dataset/UCP_Combined_10000.csv")
