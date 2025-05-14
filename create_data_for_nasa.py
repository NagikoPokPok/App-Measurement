import pandas as pd
import numpy as np

# Đọc lại dữ liệu từ file
df_cocomo81 = pd.read_csv("dataset/nasa93.arff.csv", delimiter=";")

# === BƯỚC 1: Bảng rating_map để chuyển giá trị chuỗi thành số ===
rating_map = {
    'vl': 0.5,   # Very Low
    'l': 0.7,    # Low
    'n': 1.0,    # Nominal
    'h': 1.15,   # High
    'vh': 1.4,   # Very High
    'xh': 1.65   # Extra High
}

# Bảng ngược lại để chuyển từ số về chuỗi
reverse_rating_map = {v: k for k, v in rating_map.items()}

# === BƯỚC 2: Hàm chuyển chuỗi thành số dựa trên rating_map ===
def convert_to_numeric(val):
    return rating_map.get(val, np.nan)  # Trả về NaN nếu giá trị không hợp lệ

# Hàm để chuyển từ số về chuỗi (ngược lại)
def convert_to_string(val):
    return reverse_rating_map.get(val, val)  # Trả về giá trị gốc nếu không có trong rating_map

# Áp dụng chuyển đổi cho các cột chuỗi thành số trong quá trình tính Effort
cols_to_convert = ['rely', 'data', 'cplx', 'time', 'stor', 'virt', 'turn', 'acap', 'aexp', 'pcap', 'vexp', 'lexp', 'modp', 'tool', 'sced']
for col in cols_to_convert:
    df_cocomo81[col] = df_cocomo81[col].apply(convert_to_numeric)

# === BƯỚC 3: Tính Effort từ các yếu tố có sẵn ===
# Các hằng số cho mô hình semidetached
a = 3.0
b = 1.12

# Tính EAF (Effort Adjustment Factor)
def calculate_eaf(row):
    eaf = 1
    # Nhân các yếu tố như rely, data, cplx...
    for col in cols_to_convert:
        eaf *= row[col]
    return eaf

# Tính Effort
def calculate_effort(row):
    kloc = row['equivphyskloc'] / 1000  # Chuyển loc thành KLOC
    eaf = calculate_eaf(row)
    # Tính Effort từ công thức COCOMO
    return a * (kloc ** b) * eaf

# Tính Effort cho từng dòng dữ liệu
df_cocomo81['Effort'] = df_cocomo81.apply(calculate_effort, axis=1)

# === BƯỚC 4: Sinh thêm dữ liệu để đủ 10,000 dòng ===
num_existing = len(df_cocomo81)
num_to_generate = 10000 - num_existing

# Tạo dữ liệu giả cho các cột cần thiết
def generate_synthetic_data(df, num_samples):
    synthetic_data = {}
    
    for col in df.columns:
        if df[col].dtype == object:  # Nếu là chuỗi, sinh ngẫu nhiên
            synthetic_data[col] = np.random.choice(df[col].dropna().unique(), num_samples)
        elif df[col].dtype in [np.int64, np.float64]:  # Nếu là số, sinh ngẫu nhiên
            synthetic_data[col] = np.random.choice(df[col].dropna().unique(), num_samples)
    
    synthetic_df = pd.DataFrame(synthetic_data)
    return synthetic_df

# Sinh dữ liệu giả và tính Effort cho nó
df_synthetic = generate_synthetic_data(df_cocomo81, num_to_generate)

# Gộp lại dữ liệu gốc và dữ liệu sinh thêm
df_combined = pd.concat([df_cocomo81, df_synthetic], ignore_index=True)

# === BƯỚC 5: Chuyển các giá trị số trở lại chuỗi trước khi lưu ===
for col in cols_to_convert:
    df_combined[col] = df_combined[col].apply(convert_to_string)

# === BƯỚC 6: Lưu kết quả ra file CSV ===
df_combined.drop(columns=['Effort'], inplace=True)  # Loại bỏ cột 'Effort' trước khi lưu
df_combined.to_csv("dataset/nasa93_Combined_10000_test.csv", index=False)

print("✅ File 10,000 dòng đã được tạo thành công tại: nasa93_Combined_10000_COCOMO.csv")
