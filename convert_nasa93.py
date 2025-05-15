import pandas as pd

# Bước 1: Đọc và xử lý file cost_drivers.xlsx
cost_drivers_df = pd.read_excel(
    './dataset/cost_drivers.xlsx', 
    sheet_name="cost driver of COCOMO 1", 
    header=0
)

# Chuyển cột "Cost Driver" sang kiểu chuỗi, xử lý NaN và chuẩn hóa
cost_drivers_df["Cost Driver"] = (
    cost_drivers_df["Cost Driver"]
    .astype(str)  # Chuyển mọi giá trị sang string (kể cả NaN → "nan")
    .str.replace("nan", "")  # Thay "nan" thành chuỗi rỗng
    .str.strip()  # Loại bỏ khoảng trắng thừa
    .str.lower()  # Chuyển về chữ thường
)

# Tạo từ điển ánh xạ cost driver
cost_drivers_dict = {}
for _, row in cost_drivers_df.iterrows():
    driver = row["Cost Driver"]
    if not driver:  # Bỏ qua hàng trống
        continue
    # Xử lý các mức độ (vl, l, n, h, vh, xh)
    levels = {
        "vl": row["vl"] if pd.notna(row["vl"]) else None,
        "l": row["l"] if pd.notna(row["l"]) else None,
        "n": row["n"] if pd.notna(row["n"]) else None,
        "h": row["h"] if pd.notna(row["h"]) else None,
        "vh": row["vh"] if pd.notna(row["vh"]) else None,
        "xh": row["xh"] if pd.notna(row["xh"]) else None,
    }
    # Lọc bỏ giá trị None và thêm vào từ điển
    cost_drivers_dict[driver] = {k: v for k, v in levels.items() if v is not None}

# Bước 2: Đọc file nasa93.xlsx
nasa93_df = pd.read_excel("./dataset/nasa93.xlsx", sheet_name="nasa93")

# Xác định các cột cần chuyển đổi (H đến V, tương ứng các cost driver)
columns_to_convert = nasa93_df.columns[7:22]  # Từ cột 'rely' đến 'sced'

# Bước 3: Thay thế giá trị chữ bằng số
for col in columns_to_convert:
    # Chuẩn hóa tên cột để khớp với key trong từ điển
    driver_name = col.strip().lower()
    if driver_name not in cost_drivers_dict:
        print(f"Warning: Cost driver '{driver_name}' không tồn tại trong từ điển!")
        continue
    
    # Ánh xạ giá trị
    nasa93_df[col] = (
        nasa93_df[col]
        .astype(str)  # Chuyển sang string để xử lý
        .str.strip()  # Loại bỏ khoảng trắng
        .str.lower()  # Chuyển về chữ thường
        .map(cost_drivers_dict[driver_name])  # Ánh xạ dựa trên từ điển
    )
    
    # Thay thế giá trị không tồn tại bằng NaN (hoặc giá trị mặc định)
    nasa93_df[col] = nasa93_df[col].replace("", None)

# Bước 4: Xuất file mới
nasa93_df.to_excel("./dataset/nasa93_converted.xlsx", index=False)
print("Hoàn thành chuyển đổi!")