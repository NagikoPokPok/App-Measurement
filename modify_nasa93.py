import pandas as pd

def convert_nasa93_with_named_columns():
    # Bước 1: Đọc và xử lý file cost_drivers.xlsx
    cost_drivers_df = pd.read_excel(
        "dataset/cost_drivers.xlsx", 
        sheet_name="cost driver of COCOMO 1",  # Đọc sheet "cost driver of COCOMO 1"
        header=0 # Dòng đầu tiên chứa tên cột
    )

    # Chuyển cột "Cost Driver" sang kiểu chuỗi và chuẩn hóa
    cost_drivers_df["Cost Driver"] = (
        cost_drivers_df["Cost Driver"]
        .astype(str) # Chuyển sang kiểu chuỗi (đảm bảo giá trị trong cột là chuỗi (kể cả số hoặc NaN))
        .str.replace("nan", "") # Thay giá trị "nan" thành chuỗi rỗng
        .str.strip() # Loại bỏ khoảng trắng ở đầu và cuối chuỗi
        .str.lower() # Chuyển sang chữ thường tất cả các ký tự
    )

    # Tạo từ điển ánh xạ cost driver
    cost_drivers_dict = {}
    for _, row in cost_drivers_df.iterrows(): # Duyệt qua TỪNG HÀNG trong DataFrame, trả về một cặp (index, row)
        # Dấu _: Bỏ qua chỉ số hàng (index), vì chúng ta không cần sử dụng nó.
        #row: Là một Series chứa dữ liệu của hàng hiện tại.

        driver = row["Cost Driver"] # Lấy tên cost driver (vd: "rely", "data", ...)
        if not driver: # Kiểm tra nếu driver là chuỗi rỗng ("") hoặc None.
            continue
        
        levels = { # Tạo dictionary với các mức độ (vl, l, n, h, vh, xh) và giá trị tương ứng
            # pd.notna(): Kiểm tra giá trị có phải NaN không, nếu có thì gán None
            "vl": row["vl"] if pd.notna(row["vl"]) else None,
            "l": row["l"] if pd.notna(row["l"]) else None,
            "n": row["n"] if pd.notna(row["n"]) else None,
            "h": row["h"] if pd.notna(row["h"]) else None,
            "vh": row["vh"] if pd.notna(row["vh"]) else None,
            "xh": row["xh"] if pd.notna(row["xh"]) else None,
        }

        # Dùng Dictionary Comprehension để lọc ra các cặp key: value từ levels mà value không phải là None
        cost_drivers_dict[driver] = {k: v for k, v in levels.items() if v is not None}
        ''' 
        Example: Dữ liệu sau khi bỏ đi những giá trị None
        cost_drivers_dict[] =
        {
            "rely": {
                "vl": 0.75,
                "l": 0.88,
                "n": 1,
                "h": 1.15,
                "vh": 1.4
            },
            "data": {
                "l": 0.94,
                "n": 1,
                "h": 1.08,
                "vh": 1.16
            }
        } 
        '''

    # Bước 2: Đọc file nasa93.xlsx
    nasa93_df = pd.read_excel("dataset/nasa93.xlsx", sheet_name="nasa93")

    # Danh sách tên cột cần chuyển đổi (từ 'rely' đến 'sced')
    columns_to_convert = [
        "rely", "data", "cplx", "time", "stor", 
        "virt", "turn", "acap", "aexp", "pcap", 
        "vexp", "lexp", "modp", "tool", "sced"
    ]

    # Kiểm tra các cột có tồn tại không
    missing_columns = [col for col in columns_to_convert if col not in nasa93_df.columns]
    if missing_columns:
        raise ValueError(f"Các cột sau không tồn tại trong file NASA93: {missing_columns}")

    # Bước 3: Thay thế giá trị chữ bằng số
    for col in columns_to_convert:
        driver_name = col.strip().lower() # Chuẩn hóa tên cột
        if driver_name not in cost_drivers_dict:
            print(f"Warning: Cost driver '{driver_name}' không tồn tại trong từ điển!")
            continue
        
        # Ánh xạ giá trị
        nasa93_df[col] = (
            nasa93_df[col] # cần phải có, nếu không các lệnh phía sau sẽ không hợp lệ
            .astype(str) 
            .str.strip()
            .str.lower()
            .map(cost_drivers_dict[driver_name]) # Ánh xạ giá trị
            # .map(): Thay thế từng giá trị trong cột bằng giá trị số tương ứng từ từ điển
        )

    # Bước 4: Xuất file mới
    output_path = "dataset/nasa93_converted.xlsx"
    nasa93_df.to_excel(output_path, index=False) 
    print(f"Chuyển đổi hoàn tất! File đã được lưu tại: {output_path}")

if __name__ == "__main__":
    convert_nasa93_with_named_columns()