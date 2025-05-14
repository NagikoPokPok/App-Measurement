import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def augment_data(input_file, output_file, total_rows=10000):
    # Đọc dữ liệu từ file
    china_data = pd.read_csv(input_file)

    # Lấy các cột đầu vào và cột "Effort" từ dữ liệu gốc
    X = china_data.drop(['ID', 'Effort', 'N_effort'], axis=1)  # Loại bỏ cột "ID", "Effort" và "N_effort"
    y = china_data['Effort']

    # Huấn luyện mô hình hồi quy tuyến tính
    model = LinearRegression()
    model.fit(X, y)

    # Tạo thêm dữ liệu giả (synthetic data) vào dữ liệu gốc
    additional_data_count = total_rows - china_data.shape[0]  # Tính số dòng dữ liệu cần thêm
    synthetic_data = X.sample(n=additional_data_count, replace=True, random_state=42)  # Lấy mẫu từ dữ liệu hiện tại
    synthetic_predicted_effort = model.predict(synthetic_data)

    # Tạo thêm cột "Effort" giả và "N_effort" giả cho dữ liệu bổ sung
    synthetic_data['Effort'] = synthetic_predicted_effort
    synthetic_data['N_effort'] = synthetic_data['Effort'] * 1.2  # Ví dụ, N_effort là Effort * 1.2

    # Loại bỏ các dòng có giá trị "Effort" âm
    synthetic_data = synthetic_data[synthetic_data['Effort'] >= 0]

    # Kiểm tra số dòng dữ liệu sau khi loại bỏ
    remaining_rows = total_rows - synthetic_data.shape[0]

    # Nếu có ít dòng hơn mong đợi, thêm dữ liệu mới để bù vào
    while synthetic_data.shape[0] < total_rows:
        missing_rows_count = total_rows - synthetic_data.shape[0]
        new_synthetic_data = X.sample(n=missing_rows_count, replace=True, random_state=42)
        new_predicted_effort = model.predict(new_synthetic_data)
        new_synthetic_data['Effort'] = new_predicted_effort
        new_synthetic_data['N_effort'] = new_synthetic_data['Effort'] * 1.2
        new_synthetic_data = new_synthetic_data[new_synthetic_data['Effort'] >= 0]
        synthetic_data = pd.concat([synthetic_data, new_synthetic_data], ignore_index=True)

    # Tạo ID mới cho các dòng dữ liệu giả
    new_ids = range(china_data['ID'].max() + 1, china_data['ID'].max() + 1 + synthetic_data.shape[0])
    synthetic_data['ID'] = new_ids

    # Kết hợp dữ liệu gốc và dữ liệu giả
    combined_data = pd.concat([china_data, synthetic_data], ignore_index=True)

    # Convert effort to integers by rounding and removing decimal places
    combined_data['Effort'] = combined_data['Effort'].round().astype(int)
    combined_data['N_effort'] = combined_data['N_effort'].round().astype(int)

    # Lưu lại file dữ liệu mới
    combined_data.to_csv(output_file, index=False)
    print(f"File dữ liệu mới đã được lưu tại {output_file}")

# Gọi hàm augment_data với các tham số input và output của bạn
augment_data('dataset/china.csv', 'dataset/augmented_china_data.csv', total_rows=10000)
