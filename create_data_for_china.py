import numpy as np

# Số lượng dữ liệu giả cần sinh
num_samples = 5000

# Khởi tạo DataFrame mới
synthetic_data = pd.DataFrame()

# Sinh dữ liệu cho các cột số
for column in ['Input', 'Output', 'Enquiry', 'File', 'Interface']:
    mean = df[column].mean()
    std = df[column].std()
    synthetic_data[column] = np.random.normal(mean, std, num_samples).round().astype(int)

# Tính toán Effort dựa trên mô hình hồi quy tuyến tính
from sklearn.linear_model import LinearRegression

# Huấn luyện mô hình trên dữ liệu gốc
X_train = df[['Input', 'Output', 'Enquiry', 'File', 'Interface']]
y_train = df['Effort']
model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán Effort cho dữ liệu giả
synthetic_data['Effort'] = model.predict(synthetic_data).round().astype(int)

# Kết hợp dữ liệu gốc và dữ liệu giả
combined_data = pd.concat([df, synthetic_data], ignore_index=True)

# Lưu dữ liệu kết hợp
combined_data.to_csv("china_expanded.csv", index=False)
