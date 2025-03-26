import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Đọc dữ liệu
print("Bước 1: Đọc dữ liệu")
data = pd.read_csv('dataset/software_effort_data.csv', sep=',')
print(f"Số lượng mẫu: {data.shape[0]}")
print(f"Các đặc trưng: {', '.join(data.columns)}")
print("\nThống kê mô tả dữ liệu:")
print(data.describe())

# 2. Tiền xử lý dữ liệu
print("\nBước 2: Tiền xử lý dữ liệu")

# 2.1 Chọn đặc trưng và mục tiêu
X = data[['LOC', 'FP', 'UCP']]
y = data['Effort']

# 2.2 Kiểm tra và xử lý giá trị thiếu
missing_values = X.isnull().sum()
print(f"Số lượng giá trị thiếu trong mỗi đặc trưng:\n{missing_values}")

# Loại bỏ giá trị thiếu
X = X.dropna()
y = y[X.index]
print(f"Số lượng mẫu sau khi xử lý dữ liệu thiếu: {X.shape[0]}")

# 2.3 Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
print("\nDữ liệu sau khi chuẩn hóa (5 dòng đầu):")
print(X_scaled_df.head())

# 3. Chia dữ liệu
print("\nBước 3: Chia dữ liệu thành tập huấn luyện và kiểm tra")
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)
print(f"Số lượng mẫu huấn luyện: {X_train.shape[0]}")
print(f"Số lượng mẫu kiểm tra: {X_test.shape[0]}")

# 4. Huấn luyện mô hình Machine Learning
print("\nBước 4: Huấn luyện các mô hình Machine Learning")

# 4.1 Linear Regression
print("Huấn luyện mô hình Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
print(f"Hệ số hồi quy: {lr_model.coef_}")
print(f"Hệ số chặn: {lr_model.intercept_}")

# 4.2 Decision Tree
print("\nHuấn luyện mô hình Decision Tree...")
dt_model = DecisionTreeRegressor(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)
print(f"Độ sâu cây: {dt_model.get_depth()}")
print(f"Số lượng nút lá: {dt_model.get_n_leaves()}")

# 4.3 Random Forest
print("\nHuấn luyện mô hình Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
print(f"Số lượng cây: {rf_model.n_estimators}")
print(f"Tầm quan trọng của các đặc trưng: {rf_model.feature_importances_}")

# 5. Tích hợp COCOMO II
print("\nBước 5: Tích hợp và tính toán theo mô hình COCOMO II")

# Các tham số COCOMO II dựa trên loại dự án
cocomo_params = {
    'Organic': {'A': 2.4, 'B': 1.05},
    'Semi-detached': {'A': 3.0, 'B': 1.12},
    'Embedded': {'A': 3.6, 'B': 1.20}
}

# Sử dụng tham số cho dự án Semi-detached (trung bình)
project_type = 'Semi-detached'
A = cocomo_params[project_type]['A']
B = cocomo_params[project_type]['B']
EM = 1  # Giả định Effort Multipliers = 1 vì không có thông tin chi tiết về Cost Drivers

print(f"Các tham số COCOMO II cho dự án loại {project_type}:")
print(f"A = {A}, B = {B}, EM = {EM}")

# Tính Effort theo COCOMO II: Effort = A × Size^B × EM
# Sử dụng 'PointsAjust' làm Size (giả định đây là Function Points)
data['KLOC'] = data['LOC'] / 1000  # Convert LOC to KLOC

# Replace the original line with this
data['Effort_COCOMO'] = A * (data['KLOC'] ** B) * EM

# Lấy Effort dự đoán của COCOMO II cho tập kiểm tra
cocomo_pred = data.loc[X_test.index, 'Effort_COCOMO']

# 6. Dự đoán và đánh giá mô hình
print("\nBước 6: Dự đoán và đánh giá mô hình")

# Dự đoán từ các mô hình Machine Learning
lr_pred = lr_model.predict(X_test)
dt_pred = dt_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

# Hàm đánh giá mô hình
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # Mean Absolute Percentage Error
    
    print(f"--- {model_name} ---")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R²: {r2:.2f}")
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape}

# Đánh giá tất cả các mô hình, bao gồm COCOMO II
results = {}
results['COCOMO II'] = evaluate_model(y_test, cocomo_pred, "COCOMO II")
results['Linear Regression'] = evaluate_model(y_test, lr_pred, "Linear Regression")
results['Decision Tree'] = evaluate_model(y_test, dt_pred, "Decision Tree")
results['Random Forest'] = evaluate_model(y_test, rf_pred, "Random Forest")

# 7. Trực quan hóa
print("\nBước 7: Trực quan hóa kết quả")

plt.figure(figsize=(16, 12))

# 7.1 So sánh effort thực tế và dự đoán bằng biểu đồ cột
plt.subplot(2, 2, 1)
# Lấy một số mẫu để hiển thị (10 mẫu đầu tiên)
sample_size = min(10, len(y_test))
sample_indices = y_test.index[:sample_size]

# Tạo DataFrame cho các giá trị dự đoán và thực tế
comparison_data = pd.DataFrame({
    'Actual': y_test.iloc[:sample_size].values,
    'COCOMO II': cocomo_pred.iloc[:sample_size].values,
    'Linear Regression': lr_pred[:sample_size],
    'Decision Tree': dt_pred[:sample_size],
    'Random Forest': rf_pred[:sample_size]
})

# Vẽ biểu đồ cột cho từng mẫu
x = np.arange(sample_size)
width = 0.15
plt.bar(x - 2*width, comparison_data['Actual'], width, label='Actual', color='gray')
plt.bar(x - width, comparison_data['COCOMO II'], width, label='COCOMO II', color='blue')
plt.bar(x, comparison_data['Linear Regression'], width, label='Linear Regression', color='red')
plt.bar(x + width, comparison_data['Decision Tree'], width, label='Decision Tree', color='green')
plt.bar(x + 2*width, comparison_data['Random Forest'], width, label='Random Forest', color='purple')

plt.xlabel('Mẫu')
plt.ylabel('Effort (người-giờ)')
plt.title('So sánh Effort Thực tế và Dự đoán (10 mẫu đầu)')
plt.xticks(x, [f'Mẫu {i+1}' for i in range(sample_size)], rotation=45)
plt.legend()

# 7.2 Scatter plot comparing actual vs predicted values
plt.subplot(2, 2, 2)
plt.scatter(y_test, cocomo_pred, label='COCOMO II', alpha=0.5, color='blue')
plt.scatter(y_test, lr_pred, label='Linear Regression', alpha=0.5, color='red')
plt.scatter(y_test, dt_pred, label='Decision Tree', alpha=0.5, color='green')
plt.scatter(y_test, rf_pred, label='Random Forest', alpha=0.5, color='purple')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Effort Thực tế (người-giờ)')
plt.ylabel('Effort Dự đoán (người-giờ)')
plt.title('So sánh Effort Thực tế và Dự đoán')
plt.legend()

# 7.3 Compare errors across models
plt.subplot(2, 2, 3)
models = list(results.keys())
mae_values = [results[model]['mae'] for model in models]
rmse_values = [results[model]['rmse'] for model in models]

x = np.arange(len(models))
width = 0.35

plt.bar(x - width/2, mae_values, width, label='MAE')
plt.bar(x + width/2, rmse_values, width, label='RMSE')
plt.xticks(x, models, rotation=45)
plt.ylabel('Sai số (người-giờ)')
plt.title('So sánh MAE và RMSE giữa các mô hình')
plt.legend()

# 7.4 R² comparison
plt.subplot(2, 2, 4)
r2_values = [results[model]['r2'] for model in models]
plt.bar(models, r2_values, color='lightgreen')
plt.ylabel('R² Score')
plt.title('So sánh R² giữa các mô hình')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('software_effort_prediction_results.png')
plt.show()

# 8. Tìm mô hình tốt nhất và đưa ra kết luận
print("\nBước 8: Kết luận và đề xuất")
best_model = min(results, key=lambda x: results[x]['rmse'])
print(f"Mô hình có hiệu suất tốt nhất dựa trên RMSE: {best_model}")
print(f"MAE: {results[best_model]['mae']:.2f}")
print(f"RMSE: {results[best_model]['rmse']:.2f}")
print(f"R²: {results[best_model]['r2']:.2f}")

# print("\nĐề xuất cải thiện:")
# print("1. Thu thập thêm dữ liệu để cải thiện độ chính xác của mô hình")
# print("2. Thử nghiệm thêm các thuật toán như Gradient Boosting hoặc Neural Networks")
# print("3. Bổ sung thêm các đặc trưng như Technical Complexity Factor (TCF), Environmental Factor (EF)")
# print("4. Tinh chỉnh tham số của các mô hình để cải thiện hiệu suất")