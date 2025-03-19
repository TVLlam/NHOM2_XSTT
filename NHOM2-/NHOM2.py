# Import các thư viện cần thiết
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Bước 1: Thu thập dữ liệu
# Đọc file dữ liệu
file_path = r'D:\XSTK_TH\BTL_XSTT\DL_VN.csv'
data = pd.read_csv(file_path)

# Bước 2: Làm sạch dữ liệu
# In ra file dữ liệu ban đầu
print("Dữ liệu ban đầu:")
print(data.head())

# Kiểm tra các giá trị thiếu
print("\nCác giá trị thiếu trong dữ liệu:")
print(data.isnull().sum())

# Xử lý các giá trị thiếu (ví dụ: điền bằng giá trị trung bình)
data.fillna(data.mean(), inplace=True)

# Kiểm tra các giá trị lỗi (ví dụ: giá trị âm không hợp lệ)
print("\nCác giá trị lỗi trong dữ liệu:")
print(data[(data < 0).any(axis=1)])

# Xử lý các giá trị lỗi (ví dụ: thay thế bằng giá trị trung bình)
data[data < 0] = np.nan
data.fillna(data.mean(), inplace=True)

# In ra bộ dữ liệu mới sau khi đã làm sạch
print("\nDữ liệu sau khi làm sạch:")
print(data.head())

# Bước 3: EDA (Exploratory Data Analysis)
# In ra các chỉ số thống kê cơ bản
print("\nCác chỉ số thống kê cơ bản:")
print(data.describe())

# In ra các chỉ số thống kê cho từng biến
for column in data.columns:
    print(f"\nThống kê cho biến {column}:")
    print(data[column].describe())

# Vẽ biểu đồ phân phối của giá nhà
plt.figure(figsize=(10, 6))
sns.histplot(data['Giá_Nhà'], kde=True)
plt.title('Phân phối giá nhà')
plt.xlabel('Giá nhà')
plt.ylabel('Tần suất')
plt.show()

# Vẽ biểu đồ cột cho số phòng ngủ
plt.figure(figsize=(10, 6))
sns.countplot(x='Số_Phòng_Ngủ', data=data)
plt.title('Số lượng phòng ngủ')
plt.xlabel('Số phòng ngủ')
plt.ylabel('Số lượng')
plt.show()

# Vẽ biểu đồ tương quan giữa các biến
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Ma trận tương quan')
plt.show()

# Bước 4: Xây dựng mô hình
# Chuẩn bị dữ liệu
# Chia dữ liệu thành biến độc lập (X) và biến phụ thuộc (y)
X = data.drop('Giá_Nhà', axis=1)
y = data['Giá_Nhà']

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Xây dựng mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Đánh giá mô hình
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Thử nghiệm mô hình Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Đánh giá mô hình Random Forest
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"\nRandom Forest - Mean Squared Error: {mse_rf}")
print(f"Random Forest - R-squared: {r2_rf}")

# Bước 5: Phân tích kết quả
# Tóm tắt kết quả
if r2 > 0.7:
    print("Mô hình dự đoán tốt.")
else:
    print("Mô hình cần cải thiện.")

# Đề xuất các cách cải thiện
print("\nĐề xuất cải thiện:")
print("1. Thử nghiệm các thuật toán khác như Random Forest, Gradient Boosting.")
print("2. Thu thập thêm dữ liệu để cải thiện độ chính xác.")
print("3. Xử lý tốt hơn các biến đầu vào, loại bỏ các biến không cần thiết.")