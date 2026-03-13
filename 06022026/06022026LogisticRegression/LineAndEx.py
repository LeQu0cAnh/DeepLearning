# Import thư viện: machine learning: pandas, nump, sklearn; visualization: matplotlib, seaborn 
# Tạo DataFrame
# Tách features và target ra
# Chia dữ liệu train và test (đảm bảo các lớp đồng đều bằng stratify = y)
# Khởi tạo StandardScaler
# Fit trên train set và transform cả train và test
# Kiểm tra mean và std sau chuẩn hóa
# Tạo mô hình Logistic Regression
# Huấn luyện mô hình
# Dự đoán xác suất cho mỗi lớp
# Tính accuracy
# Ma trận nhầm lẫn
# Vẽ heatmap
# Báo cáo chi tiết


# # làm với hoa Iris
# # Import thư viện
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import numpy as np
# # 1. Load dữ liệu
# iris = load_iris()
# X, y = iris.data, iris.target
# # 2. Chia train/test (70/30)
# X_train, X_test, y_train, y_test = train_test_split(
# X, y, test_size=0.3, random_state=42, stratify=y
# )
# # 3. Chuẩn hóa
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
# # 4. Khởi tạo và huấn luyện mô hình
# model = LogisticRegression(
# multi_class='multinomial',
# solver='lbfgs',
# max_iter=1000,
# random_state=42
# )
# model.fit(X_train_scaled, y_train)
# # 5. Dự đoán
# y_pred = model.predict(X_test_scaled)
# # 6. Đánh giá
# print("="*50)
# print("KẾT QUẢ LOGISTIC REGRESSION TRÊN IRIS")
# print("="*50)
# print(f"Train Accuracy: {model.score(X_train_scaled, y_train):.4f}")
# print(f"Test Accuracy: {model.score(X_test_scaled, y_test):.4f}")
# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, y_pred))
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred, target_names=iris.target_names))
# # 7. Dự đoán cho một mẫu mới
# new_sample = np.array([[5.1, 3.5, 1.4, 0.2]]) # Ví dụ: Setosa
# new_sample_scaled = scaler.transform(new_sample)
# prediction = model.predict(new_sample_scaled)
# probability = model.predict_proba(new_sample_scaled)
# print(f"\nDự đoán cho mẫu mới: {iris.target_names[prediction[0]]}")
# print(f"Xác suất: {probability[0]}")




#===========================================================================
# MNIST:
# Import thư viện
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import time
# 1. Load dữ liệu MNIST (lần đầu sẽ download)
print("Đang tải MNIST...")
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X = mnist.data.astype('float32')
y = mnist.target.astype('int')
# 2. Chia train/test (60,000 train / 10,000 test)
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]
# 3. Chuẩn hóa về [0, 1]
X_train_normalized = X_train / 255.0
X_test_normalized = X_test / 255.0
print(f"Shape Train: {X_train_normalized.shape}")
print(f"Shape Test: {X_test_normalized.shape}")
# 4. Khởi tạo mô hình
model = LogisticRegression(
multi_class='multinomial',
solver='saga', # Hiệu quả với dữ liệu lớn
max_iter=100, # Có thể giảm xuống 20-50 để nhanh hơn
tol=0.1,
verbose=1, # Hiển thị tiến trình
n_jobs=-1, # Dùng tất cả CPU cores
random_state=42
)
# 5. Huấn luyện (có thể mất 5-10 phút)
print("\nBắt đầu huấn luyện...")
start_time = time.time()
model.fit(X_train_normalized, y_train)
end_time = time.time()
print(f"Hoàn thành sau {end_time - start_time:.2f} giây")
# 6. Dự đoán
y_pred = model.predict(X_test_normalized)
# 7. Đánh giá
print("="*50)
print("KẾT QUẢ LOGISTIC REGRESSION TRÊN MNIST")
print("="*50)
print(f"Train Accuracy: {model.score(X_train_normalized, y_train):.4f}")
print(f"Test Accuracy: {model.score(X_test_normalized, y_test):.4f}")
print(f"\nSố dự đoán đúng: {accuracy_score(y_test, y_pred, normalize=False)}/10000")
# 8. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
# 9. Tìm chữ số dễ nhầm nhất
# Đường chéo = dự đoán đúng
correct_per_digit = np.diag(cm)
total_per_digit = cm.sum(axis=1)
accuracy_per_digit = correct_per_digit / total_per_digit
print("\nAccuracy theo từng chữ số:")
for digit in range(10): print(f"Chữ số {digit}: {accuracy_per_digit[digit]:.4f}")
worst_digit = np.argmin(accuracy_per_digit)
print(f"\nChữ số khó nhất: {worst_digit} (accuracy: {accuracy_per_digit[worst_digit]:.4f})")
