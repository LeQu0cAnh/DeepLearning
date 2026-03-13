#load dữ liệu MNIST
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
# Download MNIST (có thể mất vài phút lần đầu)
mnist = fetch_openml('mnist_784', version=1, parser='auto')
# Tách features và labels
X = mnist.data.astype('float32').values # Shape: (70000, 784)
y = mnist.target.astype('int') # Shape: (70000,)
print(f"Shape của X: {X.shape}")
print(f"Shape của y: {y.shape}")
print(f"Giá trị pixel: min={X.min()}, max={X.max()}")


#Flatten ảnh thành vector

# Ảnh 28x28 cần chuyển thành vector 784 chiều
# Nếu load từ keras (shape: 60000, 28, 28)
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]
# Nếu load từ sklearn thì đã flat sẵn (70000, 784)
print(f"Shape sau flatten: {X_train.shape}")
# Giải thích: mỗi hàng là một ảnh được "dẹp"
# từ ma trận 28x28 thành vector 784 phần tử


#chuẩn hóa
# Chia cho 255 để đưa về khoảng [0, 1]
X_train_normalized = X_train / 255.0
X_test_normalized = X_test / 255.0
print(f"Giá trị sau chuẩn hóa: min={X_train_normalized.min()}, max={X_train_normalized.max()}")
# Tại sao chia 255?
# - Giá trị pixel gốc: 0-255 (uint8)
# - Sau chia: 0.0-1.0 (float)
# - Giúp gradient descent ổn định hơn
# - Tránh numerical issues với exponential trong sigmoid


# Khởi tạo mô hình Logistic Regression
from sklearn.linear_model import LogisticRegression
# Cấu hình cho MNIST
model = LogisticRegression(
multi_class='multinomial',
solver='saga', # 'saga' tốt cho dữ liệu lớn
max_iter=100, # Giảm để tiết kiệm thời gian
tol=0.1, # Tolerance cho convergence
verbose=1, # Hiển thị progress
n_jobs=-1, # Sử dụng tất cả CPU cores
random_state=42
)

# Huấn luyện mô hình
import time
print("Bắt đầu huấn luyện...")
start_time = time.time()
# Train trên 60,000 ảnh
model.fit(X_train_normalized, y_train)
end_time = time.time()
print(f"Hoàn thành sau {end_time - start_time:.2f} giây")
# Tips để tăng tốc:
# 1. Giảm số mẫu: Chỉ dùng 10,000 mẫu đầu
# X_train_normalized[:10000], y_train[:10000]
# 2. Giảm max_iter xuống 10-20
# 3. Tăng tol lên 0.1 hoặc 1.0

# Dự đoán cho 10,000 ảnh test
y_pred = model.predict(X_test_normalized)
# Xem một số dự đoán
print("10 dự đoán đầu tiên:", y_pred[:10])
print("10 nhãn thực tế:", y_test[:10])
# Dự đoán xác suất
y_pred_proba = model.predict_proba(X_test_normalized)
print("\nXác suất cho ảnh đầu tiên:")
print(y_pred_proba[0])
print(f"Dự đoán: {y_pred[0]}, Xác suất: {y_pred_proba[0][y_pred[0]]:.4f}")


# Đánh giá mô hình:
# Tính accuracy
from sklearn.metrics import accuracy_score
train_acc = model.score(X_train_normalized, y_train)
test_acc = model.score(X_test_normalized, y_test)
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
# Kết quả điển hình:
# Train: ~93-94%
# Test: ~91-92%


# Ma trận nhầm lẫn (confusion matrix)
from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - MNIST')
plt.show()
# Ma trận 10x10 cho 10 chữ số
# Đường chéo: dự đoán đúng
# Ngoài đường chéo: nhầm lẫn

# Tìm các mẫu dự đoán sai
incorrect_indices = np.where(y_pred != y_test)[0]
print(f"Số lượng dự đoán sai: {len(incorrect_indices)}")
# Visualize một số ảnh dự đoán sai
fig, axes = plt.subplots(3, 5, figsize=(15, 10))
axes = axes.ravel()
for i, idx in enumerate(incorrect_indices[:15]):
    axes[i].imshow(X_test[idx].reshape(28, 28), cmap='gray')
    axes[i].set_title(f"True: {y_test[idx]}, Pred: {y_pred[idx]}")
    axes[i].axis('off')
plt.tight_layout()
plt.show()
