import pandas as pd
import numpy as np
# Thư viện Machine Learning
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Thư viện visualization
import matplotlib.pyplot as plt
import seaborn as sns



iris = load_iris()
# Tạo DataFrame để dễ xử lý
df = pd.DataFrame(
data=iris.data,
columns=iris.feature_names
)
df['target'] = iris.target
df['species'] = df['target'].map({
0: 'setosa',
1: 'versicolor',
2: 'virginica'
})
# Xem thông tin cơ bản
print(df.head())
print(df.info())
print(df.describe())




# Tách features và target
X = df[['sepal length (cm)', 'sepal width (cm)',
'petal length (cm)', 'petal width (cm)']]
y = df['target']
# Hoặc sử dụng trực tiếp từ iris object
# X = iris.data
# y = iris.target
print(f"Shape của X: {X.shape}")
print(f"Shape của y: {y.shape}")

# Chia dữ liệu 70% train, 30% test
X_train, X_test, y_train, y_test = train_test_split(
X, y,
test_size=0.3,
random_state=42,
stratify=y # Đảm bảo tỷ lệ các lớp đồng đều
)
print(f"Train set: {X_train.shape}")
print(f"Test set: {X_test.shape}")


# Khởi tạo StandardScaler
scaler = StandardScaler()
# Fit trên train set và transform cả train và test
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Kiểm tra mean và std sau chuẩn hóa
print("Mean sau scaling:",
X_train_scaled.mean(axis=0))
print("Std sau scaling:", X_train_scaled.std(axis=0))



# Tạo mô hình Logistic Regression
model = LogisticRegression(
multi_class='multinomial', # Cho phân loại đa lớp
solver='lbfgs', # Thuật toán tối ưu
max_iter=1000, # Số iterations tối đa
random_state=42 # Reproducibility
)




# Huấn luyện mô hình
model.fit(X_train_scaled, y_train)
# Dự đoán trên tập test
y_pred = model.predict(X_test_scaled)
# Dự đoán xác suất cho mỗi lớp
y_pred_proba = model.predict_proba(X_test_scaled)
print("Dự đoán (5 mẫu đầu):", y_pred[:5])
print("Xác suất (5 mẫu đầu):")
print(y_pred_proba[:5])




# Tính accuracy
train_accuracy = model.score(X_train_scaled, y_train)
test_accuracy = model.score(X_test_scaled, y_test)
print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
# # Hoặc sử dụng accuracy_score
# from sklearn.metrics import accuracy_score
# acc = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {acc:.4f}")



# Ma trận nhầm lẫn
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# Vẽ heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d',
xticklabels=iris.target_names,
yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()




# Báo cáo chi tiết
from sklearn.metrics import classification_report
report = classification_report(
y_test, y_pred,
target_names=iris.target_names
)
print(report)