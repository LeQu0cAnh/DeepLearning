import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier

# 1. Load data
# Lưu ý: Thay đổi đường dẫn theo cấu trúc folder của bạn
train_data = pd.read_csv("./06032026/train.csv")
test_data = pd.read_csv("./06032026/test.csv")

# 2. Prepare data
y_train_full = train_data["Survived"]
train_data_features = train_data.drop(labels="Survived", axis=1)

# SỬA LỖI: Thay .append bằng pd.concat
full_data = pd.concat([train_data_features, test_data])

# Loại bỏ các cột không cần thiết
drop_columns = ["Name", "Age", "SibSp", "Ticket", "Cabin", "Parch", "Embarked"]
full_data.drop(labels=drop_columns, axis=1, inplace=True)

# Chuyển đổi giới tính thành số (One-hot encoding) và điền giá trị thiếu
full_data = pd.get_dummies(full_data, columns=["Sex"])
full_data.fillna(value=0.0, inplace=True)

# Tách lại tập train và test
X_train_full = full_data.values[0:891]
X_test_final = full_data.values[891:]

# Chuẩn hóa dữ liệu về khoảng [0, 1]
scaler = MinMaxScaler()
X_train_full = scaler.fit_transform(X_train_full)
X_test_final = scaler.transform(X_test_final)

# Chia tập Validation (30%) để tối ưu tham số
state = 12
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.30, random_state=state
)

# 3. Tối ưu hóa Learning Rate
print("--- Learning Rate Optimization ---")
lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]
for learning_rate in lr_list:
    gb_clf = GradientBoostingClassifier(
        n_estimators=20, 
        learning_rate=learning_rate, 
        max_features=2, 
        max_depth=2, 
        random_state=0
    )
    gb_clf.fit(X_train, y_train)
    
    print(f"Learning rate: {learning_rate}")
    print(f"Accuracy score (training): {gb_clf.score(X_train, y_train):.3f}")
    print(f"Accuracy score (validation): {gb_clf.score(X_val, y_val):.3f}")
    print("-" * 20)

# 4. Chạy mô hình tốt nhất (LR=0.5) và xem báo cáo chi tiết
print("\n--- Final Model Evaluation (LR=0.5) ---")
best_gb = GradientBoostingClassifier(n_estimators=20, learning_rate=0.5, max_features=2, max_depth=2, random_state=0)
best_gb.fit(X_train, y_train)
predictions = best_gb.predict(X_val)

print("Confusion Matrix:")
print(confusion_matrix(y_val, predictions))
print("\nClassification Report:")
print(classification_report(y_val, predictions))