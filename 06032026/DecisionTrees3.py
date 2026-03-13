from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# 1. Tải dữ liệu
iris = load_iris()
X, y = iris.data, iris.target

# 2. Chia dữ liệu: Để khách quan, ta nên dùng train_test_split để xáo trộn dữ liệu
# Nếu làm theo code cũ (X[0:135]), tập test sẽ bị mất cân bằng (chỉ có nhãn 2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

print(f"Số lượng mẫu huấn luyện: {len(X_train)}")
print(f"Số lượng mẫu kiểm tra: {len(X_test)}")

# 3. Khởi tạo và huấn luyện mô hình Decision Tree
# Bạn có thể chọn criterion='entropy' (giống ID3/C4.5) hoặc 'gini' (mặc định - giống CART)
clf = tree.DecisionTreeClassifier(criterion='entropy', random_state=42)
clf = clf.fit(X_train, y_train)

# 4. Dự báo trên tập kiểm tra
y_pred = clf.predict(X_test)

# 5. Hiển thị kết quả đánh giá
print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 6. Vẽ sơ đồ cây quyết định trực quan
plt.figure(figsize=(12, 8))
tree.plot_tree(clf, 
               feature_names=iris.feature_names,  
               class_names=iris.target_names,
               filled=True, 
               rounded=True)
plt.title("Decision Tree Visualization (Sklearn)")
plt.show()