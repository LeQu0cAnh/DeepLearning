import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
import pydot

# 1. Đọc dữ liệu
# Đảm bảo bạn có file data.csv trong thư mục
features = pd.read_csv('06032026/data.csv')

# 2. Tiền xử lý (One-hot encode cho các cột chữ như 'week')
features = pd.get_dummies(features)

# 3. Tách nhãn (Label) và dữ liệu huấn luyện (Features)
labels = np.array(features['actual'])
features = features.drop('actual', axis = 1)
feature_list = list(features.columns)
features_array = np.array(features)

# 4. Chia tập dữ liệu Train/Test (75% train, 25% test)
train_features, test_features, train_labels, test_labels = train_test_split(
    features_array, labels, test_size = 0.25, random_state = 42
)

# 5. Huấn luyện mô hình Random Forest với 1000 cây quyết định
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(train_features, train_labels)

# 6. Dự báo và tính toán lỗi (Sửa lỗi NameError ở đây)
predictions = rf.predict(test_features)
errors = abs(predictions - test_labels) # Tính lỗi tuyệt đối
mae = round(np.mean(errors), 2)
print('Mean Absolute Error (Lỗi trung bình):', mae, 'degrees.')

# Tính toán độ chính xác (Accuracy) theo phần trăm
mape = 100 * (errors / test_labels)
accuracy = 100 - np.mean(mape)
print('Accuracy (Độ chính xác):', round(accuracy, 2), '%.')

# 7. Đánh giá mức độ quan trọng của các biến (Feature Importance)
importances = list(rf.feature_importances_)
feature_importances = sorted(zip(importances, feature_list), reverse=True)

print("\nThứ tự quan trọng của các biến:")
for importance, name in feature_importances:
    print(f"Variable: {name:20} Importance: {round(importance, 2)}")

# 8. Trực quan hóa mức độ quan trọng
plt.style.use('fivethirtyeight')
x_values = list(range(len(importances)))
plt.bar(x_values, importances, orientation = 'vertical')
plt.xticks(x_values, feature_list, rotation='vertical')
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances')
plt.show()