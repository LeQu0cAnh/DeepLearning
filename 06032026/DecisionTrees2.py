from chefboost import Chefboost as chef
import pandas as pd
import os
import sys

# Đảm bảo Python tìm thấy module 'outputs'
sys.path.append(os.getcwd())

# 1. Đọc dữ liệu Iris
# Lưu ý: Bạn hãy kiểm tra lại đường dẫn file iris.data trên máy của mình
# File iris.data thường không có header, ta nên đặt tên cột để Chefboost hiểu cột cuối là 'Decision'
column_names = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Decision']
file_path = "06032026/iris.data" # Thay đổi path này cho đúng thư mục của bạn

try:
    df = pd.read_csv(file_path, names=column_names)
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file tại {file_path}")
    # Code dự phòng nếu bạn để file ở folder dataset như trong hướng dẫn
    df = pd.read_csv("dataset/iris.data", names=column_names)

# Trộn dữ liệu ngẫu nhiên (Shuffle)
df = df.sample(frac=1).reset_index(drop=True)

print("Dữ liệu Iris sau khi trộn:")
print(df.head())
print("-" * 30)

def run_iris_experiment(algorithm_name, train_df, test_df):
    print(f"\n--- {algorithm_name} algorithm ---")
    config = {'algorithm': algorithm_name}
    
    # Train trên 100 dòng đầu
    model = chef.fit(train_df.copy(), config)
    
    # Test trên phần còn lại (50 dòng)
    print(f"\nPredictions for {algorithm_name} on testing set:")
    correct = 0
    for index, instance in test_df.iterrows():
        prediction = chef.predict(model, instance)
        actual = instance['Decision']
        
        # In kết quả so sánh
        print(f"Actual: {actual} - Predicted: {prediction}")
        
        if str(actual).strip() == str(prediction).strip():
            correct += 1
            
    acc = (correct / len(test_df)) * 100
    print(f"\n>> Final Accuracy on Test Set ({algorithm_name}): {round(acc, 2)}%")
    print("-" * 30)

# Chia dữ liệu: 100 train / 50 test
train_data = df[:100]
test_data = df[100:]

# Thực hiện lần lượt các thuật toán
for alg in ['ID3', 'C4.5', 'CART']:
    run_iris_experiment(alg, train_data, test_data)