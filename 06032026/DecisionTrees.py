from chefboost import Chefboost as chef # Import đúng module chứa hàm fit
import pandas as pd
import os
import sys

# Đảm bảo Python tìm thấy module 'outputs'
sys.path.append(os.getcwd())

# Đường dẫn file của bạn
file_path = "06032026/golf.txt"
df = pd.read_csv(file_path)

def evaluate_model(algorithm_name, train_df, test_df):
    print(f"\n--- {algorithm_name} algorithm ---")
    
    # Cấu hình thuật toán
    config = {'algorithm': algorithm_name}
    
    # Huấn luyện mô hình
    # Bây giờ chef đã trỏ thẳng vào Chefboost nên gọi .fit() sẽ chạy được
    print(f"Train model on training dataset ({algorithm_name}):")
    model = chef.fit(train_df.copy(), config)
    
    # Dự báo cho một vector x cụ thể
    xrow = ['Sunny', 'Hot', 'High', 'Weak']
    prediction = chef.predict(model, xrow)
    print(f"Prediction for {xrow}: {prediction}")
    
    # Dự báo trên tập TESTING
    print("\nCompare predictions and expected results on TESTING dataset:")
    correct_count = 0
    for index, instance in test_df.iterrows():
        # Chefboost.predict cần truyền model và instance (dòng dữ liệu)
        prediction = chef.predict(model, instance)
        actual = instance['Decision']
        print(f"Actual: {actual} - Predicted: {prediction}")
        
        if str(actual).strip() == str(prediction).strip():
            correct_count += 1
            
    accuracy = round(correct_count / test_df.shape[0], 4)
    print(f"Accuracy on TESTING data set: {accuracy * 100}%")

# Chia dữ liệu
train_data = df[:8]
test_data = df[8:]

# Chạy thử thuật toán ID3 trước để kiểm tra
evaluate_model('ID3', train_data, test_data)

# Nếu ID3 chạy tốt, bạn có thể gọi tiếp C4.5 và CART bên dưới
# evaluate_model('C4.5', train_data, test_data)
# evaluate_model('CART', train_data, test_data)