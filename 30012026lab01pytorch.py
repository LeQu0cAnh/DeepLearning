# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# import numpy as np

# # === BƯỚC 1: ĐỊNH NGHĨA MÔ HÌNH ===
# class LinearRegressionModel(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(LinearRegressionModel, self).__init__()
#         self.linear = nn.Linear(input_dim, output_dim)

#     def forward(self, x):
#         return self.linear(x)

# # === BƯỚC 2: TẠO DỮ LIỆU ===
# torch.manual_seed(42)
# X_train = torch.randn(100, 1)
# y_train = 4 + 3 * X_train + torch.randn(100, 1) * 0.5

# X_test = torch.randn(20, 1)
# y_test = 4 + 3 * X_test + torch.randn(20, 1) * 0.5

# # === BƯỚC 3: KHỞI TẠO MÔ HÌNH VÀ CÔNG CỤ ===
# model = LinearRegressionModel(input_dim=1, output_dim=1)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)

# # === BƯỚC 4: HUẤN LUYỆN MÔ HÌNH ===
# num_epochs = 1000
# losses = []

# for epoch in range(num_epochs):
#     # Forward pass
#     y_pred = model(X_train)
#     loss = criterion(y_pred, y_train)
#     losses.append(loss.item())

#     # Backward pass và optimization
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     # In thông tin mỗi 100 epochs
#     if (epoch + 1) % 100 == 0:
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# # === BƯỚC 5: ĐÁNH GIÁ MÔ HÌNH ===
# model.eval() # Chuyển sang chế độ evaluation
# with torch.no_grad(): # Tắt gradient computation
#     y_pred_test = model(X_test)
#     test_loss = criterion(y_pred_test, y_test)

#     # Tính R2 score
#     ss_res = torch.sum((y_test - y_pred_test) ** 2)
#     ss_tot = torch.sum((y_test - torch.mean(y_test)) ** 2)
#     r2_score = 1 - ss_res / ss_tot

# print(f"\n=== KẾT QUẢ ===")
# print(f"Test Loss (MSE): {test_loss.item():.4f}")
# print(f"R2 Score: {r2_score.item():.4f}")
# print(f"Learned W: {model.linear.weight.item():.4f}")
# print(f"Learned b: {model.linear.bias.item():.4f}")

# # === BƯỚC 6: TRỰC QUAN HÓA ===
# plt.figure(figsize=(15, 5))

# # Subplot 1: Loss curve
# plt.subplot(1, 3, 1)
# plt.plot(losses)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Loss qua các Epochs')
# plt.grid(True, alpha=0.3)

# # Subplot 2: Training data và đường hồi quy
# plt.subplot(1, 3, 2)
# plt.scatter(X_train.numpy(), y_train.numpy(), alpha=0.5, label='Dữ liệu huấn luyện')
# X_range = torch.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
# with torch.no_grad():
#     y_range = model(X_range)
# plt.plot(X_range.numpy(), y_range.numpy(), 'r-', linewidth=2, label='Đường hồi quy')
# plt.xlabel('X')
# plt.ylabel('y')
# plt.title('Mô hình hồi quy tuyến tính')
# plt.legend()
# plt.grid(True, alpha=0.3)

# # Subplot 3: Actual vs Predicted
# plt.subplot(1, 3, 3)
# with torch.no_grad():
#     plt.scatter(y_test.numpy(), y_pred_test.numpy(), alpha=0.6)
#     plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)

# plt.xlabel('Giá trị thực tế')
# plt.ylabel('Giá trị dự đoán')
# plt.title('Actual vs Predicted (Test Set)')
# plt.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()



#================================================================================
# sử dụng dữ liệu kaggle với pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# === BƯỚC 1: ĐỊNH NGHĨA MÔ HÌNH ===
class LinearRegressionModel(nn.Module):

    def __init__(self, input_dim, output_dim):
    
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    

    def forward(self, x):
    
        return self.linear(x)
    


def train_housing_full_pytorch():

    try:
        # === BƯỚC 2: TẠO DỮ LIỆU TỪ FILE THỰC ===
        df = pd.read_csv('housing.csv').dropna()
        X_raw = df[['median_income']].values.astype(np.float32)
        y_raw = df[['median_house_value']].values.astype(np.float32)

        sc_X, sc_y = StandardScaler(), StandardScaler()
        X_scaled = sc_X.fit_transform(X_raw)
        y_scaled = sc_y.fit_transform(y_raw)

        X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42
        )

        X_train = torch.from_numpy(X_train_np)
        y_train = torch.from_numpy(y_train_np)
        X_test = torch.from_numpy(X_test_np)
        y_test = torch.from_numpy(y_test_np)

        # === BƯỚC 3: KHỞI TẠO MÔ HÌNH VÀ CÔNG CỤ ===
        model = LinearRegressionModel(1, 1)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # === BƯỚC 4: HUẤN LUYỆN MÔ HÌNH ===
        num_epochs = 1000
        losses = []
        for epoch in range(num_epochs):
        
            model.train()
            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 200 == 0:
            
                print(f'Housing Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            
        

        # === BƯỚC 5: ĐÁNH GIÁ MÔ HÌNH (ĐẦY ĐỦ THÔNG SỐ) ===
        model.eval()
        with torch.no_grad():
        
            y_pred_test = model(X_test)
            mse = criterion(y_pred_test, y_test).item()
            rmse = np.sqrt(mse)
            mae = torch.mean(torch.abs(y_pred_test - y_test)).item()
            
            ss_res = torch.sum((y_test - y_pred_test) ** 2)
            ss_tot = torch.sum((y_test - torch.mean(y_test)) ** 2)
            r2 = (1 - ss_res / ss_tot).item()

        print(f"\n=== KẾT QUẢ GIÁ NHÀ (PYTORCH) ===")
        print(f"MSE: {mse:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
        print(f"R2 Score: {r2:.4f}")
        print(f"Learned W: {model.linear.weight.item():.4f}")
        print(f"Learned b: {model.linear.bias.item():.4f}")

        # === BƯỚC 6: TRỰC QUAN HÓA (3 SUBPLOTS) ===
        plt.figure(figsize=(15, 5))
        
        # 1. Loss Curve
        plt.subplot(1, 3, 1)
        plt.plot(losses, color='blue')
        plt.title('Housing Loss Curve')

        # 2. Regression Line (Chỉ vẽ 200 mẫu cho đỡ rối)
        plt.subplot(1, 3, 2)
        plt.scatter(X_test_np[:200], y_test_np[:200], alpha=0.4, label='Thực tế')
        plt.plot(X_test_np[:200], y_pred_test.numpy()[:200], color='red', label='Dự đoán')
        plt.title('Income vs Price')
        plt.legend()

        # 3. Actual vs Predicted
        plt.subplot(1, 3, 3)
        plt.scatter(y_test_np, y_pred_test.numpy(), alpha=0.2, color='purple')
        plt.plot([-2, 3], [-2, 3], 'r--')
        plt.title('Actual vs Predicted')
        plt.tight_layout()
        plt.show()

        # === BƯỚC 7: DỰ ĐOÁN DỮ LIỆU MỚI ===
        new_incomes = np.array([[3.0], [6.0], [9.0]], dtype=np.float32)
        new_scaled = torch.from_numpy(sc_X.transform(new_incomes))
        with torch.no_grad():
            preds = sc_y.inverse_transform(model(new_scaled).numpy())
        
        print("\n=== DỰ BÁO GIÁ NHÀ MỚI ===")
        for inc, p in zip(new_incomes, preds):
            print(f"Thu nhập {inc[0]} -> Giá dự báo: {p[0]:,.0f} USD")

    except Exception as e:
    
        print(f"Lỗi: {e}")
    


if __name__ == "__main__":

    train_housing_full_pytorch()

#================================================================================
# #giá vàng

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler

# # === BƯỚC 1: ĐỊNH NGHĨA MÔ HÌNH ===
# class GoldRegressionModel(nn.Module):

#     def __init__(self, input_dim, output_dim):
    
#         super(GoldRegressionModel, self).__init__()
#         self.linear = nn.Linear(input_dim, output_dim)
    

#     def forward(self, x):
    
#         return self.linear(x)
    


# def train_gold_full_pytorch():

#     try:
#         # === BƯỚC 2: CHUẨN BỊ DỮ LIỆU ===
#         df = pd.read_csv('monthly_csv.csv').dropna()
#         X_raw = np.arange(len(df)).reshape(-1, 1).astype(np.float32)
#         y_raw = df[['Price']].values.astype(np.float32)

#         sc_X, sc_y = StandardScaler(), StandardScaler()
#         X_scaled = torch.from_numpy(sc_X.fit_transform(X_raw))
#         y_scaled = torch.from_numpy(sc_y.fit_transform(y_raw))

#         # === BƯỚC 3: KHỞI TẠO ===
#         model = GoldRegressionModel(1, 1)
#         criterion = nn.MSELoss()
#         optimizer = optim.Adam(model.parameters(), lr=0.01)

#         # === BƯỚC 4: HUẤN LUYỆN ===
#         num_epochs = 1200
#         losses = []
#         for epoch in range(num_epochs):
        
#             model.train()
#             outputs = model(X_scaled)
#             loss = criterion(outputs, y_scaled)
#             losses.append(loss.item())

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             if (epoch + 1) % 300 == 0:
            
#                 print(f'Gold Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            
        

#         # === BƯỚC 5: ĐÁNH GIÁ (ĐẦY ĐỦ THÔNG SỐ) ===
#         model.eval()
#         with torch.no_grad():
        
#             y_pred = model(X_scaled)
#             mse = criterion(y_pred, y_scaled).item()
#             rmse = np.sqrt(mse)
#             mae = torch.mean(torch.abs(y_pred - y_scaled)).item()
#             r2 = (1 - torch.sum((y_scaled - y_pred)**2) / torch.sum((y_scaled - torch.mean(y_scaled))**2)).item()

#         print(f"\n=== KẾT QUẢ GIÁ VÀNG (PYTORCH) ===")
#         print(f"MSE: {mse:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
#         print(f"R2 Score: {r2:.4f}")
#         print(f"Learned W (Trend): {model.linear.weight.item():.4f}")
#         print(f"Learned b: {model.linear.bias.item():.4f}")

#         # === BƯỚC 6: TRỰC QUAN HÓA (3 SUBPLOTS) ===
#         plt.figure(figsize=(15, 5))
        
#         plt.subplot(1, 3, 1)
#         plt.plot(losses, color='orange')
#         plt.title('Gold Loss Curve')

#         plt.subplot(1, 3, 2)
#         plt.plot(sc_y.inverse_transform(y_scaled.numpy()), label='Giá thật', color='gold', linewidth=2)
#         plt.plot(sc_y.inverse_transform(y_pred.numpy()), color='red', linestyle='--', label='Xu hướng')
#         plt.title('Gold Price Trend')
#         plt.legend()

#         plt.subplot(1, 3, 3)
#         plt.scatter(y_scaled.numpy(), y_pred.numpy(), alpha=0.5, color='orange')
#         plt.plot([-1.5, 3], [-1.5, 3], 'k--')
#         plt.title('Actual vs Predicted')
#         plt.tight_layout()
#         plt.show()

#         # === BƯỚC 7: DỰ BÁO TƯƠNG LAI ===
#         future_years = np.array([[len(df)], [len(df)+5]], dtype=np.float32)
#         future_scaled = torch.from_numpy(sc_X.transform(future_years))
#         with torch.no_grad():
#             future_preds = sc_y.inverse_transform(model(future_scaled).numpy())
        
#         print("\n=== DỰ BÁO GIÁ VÀNG TƯƠNG LAI ===")
#         print(f"Năm tiếp theo: {future_preds[0][0]:.2f} USD")
#         print(f"5 năm tới: {future_preds[1][0]:.2f} USD")

#     except Exception as e:
    
#         print(f"Lỗi: {e}")
    


# if __name__ == "__main__":

#     train_gold_full_pytorch()
