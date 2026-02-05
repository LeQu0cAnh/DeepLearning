# Import các thư viện cần thiết
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# # === BƯỚC 1: TẠO DỮ LIỆU MẪU ===
# np.random.seed(42)
# X = 2 * np.random.rand(10000, 1) # 100 điểm, 1 đặc trưng
# y = 4 + 3 * X + np.random.randn(10000, 1) # y = 4 + 3x + nhiễu

# # === BƯỚC 2: CHIA TÁCH DỮ LIỆU ===
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # === BƯỚC 3: KHỞI TẠO VÀ HUẤN LUYỆN MÔ HÌNH ===
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Hiển thị các hệ số đã học
# print(f"Hệ số chặn (β₀): {model.intercept_[0]:.4f}")
# print(f"Hệ số góc (β₁): {model.coef_[0][0]:.4f}")

# # === BƯỚC 4: DỰ ĐOÁN ===
# y_pred = model.predict(X_test)

# # === BƯỚC 5: ĐÁNH GIÁ MÔ HÌNH ===
# r2 = r2_score(y_test, y_pred)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# mae = mean_absolute_error(y_test, y_pred)

# print(f"\n=== KẾT QUẢ ĐÁNH GIÁ ===")
# print(f"R² Score: {r2:.4f}")
# print(f"RMSE: {rmse:.4f}")
# print(f"MAE: {mae:.4f}")

# # === BƯỚC 6: TRỰC QUAN HÓA ===
# plt.figure(figsize=(12, 5))

# # Subplot 1: Dữ liệu và đường hồi quy
# plt.subplot(1, 2, 1)
# plt.scatter(X_train, y_train, color='lightblue', label='Tập huấn luyện', alpha=0.6)
# plt.scatter(X_test, y_test, color='blue', label='Tập kiểm tra', alpha=0.6)
# plt.plot(X_test, y_pred, color='red', linewidth=2, label='Đường hồi quy')
# plt.xlabel('X')
# plt.ylabel('y')
# plt.title('Hồi quy tuyến tính')
# plt.legend()
# plt.grid(True, alpha=0.3)

# # Subplot 2: Actual vs Predicted
# plt.subplot(1, 2, 2)
# plt.scatter(y_test, y_pred, alpha=0.6, color='green')
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Dự đoán hoàn hảo')
# plt.xlabel('Giá trị thực tế')
# plt.ylabel('Giá trị dự đoán')
# plt.title('Actual vs Predicted')
# plt.legend()
# plt.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()

# # === BƯỚC 7: DỰ ĐOÁN CHO DỮ LIỆU MỚI ===
# new_X = np.array([[0.5], [1.0], [1.5]])
# predictions = model.predict(new_X)

# print(f"\n=== DỰ ĐOÁN CHO DỮ LIỆU MỚI ===")
# for x_val, pred in zip(new_X, predictions):
#     print(f"x = {x_val[0]:.1f} -> ŷ = {pred[0]:.4f}")

# # === BƯỚC 1: TẠO DỮ LIỆU MẪU ===
# np.random.seed(36)
# X = 2 * np.random.rand(20000, 1) # 100 điểm, 1 đặc trưng
# y = 4 + 3 * X + np.random.randn(20000, 1) # y = 4 + 3x + nhiễu

# # === BƯỚC 2: CHIA TÁCH DỮ LIỆU ===
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, random_state=36
# )

# # === BƯỚC 3: KHỞI TẠO VÀ HUẤN LUYỆN MÔ HÌNH ===
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Hiển thị các hệ số đã học
# print(f"Hệ số chặn (β₀): {model.intercept_[0]:.4f}")
# print(f"Hệ số góc (β₁): {model.coef_[0][0]:.4f}")

# # === BƯỚC 4: DỰ ĐOÁN ===
# y_pred = model.predict(X_test)

# # === BƯỚC 5: ĐÁNH GIÁ MÔ HÌNH ===
# r2 = r2_score(y_test, y_pred)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# mae = mean_absolute_error(y_test, y_pred)

# print(f"\n=== KẾT QUẢ ĐÁNH GIÁ ===")
# print(f"R² Score: {r2:.4f}")
# print(f"RMSE: {rmse:.4f}")
# print(f"MAE: {mae:.4f}")

# # === BƯỚC 6: TRỰC QUAN HÓA ===
# plt.figure(figsize=(12, 5))

# # Subplot 1: Dữ liệu và đường hồi quy
# plt.subplot(1, 2, 1)
# plt.scatter(X_train, y_train, color='lightblue', label='Tập huấn luyện', alpha=0.6)
# plt.scatter(X_test, y_test, color='blue', label='Tập kiểm tra', alpha=0.6)
# plt.plot(X_test, y_pred, color='red', linewidth=2, label='Đường hồi quy')
# plt.xlabel('X')
# plt.ylabel('y')
# plt.title('Hồi quy tuyến tính')
# plt.legend()
# plt.grid(True, alpha=0.3)

# # Subplot 2: Actual vs Predicted
# plt.subplot(1, 2, 2)
# plt.scatter(y_test, y_pred, alpha=0.6, color='green')
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Dự đoán hoàn hảo')
# plt.xlabel('Giá trị thực tế')
# plt.ylabel('Giá trị dự đoán')
# plt.title('Actual vs Predicted')
# plt.legend()
# plt.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()

# # === BƯỚC 7: DỰ ĐOÁN CHO DỮ LIỆU MỚI ===
# new_X = np.array([[0.5], [1.0], [1.5]])
# predictions = model.predict(new_X)

# print(f"\n=== DỰ ĐOÁN CHO DỮ LIỆU MỚI ===")
# for x_val, pred in zip(new_X, predictions):
#     print(f"x = {x_val[0]:.1f} -> ŷ = {pred[0]:.4f}")

#===========================================================================================================================
#Dữ liệu từ kaggle

#giá nhà = f(tiền thu nhập trung bình)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def train_full_process_with_real_data():
# {
    try:
        # === BƯỚC 1: ĐỌC DỮ LIỆU THỰC TẾ ===
        # Đọc file housing.csv từ thư mục hiện tại
        df = pd.read_csv('housing.csv', encoding='ISO-8859-1')
        df = df.dropna() # Loại bỏ các dòng trống để tránh lỗi mô hình

        # X: Thu nhập trung bình (median_income)
        # y: Giá nhà trung bình (median_house_value)
        X = df[['median_income']].values
        y = df['median_house_value'].values

        # === BƯỚC 2: CHIA TÁCH DỮ LIỆU ===
        # Chia 80% để huấn luyện, 20% để kiểm tra
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # === BƯỚC 3: KHỞI TẠO VÀ HUẤN LUYỆN MÔ HÌNH ===
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Hiển thị các hệ số đã học (Sửa lỗi index tại đây)
        print(f"Hệ số chặn (β₀): {model.intercept_:.4f}")
        print(f"Hệ số góc (β₁): {model.coef_[0]:.4f}")

        # === BƯỚC 4: DỰ ĐOÁN TRÊN TẬP KIỂM TRA ===
        y_pred = model.predict(X_test)

        # === BƯỚC 5: ĐÁNH GIÁ MÔ HÌNH ===
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        print(f"\n=== KẾT QUẢ ĐÁNH GIÁ ===")
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")

        # === BƯỚC 6: TRỰC QUAN HÓA (2 BIỂU ĐỒ) ===
        plt.figure(figsize=(12, 5))

        # Subplot 1: Dữ liệu thực tế và đường hồi quy
        plt.subplot(1, 2, 1)
        plt.scatter(X_test[:500], y_test[:500], color='lightblue', label='Thực tế (500 mẫu)', alpha=0.6)
        plt.plot(X_test[:500], y_pred[:500], color='red', linewidth=2, label='Đường hồi quy')
        plt.xlabel('Thu nhập (Median Income)')
        plt.ylabel('Giá nhà (House Value)')
        plt.title('Hồi quy tuyến tính trên dữ liệu thật')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Subplot 2: So sánh Giá trị thực tế vs Giá trị dự đoán
        plt.subplot(1, 2, 2)
        plt.scatter(y_test, y_pred, alpha=0.1, color='green')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Lý tưởng')
        plt.xlabel('Giá trị thực tế')
        plt.ylabel('Giá trị dự đoán')
        plt.title('Actual vs Predicted')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # === BƯỚC 7: DỰ ĐOÁN CHO DỮ LIỆU MỚI (KHÔNG ĐƯỢC THIẾU) ===
        # Dự đoán giá nhà cho các mức thu nhập: 2.0, 5.0, 8.0 (đơn vị 10k USD)
        new_X = np.array([[2.0], [5.0], [8.0]])
        predictions = model.predict(new_X)

        print(f"\n=== DỰ ĐOÁN CHO DỮ LIỆU MỚI ===")
        for x_val, pred in zip(new_X, predictions):
        # {
            print(f"Thu nhập = {x_val[0]:.1f} -> Dự báo giá nhà = {pred:.2f} USD")
        # }

    except Exception as e:
    # {
        print(f"Lỗi hệ thống: {e}")
    # }
# }

if __name__ == "__main__":
# {
    train_full_process_with_real_data()
# }




# # giá vàng
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# def train_gold_price_model():
# # {
#     try:
#         # === BƯỚC 1: ĐỌC DỮ LIỆU GIÁ VÀNG ===
#         # Thay 'housing.csv' bằng 'annual_csv.csv'
#         df = pd.read_csv('monthly_csv.csv', encoding='ISO-8859-1')
#         df = df.dropna()

#         # Giả định: 'Date' là biến X (thời gian) và 'Price' là biến y (giá vàng)
#         # Bạn có thể đổi lại 'Date' thành 'Year' nếu file dùng tên đó
#         X = np.arange(len(df)).reshape(-1, 1) # Chuyển thứ tự thời gian thành số để train
#         y = df['Price'].values

#         # === BƯỚC 2: CHIA TÁCH DỮ LIỆU ===
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, random_state=42
#         )

#         # === BƯỚC 3: KHỞI TẠO VÀ HUẤN LUYỆN MÔ HÌNH ===
#         model = LinearRegression()
#         model.fit(X_train, y_train)

#         # Hiển thị các hệ số đã học
#         print(f"Hệ số chặn (β₀): {model.intercept_:.4f}")
#         print(f"Hệ số góc (β₁): {model.coef_[0]:.4f}")

#         # === BƯỚC 4: DỰ ĐOÁN TRÊN TẬP KIỂM TRA ===
#         y_pred = model.predict(X_test)

#         # === BƯỚC 5: ĐÁNH GIÁ MÔ HÌNH ===
#         r2 = r2_score(y_test, y_pred)
#         rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#         mae = mean_absolute_error(y_test, y_pred)

#         print(f"\n=== KẾT QUẢ ĐÁNH GIÁ GIÁ VÀNG ===")
#         print(f"R² Score: {r2:.4f}")
#         print(f"RMSE: {rmse:.4f}")
#         print(f"MAE: {mae:.4f}")

#         # === BƯỚC 6: TRỰC QUAN HÓA (2 BIỂU ĐỒ) ===
#         plt.figure(figsize=(12, 5))

#         # Subplot 1: Xu hướng giá vàng và đường hồi quy
#         plt.subplot(1, 2, 1)
#         plt.scatter(X, y, color='gold', label='Giá vàng thực tế', alpha=0.6)
#         plt.plot(X, model.predict(X), color='red', linewidth=2, label='Xu hướng hồi quy')
#         plt.xlabel('Chỉ số thời gian (Năm)')
#         plt.ylabel('Giá vàng')
#         plt.title('Hồi quy tuyến tính: Xu hướng Giá Vàng')
#         plt.legend()
#         plt.grid(True, alpha=0.3)

#         # Subplot 2: Actual vs Predicted
#         plt.subplot(1, 2, 2)
#         plt.scatter(y_test, y_pred, alpha=0.7, color='orange')
#         plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Lý tưởng')
#         plt.xlabel('Giá thực tế')
#         plt.ylabel('Giá dự đoán')
#         plt.title('So sánh Thực tế vs Dự đoán')
#         plt.legend()
#         plt.grid(True, alpha=0.3)

#         plt.tight_layout()
#         plt.show()

#         # === BƯỚC 7: DỰ ĐOÁN CHO THỜI ĐIỂM TƯƠNG LAI ===
#         # Dự đoán cho 3 mốc thời gian tiếp theo trong danh sách
#         next_steps = np.array([[len(df)], [len(df)+1], [len(df)+2]])
#         predictions = model.predict(next_steps)

#         print(f"\n=== DỰ BÁO GIÁ VÀNG TƯƠNG LAI ===")
#         for i, pred in enumerate(predictions):
#         # {
#             print(f"Thời điểm tiếp theo +{i+1} -> Giá dự báo = {pred:.2f}")
#         # }

#     except Exception as e:
#     # {
#         print(f"Lỗi hệ thống: {e}")
#     # }
# # }

# if __name__ == "__main__":
# # {
#     train_gold_price_model()
# # }