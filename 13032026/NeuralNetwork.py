import pandas as pd
import numpy as np
import os
import tensorflow as tf
import keras
from keras.models import Sequential 
from keras.layers import Dense, Activation, BatchNormalization, Input
from sklearn.model_selection import train_test_split

# Thiết lập seed
np.random.seed(10)

# --- 1. TẢI DỮ LIỆU ---
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'haberman.csv')

column_names = ['age', 'year', 'nodes', 'y']
df = pd.read_csv(file_path, names=column_names, skiprows=1)
for col in column_names:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna()
df.loc[df['y'] == 1, 'y'] = -1
df.loc[df['y'] == 2, 'y'] = 1

# --- 2. CHUẨN HÓA ---
X = df.iloc[:, 0:3] 
Y = df.iloc[:, 3] 
X = (X - X.mean()) / X.std()

# --- 3. CHIA TẬP TRAIN/TEST ---
trainingSetSize = int(df.shape[0] * 0.8)
X1, Y1 = X.iloc[:trainingSetSize, :], Y.iloc[:trainingSetSize]
X2, Y2 = X.iloc[trainingSetSize:, :], Y.iloc[trainingSetSize:]

# --- 4. TẠO MẠNG NƠ-RON (Đã sửa RMSE và Input) ---
def rmse(y_true, y_pred):
    # Dùng tf trực tiếp để tính toán chính xác nhất
    return tf.sqrt(tf.reduce_mean(tf.square(tf.cast(y_pred, tf.float32) - tf.cast(y_true, tf.float32)), axis=-1))

nn_model = Sequential([
    Input(shape=(3,)), # Thay cho input_dim để hết Warning
    Dense(4, activation='relu'),
    BatchNormalization(),
    Dense(3, activation='tanh'),
    BatchNormalization(),
    Dense(1, activation='tanh')
])

nn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', rmse])

# --- 5. HUẤN LUYỆN ---
print("\n--- Đang huấn luyện... ---")
# Đổi verbose=1 để bạn thấy nó chạy từng Epoch cho vui mắt nhé
nn_model.fit(X1.values, Y1.values, epochs=450, batch_size=10, verbose=1)

# --- 6. ĐÁNH GIÁ ---
print("\n--- Kết quả sau huấn luyện ---")
Y1_pred = nn_model.predict(X1.values)
Y1_pred = np.where(Y1_pred >= 0, 1, -1)

success_train = 100 * np.sum(Y1_pred[:, 0] == Y1.values) / len(Y1)
print(f"Tỷ lệ thành công (Train): {success_train:.2f}%")

Y2_pred = nn_model.predict(X2.values)
Y2_pred = np.where(Y2_pred >= 0, 1, -1)
success_test = 100 * np.sum(Y2_pred[:, 0] == Y2.values) / len(Y2)
print(f"Tỷ lệ thành công (Test): {success_test:.2f}%")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# --- 7. VẼ ĐỒ THỊ QUÁ TRÌNH HUẤN LUYỆN ---
plt.figure(figsize=(12, 5))

# Đồ thị Loss
plt.subplot(1, 2, 1)
plt.plot(nn_model.history.history['loss'], label='Train Loss', color='red')
plt.title('Mô hình Loss qua các Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Đồ thị Accuracy
plt.subplot(1, 2, 2)
plt.plot(nn_model.history.history['accuracy'], label='Train Accuracy', color='blue')
plt.title('Độ chính xác qua các Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# --- 8. IN BÁO CÁO CHI TIẾT ---
print("\n--- Báo cáo chi tiết tập Test ---")
print(classification_report(Y2.values, Y2_pred))

# Vẽ Ma trận nhầm lẫn (Confusion Matrix)
cm = confusion_matrix(Y2.values, Y2_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Sống', 'Chết'], yticklabels=['Sống', 'Chết'])
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.title('Ma trận nhầm lẫn (Confusion Matrix)')
plt.show()