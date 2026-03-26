import numpy as np

# Bài toán XOR: Có 4 mẫu đầu vào và đầu ra tương ứng
# Tạo mảng X chứa 4 mẫu đầu vào: [0,0], [0,1], [1,0], [1,1]
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Đầu vào
# Tạo mảng y chứa kết quả đầu ra tương ứng với 4 mẫu đầu vào theo phép XOR: 0, 1, 1, 0
y = np.array([[0], [1], [1], [0]])  # Đầu ra mong muốn

# Hàm kích hoạt sigmoid và đạo hàm của nó
def sigmoid(x):
    # Hàm sigmoid: f(x) = 1/(1+e^(-x)), giới hạn đầu ra trong khoảng (0,1)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    # Đạo hàm của hàm sigmoid: f'(x) = f(x) * (1 - f(x))
    # Lưu ý: x ở đây đã là giá trị của hàm sigmoid, không phải đầu vào gốc
    return x * (1 - x)

# Định nghĩa lớp mạng nơ-ron
class NeuralNetwork:
    def __init__(self, x, y):
        # Khởi tạo mạng với dữ liệu đầu vào x và đầu ra mong muốn y
        self.input = x
        # Khởi tạo ngẫu nhiên trọng số kết nối từ lớp đầu vào (2 node) đến lớp ẩn (4 node)
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        # Khởi tạo ngẫu nhiên trọng số kết nối từ lớp ẩn (4 node) đến lớp đầu ra (1 node)
        self.weights2 = np.random.rand(4, 1)
        # Lưu trữ đầu ra mong muốn
        self.y = y
        # Khởi tạo mảng đầu ra với kích thước giống y và giá trị ban đầu bằng 0
        self.output = np.zeros(y.shape)

    def feedforward(self):
        # Lan truyền thuận - tính đầu ra của mạng với trọng số hiện tại
        # Tính đầu ra của lớp ẩn: input * weights1 qua hàm sigmoid
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        # Tính đầu ra của lớp output: layer1 * weights2 qua hàm sigmoid
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # Lan truyền ngược - cập nhật trọng số dựa trên lỗi
        # Tính gradient
        # Tính đạo hàm của lỗi theo weights2:
        # (layer1.T là chuyển vị của layer1) nhân với (2 * lỗi * đạo hàm sigmoid tại output)
        # Hệ số 2 từ đạo hàm của hàm lỗi bình phương
        d_weights2 = np.dot(self.layer1.T, 
                            (2 * (self.y - self.output) * sigmoid_derivative(self.output)))

        # Tính đạo hàm của lỗi theo weights1:
        # (input.T là chuyển vị của input) nhân với (đạo hàm lỗi theo layer1 * đạo hàm sigmoid tại layer1)
        # Đạo hàm lỗi theo layer1 = đạo hàm lỗi theo output nhân với weights2.T
        d_weights1 = np.dot(self.input.T, 
                            (np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output), 
                                    self.weights2.T) * sigmoid_derivative(self.layer1)))

        # Cập nhật trọng số bằng cách cộng với gradient (learning rate mặc định là 1)
        self.weights1 += d_weights1
        self.weights2 += d_weights2

# Khởi tạo đối tượng mạng neural với dữ liệu X và y
nn = NeuralNetwork(X, y)

# Huấn luyện mạng neural qua 10000 vòng lặp (epochs)
for i in range(10000):
    # Tính đầu ra với trọng số hiện tại
    nn.feedforward()
    # Cập nhật trọng số dựa trên lỗi
    nn.backprop()

    # In ra lỗi sau mỗi 1000 epochs để theo dõi quá trình huấn luyện
    # Lỗi được tính bằng trung bình bình phương sai khác giữa y và output
    if i % 1000 == 0:
        loss = np.mean(np.square(y - nn.output))
        print(f"Epoch {i}: Loss = {loss}")

# Tính đầu ra cuối cùng với trọng số đã huấn luyện
nn.feedforward()
# In kết quả dự đoán của mạng neural
print("\nKết quả dự đoán:")
print(nn.output)