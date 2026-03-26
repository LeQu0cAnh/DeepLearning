# import theano.tensor as T
# from theano import function

# # sigmoid
# a = T.dmatrix('a')
# f_a = T.nnet.sigmoid(a)
# f_sigmoid = function([a],[f_a])
# print("sigmoid:", f_sigmoid([[-1,0,1]]))

# # tanh
# b = T.dmatrix('b')
# f_b = T.tanh(b)
# f_tanh = function([b],[f_b])
# print("tanh:", f_tanh([[-1,0,1]]))

# # fast sigmoid
# c = T.dmatrix('c')
# f_c = T.nnet.ultra_fast_sigmoid(c)
# f_fast_sigmoid = function([c],[f_c])
# print("fast sigmoid:", f_fast_sigmoid([[-1,0,1]]))

# # softplus
# d = T.dmatrix('d')
# f_d = T.nnet.softplus(d)
# f_softplus = function([d],[f_d])
# print("soft plus:",f_softplus([[-1,0,1]]))
# # relu
# e = T.dmatrix('e')
# f_e = T.nnet.relu(e)
# f_relu = function([e],[f_e])
# print("relu:",f_relu([[-1,0,1]]))

# # softmax
# f = T.dmatrix('f')
# f_f = T.nnet.softmax(f)
# f_softmax = function([f],[f_f])
# print("soft max:",f_softmax([[-1,0,1]]))
import torch
import torch.nn.functional as F

# Dữ liệu đầu vào (Tensor)
# Trong PyTorch, chúng ta dùng torch.tensor thay cho T.dmatrix
input_data = torch.tensor([[-1.0, 0.0, 1.0]])

print("--- Activation Functions (PyTorch) ---")

# 1. Sigmoid
f_sigmoid = torch.sigmoid(input_data)
print("Sigmoid:     ", f_sigmoid.numpy())

# 2. Tanh
f_tanh = torch.tanh(input_data)
print("Tanh:        ", f_tanh.numpy())

# 3. Fast Sigmoid (Trong PyTorch thường dùng Sigmoid chuẩn vì đã được tối ưu rất tốt)
# Nếu muốn xấp xỉ nhanh, có thể dùng công thức: x / (1 + abs(x))
f_fast_sigmoid = 0.5 * (input_data / (1 + torch.abs(input_data))) + 0.5
print("Fast Sigmoid:", f_fast_sigmoid.numpy())

# 4. Softplus
f_softplus = F.softplus(input_data)
print("Softplus:    ", f_softplus.numpy())

# 5. ReLU
f_relu = F.relu(input_data)
print("ReLU:        ", f_relu.numpy())

# 6. Softmax
f_softmax = F.softmax(input_data, dim=1)
print("Softmax:     ", f_softmax.numpy())