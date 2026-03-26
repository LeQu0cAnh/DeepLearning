# import theano.tensor as T
# from theano import function
# from theano import shared
# import numpy
# x = T.dmatrix('x')
# y = shared(numpy.array([[4, 5, 6]]))
# z = T.sum(((x * x) + y) * x)
# f = function(inputs = [x], outputs = [z])
# g = T.grad(z,[x])
# g_f = function([x], g)

# print("Original:", f([[1, 2, 3]]))
# print("Original Gradient:", g_f([[1, 2, 3]]))
# y.set_value(numpy.array([[1, 1, 1]]))
# print("Updated:", f([[1, 2, 3]]))
# print("Updated Gradient", g_f([[1, 2, 3]]))
# # Original: [array(68.0)]
# # Original Gradient: [array([[ 7., 17., 33.]])]
# # Updated: [array(42.0)]
# # Updated Gradient [array([[ 4., 13., 28.]])]
import torch

y = torch.tensor([[4.0, 5.0, 6.0]])

def compute_and_grad(x_val, y_val):
    x = torch.tensor(x_val, requires_grad=True)
    
    z = torch.sum(((x * x) + y_val) * x)
    
    z.backward()
    
    return z, x.grad

z_val, grad_val = compute_and_grad([[1.0, 2.0, 3.0]], y)
print("Original Value:", z_val.item())
print("Original Gradient:", grad_val.numpy())

y = torch.tensor([[1.0, 1.0, 1.0]])

z_val_new, grad_val_new = compute_and_grad([[1.0, 2.0, 3.0]], y)
print("Updated Value:", z_val_new.item())
print("Updated Gradient:", grad_val_new.numpy())