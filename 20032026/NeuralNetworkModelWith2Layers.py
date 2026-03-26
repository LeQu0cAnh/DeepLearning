# import numpy
# import theano
# import theano.tensor as T
# import sklearn.metrics

# def l2(x):
#     return T.sum(x**2)
# examples = 1000
# features = 100
# hidden = 10

# D = (numpy.random.randn(examples, features), numpy.random.randint(size=examples,low=0, high=2))
# training_steps = 1000

# x = T.dmatrix("x")
# y = T.dvector("y")
# w1 = theano.shared(numpy.random.randn(features, hidden), name="w1")
# b1 = theano.shared(numpy.zeros(hidden), name="b1")
# w2 = theano.shared(numpy.random.randn(hidden), name="w2")
# b2 = theano.shared(0., name="b2")
# p1 = T.tanh(T.dot(x, w1) + b1)
# p2 = T.tanh(T.dot(p1, w2) + b2)
# prediction = p2 > 0.5

# error = T.nnet.binary_crossentropy(p2,y)

# loss = error.mean() + 0.01 * (l2(w1) + l2(w2))

# gw1, gb1, gw2, gb2 = T.grad(loss, [w1, b1, w2, b2])

# train = theano.function(inputs=[x,y],outputs=[p2, error], updates=((w1, w1 - 0.1 * gw1),
# (b1, b1 - 0.1 * gb1), (w2, w2 - 0.1 * gw2), (b2, b2 - 0.1 * gb2)))

# predict = theano.function(inputs=[x], outputs=[prediction])

# print("Accuracy before Training:", sklearn.metrics.accuracy_score(D[1], numpy.array(predict(D[0])).ravel()))
# for i in range(training_steps):
#     prediction, error = train(D[0], D[1])
# print("Accuracy after Training:", sklearn.metrics.accuracy_score(D[1],numpy.array(predict(D[0])).ravel()))
# # Accuracy before Training: 0.51
# # Accuracy after Training: 0.716
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score

examples = 1000
features = 100
hidden = 10
training_steps = 1000

X_np = np.random.randn(examples, features).astype(np.float32)
y_np = np.random.randint(size=examples, low=0, high=2).astype(np.float32)

X = torch.from_numpy(X_np)
y = torch.from_numpy(y_np)

class SimpleNet(nn.Module):
    def __init__(self, in_features, hidden_size):
        super(SimpleNet, self).__init__()
    
        self.layer1 = nn.Linear(in_features, hidden_size)
    
        self.layer2 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        p1 = torch.tanh(self.layer1(x))
        p2 = torch.tanh(self.layer2(p1)).squeeze()
        return p2

model = SimpleNet(features, hidden)

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=0.01) # weight_decay chính là L2

def get_accuracy():
    with torch.no_grad():
        outputs = model(X)
        predictions = (outputs > 0.5).float()
        return accuracy_score(y_np, predictions.numpy())

print(f"Accuracy before Training: {get_accuracy():.3f}")

for i in range(training_steps):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(X)
    
    outputs_normalized = (outputs + 1) / 2
    loss = criterion(outputs_normalized, y)
    
    loss.backward()
    optimizer.step()

print(f"Accuracy after Training: {get_accuracy():.3f}")