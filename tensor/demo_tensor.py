import random
import numpy as np
from tensor import Tensor
from nn_tensor import TensorMLP

xs = [np.array([[0.0], [0.0]]), np.array([[0.0], [1.0]]),
      np.array([[1.0], [0.0]]), np.array([[1.0], [1.0]])]   # each (2,1)
ys = [0.0, 1.0, 1.0, 0.0]

np.random.seed(0)
mlp = TensorMLP(2, [8, 8, 1], hidden_activation='tanh')

for k in range(200):
    loss = Tensor(0.0)
    for x, y in zip(xs, ys):
        pred = mlp(x)  # shape (1, 1)
        diff = pred - Tensor(np.array([[y]]))  # (1,1) - (1,1)
        loss = loss + (diff * diff).sum()
    mlp.zero_grad()
    loss.backward()
    for p in mlp.parameters():
        p.data -= 0.1 * p.grad
    if k % 40 == 0:
        print(f"iter {k}: loss {loss.data:.4f}")