import numpy as np
from tensor import Tensor   # adjust import to your path

class TensorLayer:
    def __init__(self, nin, nout, activation='tanh'):
        # He-ish / small random init, scaled to keep activations sane
        self.W = Tensor(np.random.randn(nout, nin) * (1.0 / nin**0.5))
        self.b = Tensor(np.zeros((nout, 1)))  
        self.activation = activation

    def __call__(self, x):
        act = self.W @ x + self.b      # (nout,nin)@(nin,) + (nout,) = (nout,)
        if self.activation == 'tanh':
            return act.tanh()
        elif self.activation == 'relu':
            return act.relu()
        elif self.activation == 'linear':
            return act
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def parameters(self):
        return [self.W, self.b]


class TensorMLP:
    def __init__(self, nin, nouts, hidden_activation='tanh'):
        sz = [nin] + nouts
        self.layers = []
        for i in range(len(nouts)):
            is_last = (i == len(nouts) - 1)
            act = 'linear' if is_last else hidden_activation
            self.layers.append(TensorLayer(sz[i], sz[i+1], activation=act))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

