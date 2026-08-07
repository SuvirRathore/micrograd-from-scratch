import random


from engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, nin, activation='tanh'):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.activation = activation
        if activation == 'blend':
            self.alpha_raw = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum(wi * xi for wi, xi in zip(self.w, x)) + self.b
        if self.activation == 'tanh':
            return act.tanh()
        elif self.activation == 'relu':
            return act.relu()
        elif self.activation == 'linear':
            return act
        elif self.activation == 'blend':
            a = self.alpha_raw.sigmoid()
            return a * act.tanh() + (1 - a) * act.relu()
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def parameters(self):
        params = self.w + [self.b]
        if self.activation == 'blend':
            params.append(self.alpha_raw)
        return params


#class Neuron(Module):
#    def __init__(self, nin, activation='tanh'):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.activation = activation

#    def __call__(self, x):
        act = sum(wi * xi for wi, xi in zip(self.w, x)) + self.b
        if self.activation == 'tanh':
            return act.tanh()
        elif self.activation == 'relu':
            return act.relu()
        elif self.activation == 'linear':
            return act
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

#    def parameters(self):
        return self.w + [self.b]

class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    def __init__(self, nin, nouts, hidden_activations='tanh'):
        n_hidden = len(nouts) - 1
        if isinstance(hidden_activations, str):
            hidden_activations = [hidden_activations] * n_hidden
        assert len(hidden_activations) == n_hidden, \
            f"need {n_hidden} hidden activations, got {len(hidden_activations)}"
        activations = list(hidden_activations) + ['linear']

        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1], activation=activations[i])
                       for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


# could also start neuron like this:
#class Neuron(Module):
#    def __init__(self, nin, activation=None):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.activation = activation

#    def __call__(self, x):
        act = sum(wi * xi for wi, xi in zip(self.w, x)) + self.b
        return self.activation(act) if self.activation else act


# for uniform hidden activation across all layers (except last):

#class MLP(Module):

#    def __init__(self, nin, nouts, hidden_activation='tanh'):
        sz = [nin] + nouts
        self.layers = []
        for i in range(len(nouts)):
            is_last = (i == len(nouts) - 1)
            act = 'linear' if is_last else hidden_activation
            self.layers.append(Layer(sz[i], sz[i+1], activation=act))


