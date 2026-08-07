from engine import Value
from nn import MLP
import random
import matplotlib.pyplot as plt
import math

xs1 = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys1 = [1.0, -1.0, -1.0, 1.0]
# linear wins here, then tanh then relu

xs2 = [[0, 0], [0, 1], [1, 0], [1, 1]]
ys2 = [0, 1, 1, 0]
#


def train(mlp, xs, ys, n_iters=100, lr=0.01):
    losses = []
    for k in range(n_iters):
        ypred = [mlp(x) for x in xs]
        loss = sum((ysi - ypredi) ** 2 for ysi, ypredi in zip(ys, ypred))

        mlp.zero_grad()
        loss.backward()
        for p in mlp.parameters():
            p.data -= lr * p.grad

        losses.append(loss.data)  # <-- extract the float and store
    return losses


def train_rmsprop(mlp, xs, ys, n_iters=100, lr=0.05, eps=1e-8):
    losses = []
    accum = [0.0] * len(mlp.parameters())  # running sum of g² per param

    for k in range(n_iters):
        ypred = [mlp(x) for x in xs]
        loss = sum((ysi - ypredi) ** 2 for ysi, ypredi in zip(ys, ypred))

        mlp.zero_grad()
        loss.backward()

        for i, p in enumerate(mlp.parameters()):
            #accum[i] += p.grad ** 2 for adagrad
            decay = 0.9 #for rmsprop
            accum[i] = decay * accum[i] + (1 - decay) * p.grad ** 2  # decay ~ 0.9
            effective_lr = lr / (math.sqrt(accum[i]) + eps)
            p.data -= effective_lr * p.grad

        losses.append(loss.data)
    return losses


def train_adam(mlp, xs, ys, n_iters=100, lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8):
    losses = []
    m = [0.0] * len(mlp.parameters())
    v = [0.0] * len(mlp.parameters())

    for k in range(n_iters):
        ypred = [mlp(x) for x in xs]
        loss = sum((ysi - ypredi) ** 2 for ysi, ypredi in zip(ys, ypred))

        mlp.zero_grad()
        loss.backward()

        for i, p in enumerate(mlp.parameters()):
            m[i] = beta1 * m[i] + (1 - beta1) * p.grad
            v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
            effective_lr = lr / (math.sqrt(v[i]) + eps)
            p.data -= effective_lr * m[i]

        losses.append(loss.data)
    return losses



xs_xor = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
ys_xor = [0.0, 1.0, 1.0, 0.0]

random.seed(42)
mlp = MLP(2, [4, 4, 1], hidden_activations='blend')


# Extract converged alpha values
def sigmoid_scalar(x):
    return 1 / (1 + math.exp(-x))


random.seed(44)
mlp_lin = MLP(2, [4, 4, 1], hidden_activations='linear')
losses_lin = train_adam(mlp_lin,xs_xor,ys_xor)

random.seed(44)
mlp_tanh = MLP(2, [4, 4, 1], hidden_activations='tanh')
losses_tanh = train_adam(mlp_tanh,xs_xor,ys_xor)
losses_tanh2 = train_rmsprop(mlp_tanh,xs_xor,ys_xor)

random.seed(44)
mlp_relu = MLP(2, [4, 4, 1], hidden_activations='relu')
losses_relu = train_adam(mlp_relu,xs_xor,ys_xor)

random.seed(44)
mlp_mix = MLP(2, [4, 4, 1], hidden_activations=['tanh', 'relu'])
losses_mix = train_adam(mlp_mix,xs_xor,ys_xor)
losses_mix2 = train_rmsprop(mlp_mix,xs_xor,ys_xor)


# hidden layers get 'blend', output layer gets 'linear' as before
random.seed(44)
mlp_blend = MLP(2, [4, 4, 1], hidden_activations='blend')
losses_blend = train_adam(mlp_blend,xs_xor,ys_xor)

alphas = []
for layer in mlp_blend.layers:
    for neuron in layer.neurons:
        if neuron.activation == 'blend':
            alphas.append(sigmoid_scalar(neuron.alpha_raw.data))

plt.plot(losses_lin, label='lin')
plt.plot(losses_tanh, label='tanh')
#plt.plot(losses_tanh2, label='tanh rms')
plt.plot(losses_relu, label='ReLU')
plt.plot(losses_mix, label='Mix')
#plt.plot(losses_mix2, label='Mix rms')
plt.plot(losses_blend, label='Blend')
plt.yscale('log')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.legend()
plt.show()

print(f"Converged alphas (8 hidden neurons): {[f'{a:.3f}' for a in alphas]}")
print(f"Mean alpha: {sum(alphas)/len(alphas):.3f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(losses_blend)
ax1.set_yscale('log')
ax1.set_xlabel('iteration')
ax1.set_ylabel('loss')
ax1.set_title('Training loss with learnable blend')

ax2.hist(alphas, bins=10, range=(0, 1))
ax2.set_xlabel('converged sigmoid(alpha_raw)')
ax2.set_ylabel('count')
ax2.set_title('1.0 = pure tanh,  0.0 = pure ReLU,  0.5 = equal blend')
ax2.set_xlim(0, 1)
plt.tight_layout()
plt.show()


print("alpha_raw values (pre-sigmoid):",
      [f'{n.alpha_raw.data:.3f}' for layer in mlp_blend.layers
       for n in layer.neurons if n.activation == 'blend'])
