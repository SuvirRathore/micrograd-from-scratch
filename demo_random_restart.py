import random, math
import matplotlib.pyplot as plt
from nn import MLP
import numpy as np


xs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
ys = [0.0, 1.0, 1.0, 0.0]

def sigmoid_scalar(x):
    return 1 / (1 + math.exp(-x))

def train_adam0(mlp, xs, ys, n_iters=200, lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8):
    m = [0.0]*len(mlp.parameters()); v = [0.0]*len(mlp.parameters())
    for k in range(n_iters):
        ypred = [mlp(x) for x in xs]
        loss = sum((yi - ypi)**2 for yi, ypi in zip(ys, ypred))
        mlp.zero_grad(); loss.backward()
        for i, p in enumerate(mlp.parameters()):
            m[i] = beta1*m[i] + (1-beta1)*p.grad
            v[i] = beta2*v[i] + (1-beta2)*p.grad**2
            p.data -= lr / (math.sqrt(v[i]) + eps) * m[i]
    return loss.data

import random, math
import matplotlib.pyplot as plt
from nn import MLP

xs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
ys = [0.0, 1.0, 1.0, 0.0]

def sigmoid_scalar(x):
    return 1 / (1 + math.exp(-x))

def train_adam(mlp, xs, ys, n_iters=200, lr=0.05,
               beta1=0.9, beta2=0.999, eps=1e-8, track_neurons=None):
    """Adam with bias correction. If track_neurons is given (a list of
    blend Neurons), also records each one's sigmoid(alpha_raw) per iteration."""
    m = [0.0] * len(mlp.parameters())
    v = [0.0] * len(mlp.parameters())
    losses = []
    trajectories = [[] for _ in track_neurons] if track_neurons else None

    for k in range(n_iters):
        ypred = [mlp(x) for x in xs]
        loss = sum((yi - ypi)**2 for yi, ypi in zip(ys, ypred))
        mlp.zero_grad()
        loss.backward()

        for i, p in enumerate(mlp.parameters()):
            m[i] = beta1 * m[i] + (1 - beta1) * p.grad
            v[i] = beta2 * v[i] + (1 - beta2) * p.grad**2
            m_hat = m[i] / (1 - beta1 ** (k + 1))   # bias correction
            v_hat = v[i] / (1 - beta2 ** (k + 1))
            p.data -= lr / (math.sqrt(v_hat) + eps) * m_hat

        losses.append(loss.data)
        if track_neurons:
            for j, neuron in enumerate(track_neurons):
                trajectories[j].append(sigmoid_scalar(neuron.alpha_raw.data))

    return losses, trajectories

n_seeds = 15
final_losses = []
all_alphas = []
for seed in range(n_seeds):
    random.seed(seed)
    mlp = MLP(2, [4, 4, 1], hidden_activations='blend')
    final_loss = train_adam0(mlp, xs, ys)
    final_losses.append(final_loss)
    for layer in mlp.layers:
        for neuron in layer.neurons:
            if neuron.activation == 'blend':
                all_alphas.append(sigmoid_scalar(neuron.alpha_raw.data))

# Random-restart view: how often does training reach a good solution?
print(f"Final losses across {n_seeds} seeds:")
print(f"  best:  {min(final_losses):.2e}")
print(f"  worst: {max(final_losses):.2e}")
print(f"  median: {sorted(final_losses)[n_seeds//2]:.2e}")
stuck = sum(1 for l in final_losses if l > 0.1)
print(f"  seeds stuck (loss > 0.1): {stuck}/{n_seeds}")



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
log_bins = np.logspace(np.log10(min(final_losses)), np.log10(max(final_losses)), 15)
ax1.hist(final_losses, bins=log_bins)
ax1.set_xscale('log')
ax1.set_xlabel('final loss'); ax1.set_ylabel('count')
ax1.set_title(f'Random restarts: final loss across {n_seeds} seeds')

ax2.hist(all_alphas, bins=15, range=(0, 1))
ax2.set_xlabel('converged sigmoid(alpha_raw)'); ax2.set_ylabel('count')
ax2.set_title(f'Pooled alphas ({len(all_alphas)} neurons, {n_seeds} seeds)')
ax2.set_xlim(0, 1)
plt.tight_layout(); plt.show()

random.seed(0)
mlp = MLP(2, [4, 4, 1], hidden_activations='blend')
blend_neurons = [n for layer in mlp.layers for n in layer.neurons
                 if n.activation == 'blend']

losses, trajectories = train_adam(mlp, xs, ys, track_neurons=blend_neurons)

# Trajectory plot
plt.figure(figsize=(8, 5))
for j, traj in enumerate(trajectories):
    plt.plot(traj, label=f'neuron {j}')
plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
plt.ylim(0, 1)
plt.xlim(0)
plt.xlabel('iteration'); plt.ylabel('sigmoid(alpha_raw)')
plt.title('Per-neuron alpha trajectories during training (Adam, bias-corrected)')
plt.legend(fontsize=8)
plt.show()