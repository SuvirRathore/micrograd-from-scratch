import time, random
from nn import MLP
import matplotlib.pyplot as plt

# Same small dataset, just measuring time not learning
xs = [[1.0, 2.0, 3.0]] * 4
ys = [1.0, -1.0, -1.0, 1.0]

sizes = [[4, 4, 1], [16, 16, 1], [32, 32, 1], [64, 64, 1], [128, 128, 1]]

param_counts = []
times = []
for arch in sizes:
    random.seed(0)
    mlp = MLP(3, arch)
    n_params = len(mlp.parameters())

    t0 = time.time()
    for _ in range(5):  # 5 iterations, average
        ypred = [mlp(x) for x in xs]
        loss = sum((yi - ypi) ** 2 for yi, ypi in zip(ys, ypred))
        mlp.zero_grad()
        loss.backward()
        for p in mlp.parameters():
            p.data -= 0.01 * p.grad
    elapsed = (time.time() - t0) / 5
    param_counts.append(n_params)
    times.append(elapsed * 1000)

    #print(f"arch {str(arch):16s}  params {n_params:6d}  time/iter {elapsed * 1000:8.1f} ms  param/time/iter {n_params/elapsed * 10**-5:8.1f}")

plt.plot(param_counts, times, 'o-')
plt.xlabel('number of parameters')
plt.ylabel('time per iteration (ms)')
plt.title('Scalar engine: time vs parameter count')
#plt.show()

import sys
print(f"recursion limit: {sys.getrecursionlimit()}")

# A single wide neuron builds a long addition chain via sum()
from engine import Value
import random

for width in [100, 500, 900, 1100, 2000, 2900, 3000]:
    random.seed(0)
    try:
        x = [Value(1.0) for _ in range(width)]
        w = [Value(random.uniform(-1, 1)) for _ in range(width)]
        act = sum(wi * xi for wi, xi in zip(w, x))   # chain of `width` additions
        act.backward()
        print(f"width {width:5d}: OK (depth survived)")
    except RecursionError:
        print(f"width {width:5d}: RecursionError")

import numpy as np, time

W = np.random.randn(1000, 1000)
x = np.random.randn(1000)

t0 = time.time()
for _ in range(100):
    y = W @ x          # one matmul = the entire layer's forward pass
elapsed = (time.time() - t0) / 100
print(f"NumPy (1000x1000) @ (1000): {elapsed*1e6:.1f} microseconds per forward")

import time, random
import numpy as np
from engine import Value
from nn import Layer
from tensor.tensor import Tensor   # adjust import to your path

def time_scalar(nin, nout, reps=50):
    random.seed(0)
    layer = Layer(nin, nout)
    x = [Value(random.uniform(-1, 1)) for _ in range(nin)]
    # WARM-UP: one untimed iteration to absorb one-time costs
    out = layer(x); loss = sum(out); loss.backward()
    t0 = time.time()
    for _ in range(reps):
        for p in layer.parameters():
            p.grad = 0.0
        out = layer(x)
        loss = sum(out)
        loss.backward()
    return (time.time() - t0) / reps


def time_tensor(nin, nout, reps=50):
    np.random.seed(0)
    W = Tensor(np.random.randn(nout, nin))
    b = Tensor(np.random.randn(nout))
    x = Tensor(np.random.randn(nin))
    t0 = time.time()
    for _ in range(reps):
        W.grad = np.zeros_like(W.data)
        b.grad = np.zeros_like(b.data)
        x.grad = np.zeros_like(x.data)
        out = (W @ x + b).tanh()    # one matmul + broadcast-add + tanh
        loss = out.sum()
        loss.backward()
    return (time.time() - t0) / reps


print(f"{'nin=nout':>10} {'scalar (ms)':>14} {'tensor (ms)':>14} {'ratio':>10}")
for n in [10, 40, 50, 100, 200, 500]:
    s = time_scalar(n, n)
    t = time_tensor(n, n)
    print(f"{n:>10} {s*1000:>14.2f} {t*1000:>14.3f} {s/t:>10.0f}")