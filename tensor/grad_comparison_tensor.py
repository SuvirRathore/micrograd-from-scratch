import numpy as np
from tensor import Tensor
from num_grad_tensor import numerical_grad_tensor

# Quadratic-ish loss with two parameters
a = Tensor([1.0, 2.0, 3.0])
b = Tensor([0.5, -1.0, 0.2])

def loss_fn():
    return ((a + b) * (a + b)).sum()

# Analytical
a.grad = np.zeros_like(a.data)
b.grad = np.zeros_like(b.data)
loss = loss_fn()
loss.backward()
analytical = [a.grad.copy(), b.grad.copy()]
#analytical = [a.grad, b.grad]   # both entries reference the live arrays
# ... later, numerical_grad_tensor mutates p.data temporarily
# ... but also, if anything else modifies a.grad later, analytical[0] silently changes too

# Numerical
numerical = numerical_grad_tensor(loss_fn, [a, b])

# Compare
for ai, ni, name in zip(analytical, numerical, ['a', 'b']):
    max_err = np.max(np.abs(ai - ni))
    print(f"{name}: max abs err = {max_err:.2e}")
    assert max_err < 1e-6
print("Gradient check passed")