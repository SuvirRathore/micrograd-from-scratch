import numpy as np
from tensor import Tensor

# Test 1: addition forward and backward
a = Tensor([1.0, 2.0, 3.0])
b = Tensor([4.0, 5.0, 6.0])
c = a + b
s = (c * c).sum() if hasattr(c, 'sum') else None  # we don't have sum yet
# For now, force a scalar via picking element 0 via summation-by-hand
# Simplest scalar reduction is multiply against ones via a known scalar.
# Easier: just check forward for now, defer backward until we have sum.



assert np.allclose(c.data, [5.0, 7.0, 9.0])
print("Test 1 (forward) passed")




# Test 2: addition + sum scalar reduction
a = Tensor([1.0, 2.0, 3.0])
b = Tensor([4.0, 5.0, 6.0])
loss = (a + b).sum()
loss.backward()
assert np.allclose(a.grad, [1.0, 1.0, 1.0])
assert np.allclose(b.grad, [1.0, 1.0, 1.0])
print("Test 2 passed")


# Test 3: multiplication backward
a = Tensor([1.0, 2.0, 3.0])
b = Tensor([4.0, 5.0, 6.0])
loss = (a * b).sum()
loss.backward()
# d(loss)/d(a[i]) = b[i], so a.grad should be [4, 5, 6]
# d(loss)/d(b[i]) = a[i], so b.grad should be [1, 2, 3]
assert np.allclose(a.grad, [4.0, 5.0, 6.0]), f"a.grad = {a.grad}"
assert np.allclose(b.grad, [1.0, 2.0, 3.0]), f"b.grad = {b.grad}"
print("Test 3 passed")

# Test 4: truediv
a = Tensor([1.0, 2.0, 3.0])
b = Tensor([4.0, 5.0, 6.0])
loss = (a / b).sum()
loss.backward()
# d(loss)/d(a[i]) = 1/b[i], so a.grad should be [1/4, 1/5, 1/6]
# d(loss)/d(b[i]) = -a[i]/b[i]^2, so b.grad should be [-1/16, -2/25, -3/36]
assert np.allclose(a.grad, [1/4, 1/5, 1/6]), f"a.grad = {a.grad}"
assert np.allclose(b.grad, [-1/16, -2/25, -3/36]), f"b.grad = {b.grad}"
print("Test 4 passed")

# Test 5: tanh
a = Tensor([1.0, 2.0, 3.0])
loss = (a.tanh()).sum()
loss.backward()
# d(loss)/d(a[i]) = 1-tanh(a[i])^2
assert np.allclose(a.grad, [1-np.tanh(1.0)**2, 1 - np.tanh(2.0)**2 ,1 - np.tanh(3.0)**2]), f"a.grad = {a.grad}"
print("Test 5 passed")

# Test: ReLU on positive and negative inputs
a = Tensor([2.0, -1.0, 0.5, -3.0])
loss = a.relu().sum()
loss.backward()
# ReLU gradient: 1 where input > 0, 0 elsewhere
assert np.allclose(a.grad, [1.0, 0.0, 1.0, 0.0]), f"a.grad = {a.grad}"
print("ReLU test passed")

import numpy as np
from tensor import Tensor
from num_grad_tensor import numerical_grad_tensor

def check(name, loss_fn, params):
    """Run analytical and numerical gradients, compare, report."""
    # Zero grads
    for p in params:
        p.grad = np.zeros_like(p.data)
    loss = loss_fn()
    loss.backward()
    analytical = [p.grad.copy() for p in params]
    numerical = numerical_grad_tensor(loss_fn, params)
    for ai, ni, p in zip(analytical, numerical, params):
        max_err = np.max(np.abs(ai - ni))
        status = "OK" if max_err < 1e-5 else "FAIL"
        print(f"  {name} ({p.data.shape}): max abs err {max_err:.2e}  [{status}]")

# Case 1: scalar + vector
a = Tensor([1.0, 2.0, 3.0])
b = Tensor(5.0)
check("scalar + vector, add", lambda: (a + b).sum(), [a, b])
check("scalar + vector, mul", lambda: (a * b).sum(), [a, b])

# Case 2: vector + scalar (the __radd__ case)
a = Tensor([1.0, 2.0, 3.0])
check("2 + vector", lambda: (2 + a).sum(), [a])

# Case 3: matrix + row vector (the broadcasting bread-and-butter)
A = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
v = Tensor([10.0, 20.0, 30.0])
check("matrix + row vector", lambda: (A + v).sum(), [A, v])

# Case 4: matrix + column vector
A = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
v = Tensor([[10.0], [20.0]])
check("matrix + column vector", lambda: (A + v).sum(), [A, v])

# Case 5: stress test — composition of broadcasts
a = Tensor([1.0, 2.0])
b = Tensor([[10.0], [20.0], [30.0]])
check("composed broadcast", lambda: ((a * b) + b).sum(), [a, b])

import numpy as np
from tensor import Tensor

# Forward should work — NumPy broadcasts 2 + [1,2,3] without complaint
a = Tensor([1.0, 2.0, 3.0])
out = 2 + a   # triggers __radd__, which wraps 2 as Tensor(2), then __add__

print("Forward:")
print(f"  a.data       = {a.data}")
print(f"  out.data     = {out.data}")
print(f"  a.data.shape = {a.data.shape}")
print(f"  out.data.shape = {out.data.shape}")

# Now try backward — this is where it might go wrong
loss = out.sum()
loss.backward()

print("\nBackward:")
print(f"  a.grad        = {a.grad}")
print(f"  a.grad.shape  = {a.grad.shape}")

# Find the wrapped scalar 2 — it's in out._prev
wrapped_two = [c for c in out._prev if c is not a][0]
print(f"  wrapped 2.data  = {wrapped_two.data}")
print(f"  wrapped 2.data.shape = {wrapped_two.data.shape}")
print(f"  wrapped 2.grad        = {wrapped_two.grad}")
print(f"  wrapped 2.grad.shape  = {wrapped_two.grad.shape}")

a = Tensor([1.0, 2.0, 3.0])
check("scalar pow", lambda: (a ** 2).sum(), [a])
# d/da (a²) = 2a, so a.grad should be [2, 4, 6]

a = Tensor([1.0, 2.0, 3.0])
check("mean", lambda: a.mean() * 10, [a])   # *10 to make gradients ~0.33, not 0.0333
# d/da (mean(a)) = 1/N = 1/3, so a.grad should be [3.33, 3.33, 3.33] after multiplying by 10

a = Tensor([1.0, 5.0, 3.0, 2.0])
check("max", lambda: a.max() * a.max(), [a])
# d/da (a_max²) = 2*a_max at the argmax position only
# a_max = 5, so a.grad should be [0, 10, 0, 0]

A = Tensor([[1.0, 2.0], [3.0, 4.0]])      # shape (2, 2)
B = Tensor([[5.0, 6.0], [7.0, 8.0]])      # shape (2, 2)
check("matmul square", lambda: (A @ B).sum(), [A, B])

# Non-square to make sure shape handling is right
A = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])    # (2, 3)
B = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (3, 2)
check("matmul non-square", lambda: (A @ B).sum(), [A, B])

# Composed with element-wise op
A = Tensor([[1.0, 2.0], [3.0, 4.0]])
B = Tensor([[0.5, 0.5], [0.5, 0.5]])
check("matmul composed", lambda: ((A @ B) * 2).sum(), [A, B])

