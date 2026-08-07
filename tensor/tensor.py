import numpy as np

def unbroadcast(grad, target_shape):
    """
    Reduce `grad` (shape: the broadcasted/output shape) back to `target_shape`
    by summing over the axes that were broadcast.
    """
    # Step 1: handle dimensions added on the left (target has fewer dims than grad)
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)
    # Step 2: handle size-1 dimensions that were expanded to >1
    for axis, (gdim, tdim) in enumerate(zip(grad.shape, target_shape)):
        if tdim == 1 and gdim > 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad

class Tensor:
    def __init__(self, data, _children=(), _op=''):
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        # No more shape assertion — NumPy broadcasts forward

        def _backward():
            self.grad += unbroadcast(out.grad, self.data.shape)
            other.grad += unbroadcast(out.grad, other.data.shape) # see below

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    #def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        assert self.data.shape == other.data.shape, \
            f"shape mismatch: {self.data.shape} vs {other.data.shape}"
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    #def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        assert self.data.shape == other.data.shape, \
            f"shape mismatch: {self.data.shape} vs {other.data.shape}"
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "exponent must be scalar"
        out = Tensor(self.data ** other, (self,), f'**{other}')

        def _backward():
            # d(x**n)/dx = n * x**(n-1), element-wise via NumPy
            self.grad += unbroadcast(other * self.data ** (other - 1) * out.grad, self.data.shape)

        out._backward = _backward
        return out

    def sum(self):
        out = Tensor(self.data.sum(), (self,), 'sum')

        def _backward():
            self.grad += out.grad * np.ones_like(self.data)

        out._backward = _backward
        return out

    def mean(self):
        out = Tensor(self.data.mean(), (self,), 'mean')
        N = self.data.size

        def _backward():
            self.grad += (out.grad / N) * np.ones_like(self.data)

        out._backward = _backward
        return out

    def max(self):
        max_idx = np.unravel_index(np.argmax(self.data), self.data.shape)
        # np.argmax returns a flat index for multi-dim arrays.
        # np.unravel_index converts it back to a multi-dim index. 
        # Use data[max_idx] to fetch the value and to index into the mask
        out = Tensor(self.data[max_idx], (self,), 'max')

        def _backward():
            mask = np.zeros_like(self.data)
            mask[max_idx] = 1.0
            self.grad += mask * out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), '@')

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * (-1)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __truediv__(self, other):
        return self * other ** (-1)

    def __rtruediv__(self, other):
        return other * self ** -1

    def tanh(self):
        out = Tensor(np.tanh(self.data), (self,), 'tanh')

        def _backward():
            self.grad += (1 - out.data**2) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out


    def backward(self):
        assert self.data.size == 1, "backward() only on scalar Tensors"
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)

        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()


# why do we need this unbroadcast, essentially its do with shapes. if a has shape (2,3)
# and b is a scalar, then for c = a + b, For a[i]: dc[k]/da[i] = 1 if k==i else 0, so by
# chain rule dL/da[i] = dL/dc[i] * 1 = out.grad[i]. So a.grad += out.grad.
# Element-wise, same shape.
# For b: this is where it gets interesting. b appears in every c[i],
# so it has three paths to L.
# Chain rule: dL/db = sum over i of (dL/dc[i] * dc[i]/db)
# = sum over i of (out.grad[i] * 1) = out.grad.sum().
# The single scalar b accumulates gradient from every position it was broadcast into.

# Worked example 1. a = Tensor([[1, 2, 3], [4, 5, 6]]) (shape (2, 3)), b = Tensor([10, 20, 30])
# (shape (3,)), c = a + b (shape (2, 3)).
# Suppose out.grad = [[1, 1, 1], [1, 1, 1]] (from a downstream .sum().backward()).
# For a: unbroadcast(out.grad, a.data.shape) = unbroadcast([[1,1,1],[1,1,1]], (2,3)).
# Target shape (2,3) matches grad shape (2,3) — step 1 does nothing, step 2 does nothing.
# Returns [[1,1,1],[1,1,1]]. Then a.grad += [[1,1,1],[1,1,1]].
# Each element of a contributed once to its corresponding output, so each gets gradient 1.
# For b: unbroadcast(out.grad, b.data.shape) = unbroadcast([[1,1,1],[1,1,1]], (3,)).
# Step 1: grad has 2 dims, target has 1, sum axis 0: [[1,1,1],[1,1,1]].sum(axis=0) = [2, 2, 2].
# Step 2: dims now match, nothing to do. Returns [2, 2, 2]. Then b.grad += [2, 2, 2].
# Each element of b was broadcast across two rows of a, so it accumulates gradient from both

# Worked example 2
# A = Tensor([[1.0, 2.0, 3.0],
#             [4.0, 5.0, 6.0]])     # shape (2, 3)
# v = Tensor([[10.0],
#             [20.0]])              # shape (2, 1)   — note: explicit size-1 second dim
# c = A + v                          # shape (2, 3)

# Forward broadcasting: v is stretched along its size-1 second axis to become
# effectively [[10,10,10],[20,20,20]], then added element-wise to A.
# Output c.data = [[11, 12, 13], [24, 25, 26]], shape (2, 3).
# Now suppose out.grad = [[1, 1, 1], [1, 1, 1]]
# (which is what you'd get from c.sum().backward()).
# Trace unbroadcast(out.grad, v.data.shape) = unbroadcast([[1,1,1],[1,1,1]], (2, 1)):
#
# Step 1: grad.ndim = 2, len(target_shape) = 2. Equal — loop doesn't run. Nothing happens.
# Grad shape still (2, 3).
# Step 2: enumerate axes of grad.shape paired with target_shape.
#
# Axis 0: gdim = 2, tdim = 2. tdim isn't 1 — skip.
# Axis 1: gdim = 3, tdim = 1. tdim == 1 and gdim > 1 —
# this axis was stretched in the forward pass. Run grad = grad.sum(axis=1, keepdims=True).
#
# Before: [[1, 1, 1], [1, 1, 1]], shape (2, 3).
# After: [[3], [3]], shape (2, 1).
# The three columns of each row got summed into a single column,
# and keepdims=True preserved the second axis at size 1.
# Return [[3], [3]], shape (2, 1). Matches v's shape.
#
# Then v.grad += [[3], [3]]. Each element of v was broadcast across three columns,
# so its accumulated gradient is 3 * 1 = 3 (sum of three contributions of out.grad = 1 each).
# Compare to A: unbroadcast(out.grad, A.data.shape) = unbroadcast([[1,1,1],[1,1,1]], (2,3)).
# Step 1: ndims match, skip. Step 2: every axis has gdim == tdim, never enters the if.
# Returns unchanged. A.grad += [[1,1,1],[1,1,1]] —
# each element of A contributed exactly once to its corresponding output.