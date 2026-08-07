"""Numerical gradient checker for the micrograd engine."""
# good for finding bugs in gradient descent

from engine import Value


def numerical_grad(loss_fn, params, h=1e-5):
    """
    Compute gradients of loss_fn with respect to each parameter
    via central finite differences.

    loss_fn: a zero-argument callable returning the current loss (a Value or float).
    params:  a list of Value objects whose .data will be perturbed.
    h:       perturbation size. 1e-5 is the standard default — small enough for
             accuracy, large enough to avoid floating-point noise.

    Returns: a list of floats, same length as params, the numerical estimate
             of d(loss)/d(p) for each p.
    """
    grads = []
    for p in params:
        original = p.data

        p.data = original + h
        loss_plus = loss_fn()
        if isinstance(loss_plus, Value):
            loss_plus = loss_plus.data

        p.data = original - h
        loss_minus = loss_fn()
        if isinstance(loss_minus, Value):
            loss_minus = loss_minus.data

        p.data = original  # restore

        grads.append((loss_plus - loss_minus) / (2 * h))
    return grads