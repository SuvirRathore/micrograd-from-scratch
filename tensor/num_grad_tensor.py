import numpy as np
def numerical_grad_tensor(loss_fn, params, h=1e-5):
    """
    params: list of Tensor objects. Returns list of arrays, same shapes as
    params, containing numerical estimate of d(loss)/d(each element).
    """
    grads = []
    for p in params:
        grad = np.zeros_like(p.data)
        # Iterate over every element of p.data
        for idx in np.ndindex(p.data.shape):
            original = p.data[idx]

            p.data[idx] = original + h
            loss_plus = loss_fn()
            if hasattr(loss_plus, 'data'):
                loss_plus = float(loss_plus.data)

            p.data[idx] = original - h
            loss_minus = loss_fn()
            if hasattr(loss_minus, 'data'):
                loss_minus = float(loss_minus.data)

            p.data[idx] = original  # restore
            grad[idx] = (loss_plus - loss_minus) / (2 * h)
        grads.append(grad)
    return grads