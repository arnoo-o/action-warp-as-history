def _sum(value, axis):
    try:
        return value.sum(dim=axis)
    except TypeError:
        return value.sum(axis=axis)


def _mean(value, axis=None):
    try:
        return value.mean() if axis is None else value.mean(dim=axis)
    except TypeError:
        return value.mean() if axis is None else value.mean(axis=axis)


def valid_element_normalized_loss(loss_map, valid_mask=None):
    """Reduce over valid elements only; supports both PyTorch and NumPy arrays."""
    flattened = loss_map.reshape(loss_map.shape[0], -1)
    if valid_mask is None:
        return _mean(_mean(flattened, axis=1))
    expanded_valid = loss_map * 0 + valid_mask
    numerator = _sum((loss_map * expanded_valid).reshape(loss_map.shape[0], -1), axis=1)
    denominator = _sum(expanded_valid.reshape(loss_map.shape[0], -1), axis=1)
    if hasattr(denominator, "clamp_min"):
        denominator = denominator.clamp_min(1.0)
    else:
        import numpy as np

        denominator = np.maximum(denominator, 1.0)
    return _mean(numerator / denominator)
