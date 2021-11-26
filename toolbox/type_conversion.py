import numpy as np
from torch import from_numpy, Tensor


def np2torch(numpy_array, requires_grad=True):
    """
    Converts a numpy array to torch tensor that requires grad
    :param numpy_array:
    :type numpy_array:
    :return:
    :rtype:
    """
    torch_tensor = from_numpy(numpy_array.astype(float))
    torch_tensor.requires_grad = requires_grad

    return torch_tensor


def ensure_torch(item, requires_grad=True):
    """
    A function that ensures that input item is a torch tensor.
    :param item:
    :type item:
    :return:
    :rtype:
    """
    if isinstance(item, Tensor):
        item.requires_grad = requires_grad
        return item
    elif isinstance(item, np.ndarray):
        return np2torch(item)
    elif isinstance(item, (float, int)):
        return np2torch(np.array([item]))
    else:
        raise TypeError(f'Invalid input type: "{type(item)}"')
