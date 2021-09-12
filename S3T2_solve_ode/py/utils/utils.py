import numpy as np


def call_counter(func):
    def helper(*args, **kwargs):
        helper.calls += 1
        return func(*args, **kwargs)
    helper.calls = 0
    return helper


def get_accuracy(x, y, eps=1e-17, axis=1):
    x = np.array(x)
    y = np.array(y)
    if len(x.shape) == 1:
        return -np.log10(np.clip(np.abs(x - y), eps, np.inf))
    else:
        return -np.log10(np.clip(np.linalg.norm(x - y, axis=axis), eps, np.inf))
