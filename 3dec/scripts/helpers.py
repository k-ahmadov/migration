import numpy as np

def apply_function_to_iterable(
        func,
        iterable,
        count=-1,
        dtype=np.float64
):
    result_iter = (func(item) for item in iterable)
    return np.fromiter(result_iter, dtype=dtype, count=count)
