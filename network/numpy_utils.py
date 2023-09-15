from collections import (
    namedtuple,
)

import numpy as np


InputState = namedtuple(
    "InputState",
    [
        "name",
        "shape",
        "index",
        "network_input_index",
    ],
)
IndexKey = namedtuple(
    "IndexKey",
    [
        "class_name",
        "start",
        "end",
        "data_type",
    ],
)


def array_shuffle(**kwargs):
    """
    Shuffle multiple NumPy arrays or sequences in the same order.

    This function takes multiple NumPy arrays or sequences as keyword
    arguments and returns a dictionary with shuffled values of those arguments:
    It returns a dictionary with the same keys as the input keyword arguments,
    where each value is a shuffled version of the corresponding input array or
    sequence. The shuffle algorithm used here is Fisher-Yates / Knuth's
    variant 2a.

    Parameters:
    **kwargs (dict): Keyword arguments where the keys are names of arrays or
                     sequences and the values are the arrays or sequences to
                     be shuffled.

    Returns:
    dict: A dictionary where the keys are the same as the input keyword
          arguments, and the values are shuffled versions of the corresponding
          input arrays.
    """
    if "all" in kwargs:
        kwargs = kwargs.get("all")  # X.shape[0]==Y.shape[0]
    args = []
    key_ind = {}
    for i, item in enumerate(kwargs):
        args.append(kwargs.get(item))
        key_ind[item] = i
    ind_map = np.arange(args[0].shape[0])
    # np.random.seed(seed=random_seed)
    np.random.shuffle(ind_map)
    args = list(args)
    for i in range(len(args)):
        args[i] = args[i][ind_map]
        # print (args[i][:10])
    ret_dict = {item: args[key_ind[item]] for item in kwargs}
    ret_dict["ind_map"] = ind_map
    return ret_dict


if __name__ == "__main__":
    a = np.arange(10)
    b = a + 20
    print(a, b)
    print(array_shuffle(a=a, b=b))
