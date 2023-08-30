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
