import tensorflow.keras as keras
import numpy as np

from itertools import product
import sys

"""
This function takes a learning rate and the name of an optimizer as input and returns an instance of the specified 
optimizer with the given learning rate. It uses a dictionary to map optimizer names to their corresponding integer values, 
and then uses a series of conditional statements to return the appropriate optimizer instance. The **kwargs parameter 
is used to pass any additional keyword arguments that may be needed for the specific optimizer.
"""


def opt(
    learning_rate,
    optimizer_name,
    **kwargs,
):
    opt_dict = {
        "Adam": 0,
        "Adamax": 1,
        "Nadam": 2,
        "Adadelta": 3,
        "Adagrad": 4,
        "RMSprop": 5,
        "SGD": 6,
        "Adadelta": 7,
    }
    print(
        optimizer_name,
        " initialized with learning rate ",
        learning_rate,
    )
    if opt_dict[optimizer_name] == 0:
        return keras.optimizers.Adam(
            lr=learning_rate, **kwargs
        )
    elif opt_dict[optimizer_name] == 1:
        return keras.optimizers.Adamax(
            lr=learning_rate, **kwargs
        )
    elif opt_dict[optimizer_name] == 2:
        return keras.optimizers.Nadam(
            lr=learning_rate, **kwargs
        )
    elif opt_dict[optimizer_name] == 3:
        return (
            keras.optimizers.Adadelta(
                lr=learning_rate,
                **kwargs,
            )
        )
    elif opt_dict[optimizer_name] == 4:
        return (
            keras.optimizers.Adagrad(
                lr=learning_rate,
                **kwargs,
            )
        )
    elif opt_dict[optimizer_name] == 5:
        return (
            keras.optimizers.RMSprop(
                lr=learning_rate,
                **kwargs,
            )
        )
    elif opt_dict[optimizer_name] == 6:
        return keras.optimizers.SGD(
            lr=learning_rate, **kwargs
        )
    elif opt_dict[optimizer_name] == 7:
        return (
            keras.optimizers.Adadelta(
                lr=learning_rate,
                **kwargs,
            )
        )
    else:
        raise ValueError(
            "Wrong optimizer choice ..."
        )


"""
This function takes one or more numpy arrays as input, shuffles the first axis (i.e., the rows), 
and returns a tuple of shuffled arrays along with an index map that can be used to reverse the shuffling 
later on. This function is useful for shuffling input data and corresponding labels to ensure that they 
are randomized and correspond to each other.
"""


def array_shuffle(
    *args,
):  # X.shape[0]==Y.shape[0]
    ind_map = np.arange(
        args[0].shape[0]
    )
    # np.random.seed(seed=random_seed)
    np.random.shuffle(ind_map)
    args = list(args)
    for i in range(len(args)):
        args[i] = args[i][ind_map]
        # print (args[i][:10])
    return tuple(args), ind_map


"""
This function takes a dictionary of hyperparameters as input and returns a list of all possible combinations of 
hyperparameters. It uses the product function from the itertools module to generate all possible combinations of 
hyperparameters, and returns a list of dictionaries where each dictionary contains a single combination of hyperparameters.
"""


def get_hyper_opt_kwargs(**kwargs):
    keys = []
    values = []
    for key, val in kwargs.items():
        keys.append(
            key
        ), values.append(val)
    prod = product(
        *[
            kwargs.get(key)
            for key in kwargs
        ]
    )
    return_list = []
    for item in prod:
        return_list.append(
            {
                key: val
                for key, val in zip(
                    keys, item
                )
            }
        )
    return return_list
