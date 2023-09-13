import tensorflow.keras as keras
import numpy as np

from itertools import product
import sys


def opt(
    learning_rate,
    optimizer_name,
    **kwargs,
):
    """
    Initialize a Keras optimizer with a specified learning rate.

    Args:
        learning_rate (float): The learning rate for the optimizer.
        optimizer_name (str): The name of the optimizer to be initialized.
        **kwargs: Additional keyword arguments that can be passed to the optimizer constructor.

    Returns:
        keras.optimizers.Optimizer: A Keras optimizer object with the specified learning rate.

    Raises:
        ValueError: If an invalid optimizer name is provided.

    Examples:
        1. Initialize an Adam optimizer with a learning rate of 0.001:
        ```python
        optimizer = opt(0.001, 'Adam')
        ```

        2. Initialize a Nadam optimizer with a learning rate of 0.0001 and a custom parameter `beta_1` set to 0.9:
        ```python
        optimizer = opt(0.0001, 'Nadam', beta_1=0.9)
        ```

        3. Initialize an Adadelta optimizer with a learning rate of 1.0:
        ```python
        optimizer = opt(1.0, 'Adadelta')
        ```

        4. Initialize an SGD optimizer with a learning rate of 0.01 and momentum of 0.9:
        ```python
        optimizer = opt(0.01, 'SGD', momentum=0.9)
        ```

    Note:
        Supported optimizer names include 'Adam', 'Adamax', 'Nadam', 'Adadelta', 'Adagrad', 'RMSprop', 'SGD', and 'Adadelta'.
    """
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
            lr=learning_rate,
            **kwargs,
        )
    elif opt_dict[optimizer_name] == 1:
        return keras.optimizers.Adamax(
            lr=learning_rate,
            **kwargs,
        )
    elif opt_dict[optimizer_name] == 2:
        return keras.optimizers.Nadam(
            lr=learning_rate,
            **kwargs,
        )
    elif opt_dict[optimizer_name] == 3:
        return keras.optimizers.Adadelta(
            lr=learning_rate,
            **kwargs,
        )
    elif opt_dict[optimizer_name] == 4:
        return keras.optimizers.Adagrad(
            lr=learning_rate,
            **kwargs,
        )
    elif opt_dict[optimizer_name] == 5:
        return keras.optimizers.RMSprop(
            lr=learning_rate,
            **kwargs,
        )
    elif opt_dict[optimizer_name] == 6:
        return keras.optimizers.SGD(
            lr=learning_rate,
            **kwargs,
        )
    elif opt_dict[optimizer_name] == 7:
        return keras.optimizers.Adadelta(
            lr=learning_rate,
            **kwargs,
        )
    else:
        raise ValueError("Wrong optimizer choice ...")


def array_shuffle(
    *args,
):
    """
    Shuffle multiple NumPy arrays in the same random order.

    Args:
        *args: Multiple NumPy arrays to be shuffled. Each array should have the same number of rows.

    Returns:
        tuple: A tuple containing the shuffled arrays in the same order as the input, and an index mapping showing the shuffle order.

    Example:
        Suppose you have two NumPy arrays X and Y with the same number of rows, and you want to shuffle them in the same random order:

        ```python
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        Y = np.array([0, 1, 0])

        shuffled_X, shuffled_Y, index_map = array_shuffle(X, Y)

        # The arrays X and Y have been shuffled in the same random order, and the index mapping is provided.
        ```
    """
    # X.shape[0]==Y.shape[0]
    ind_map = np.arange(args[0].shape[0])
    # np.random.seed(seed=random_seed)
    np.random.shuffle(ind_map)
    args = list(args)
    for i in range(len(args)):
        args[i] = args[i][ind_map]
        # print (args[i][:10])
    return (
        tuple(args),
        ind_map,
    )


def get_hyper_opt_kwargs(
    **kwargs,
):
    """
    Generate a list of dictionaries with hyperparameter combinations for hyperparameter optimization.

    Args:
        **kwargs: Keyword arguments where keys are hyperparameter names, and values are lists of hyperparameter values.

    Returns:
        list: A list of dictionaries, each representing a unique combination of hyperparameters.

    Example:
        Suppose you have a set of hyperparameters and their possible values in dictionaries:

        ```python
        hyperparameters = {
            'learning_rate': [0.001, 0.01],
            'batch_size': [32, 64],
            'hidden_units': [64, 128, 256],
        }

        hyperparameter_combinations = get_hyper_opt_kwargs(**hyperparameters)

        # The function will generate a list of dictionaries, each containing a unique combination of hyperparameters:
        # [{'learning_rate': 0.001, 'batch_size': 32, 'hidden_units': 64},
        #  {'learning_rate': 0.001, 'batch_size': 32, 'hidden_units': 128},
        #  ...
        #  {'learning_rate': 0.01, 'batch_size': 64, 'hidden_units': 256}]
        ```
    """
    keys = []
    values = []
    for (
        key,
        val,
    ) in kwargs.items():
        keys.append(key), values.append(val)
    prod = product(*[kwargs.get(key) for key in kwargs])
    return_list = []
    for item in prod:
        return_list.append(
            {
                key: val
                for key, val in zip(
                    keys,
                    item,
                )
            }
        )
    return return_list
