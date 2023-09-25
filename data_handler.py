import os
import sys
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.model_selection import (
    train_test_split,
)

from io.saver import (
    Unpickle,
    RunIO,
)
from genutils import (
    pool_splitter,
)


def load_data(
    classes: List[str],
    length: Optional[int] = None,
    suffix: Optional[str] = None,
    test_train_split: float = 0.25,
    input_keys: List[str] = ["high_level"],
    return_array: bool = False,
    function: Optional[callable] = None,
    run_io: bool = False,
    **kwargs: Dict[str, any]
) -> Union[Dict[str, Dict[str, Union[np.ndarray, List[np.ndarray]]]], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load data from files or I/O, preprocess it, and split it into training and
    validation sets.

    Args:
        classes (list): List of class names.
        length (int): Maximum length of the data (optional).
        suffix (str): Suffix for file paths (optional).
        test_train_split (float): Fraction of data to use for validation
            (default is 0.25).
        input_keys (list): List of input data keys
            (default is ["high_level"]).
        return_array (bool): If True, return data as arrays; if False, return
            as dictionaries (default is False).
        function (function): A function to apply to the data (optional).
        run_io (bool): If True, use I/O for loading data; if False, load from
            files (default is False).
        **kwargs: Additional keyword arguments.

    Returns:
        dict or tuple: A dictionary containing training and validation data,
            or a tuple of arrays if return_array is True.
    """
    count = 0
    X = [[] for _ in input_keys]
    for item in classes:
        if not run_io:
            if function is None:
                if "bin_name" in kwargs:
                    folder = "/" + kwargs.get("bin_name")
                else:
                    folder = "/all"
                events = Unpickle(
                    item + ".h",
                    load_path="./processed_events/" + suffix + folder,
                )
            else:
                events = pool_splitter(
                    function,
                    Unpickle(
                        item + ".h",
                        load_path="./temp_data",
                    ),
                )
        else:
            r = RunIO(
                item,
                kwargs.get("data_tag"),
                mode="r",
            )
            events = r.load_events()
        for (
            i,
            input_key,
        ) in enumerate(input_keys):
            if input_key != "high_level":
                X[i] = np.expand_dims(
                    events[input_key][:length],
                    -1,
                )
                if kwargs.get(
                    "log",
                    False,
                ):
                    print(
                        "Calculating log of " + input_key + "...",
                        np.min(X[i][np.where(X[i])]),
                        np.max(X[i][np.where(X[i])]),
                    )
                    X[i][np.where(X[i])] = np.log(X[i][np.where(X[i])])
                    print(
                        "New: ",
                        np.min(X[i][np.where(X[i])]),
                        np.max(X[i][np.where(X[i])]),
                    )
            else:
                X[i] = events[input_key][:length]
        Y = np.zeros(
            (
                len(X[0]),
                len(classes),
            )
        )
        Y[:, count] = 1.0
        print(type(X), Y.shape)
        train_index = int(len(X) * (1 - test_train_split))
        if count == 0:
            X_all, Y_all = [item[:] for item in X], Y[:]
        else:
            X_all, Y_all = [
                np.concatenate(
                    (
                        prev_item,
                        item[:],
                    ),
                    axis=0,
                )
                for prev_item, item in zip(X_all, X)
            ], np.concatenate(
                (
                    Y_all,
                    Y[:],
                ),
                axis=0,
            )
        print(
            item,
            Y[-10:],
            len(X),
        )
        count += 1
    if len(input_keys) == 1:
        X_all = X_all[0]
        assert X_all.shape[0] == Y_all.shape[0]
        (
            X_train,
            X_val,
            Y_train,
            Y_val,
        ) = train_test_split(
            X_all,
            Y_all,
            shuffle=True,
            random_state=12,
            test_size=0.25,
        )
    else:
        x_length = len(X_all)
        combined = X_all + []
        combined.append(Y_all)
        if "debug" in sys.argv:
            print(
                "combined:",
                combined[-1][:10],
                combined[-1][10:],
            )
        combined = list(
            train_test_split(
                *combined, shuffle=True, random_state=12, test_size=0.25
            )
        )
        X_train, X_val = (
            [],
            [],
        )
        for i in range(len(combined) - 2):
            print(
                type(combined[i]),
                combined[i].shape,
            )
            if i % 2 == 0:
                X_train.append(combined[i])
            else:
                X_val.append(combined[i])
        Y_train = combined[-2]
        Y_val = combined[-1]
    # if "debug" in sys.argv:
    shape_print(X_train, Y_train), shape_print(X_val, Y_val)
    train_dict = {
        "X": X_train,
        "Y": Y_train,
    }
    test_dict = {
        "X": X_val,
        "Y": Y_val,
    }
    if return_array:
        return (
            X_train,
            Y_train,
            X_val,
            Y_val,
        )
    else:
        return {
            "train": train_dict,
            "val": test_dict,
            "classes": classes,
        }


def shape_print(
    X: np.ndarray or List[np.ndarray],
    Y: np.ndarray
) -> None:
    """
    Print the shapes and some sample values of input and output data arrays.

    Args:
        X (numpy.ndarray or list of numpy.ndarray): Input data arrays or a
            list of input data arrays.
        Y (numpy.ndarray): Output data array.

    Returns:
        None
    """
    if isinstance(X, np.ndarray):
        print("X:", X.shape)
    else:
        [
            print(
                "\nX" + str(i) + " :",
                item.shape,
            )
            for i, item in enumerate(X)
        ]
    print(
        "Y: ",
        Y.shape,
        "\nY head: ",
        Y[:5],
        "\nY tail:",
        Y[-5:],
    )
    return


if __name__ == "__main__":
    classes = [
        "h_inv_jj_weak",
        "z_inv_jj_qcd",
    ]
    (
        X_train,
        Y_train,
        X_test,
        Y_test,
    ) = load_data(
        classes,
        input_keys=["tower_image"],
        suffix="low_res_tower_jet_phi",
        return_array=True,
        length=30000,
    )
