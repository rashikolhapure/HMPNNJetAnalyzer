import os
import sys

import numpy as np
from sklearn.model_selection import (
    train_test_split,
)
import matplotlib.pyplot as plt

from hep_ml.io.saver import (
    Unpickle,
    RunIO,
)
from hep_ml.genutils import (
    pool_splitter,
)
from hep_ml.plotter import Plotter


"""
This code defines a function called pad_values which takes in three arguments: events, target_shapes, and value.

events is a dictionary containing numpy arrays of different shapes, and target_shapes is also a dictionary containing 
the desired shapes for each corresponding array in events.

The function pads each array in events with zeros to match the desired shape in target_shapes. 
The padding is done along the second and third dimensions of the arrays, and the value of the 
padding is set to value, which defaults to zero.

If the command line argument "plot" is present, the function also creates a 
plot of the original and padded arrays for the 10th element of each array in events. 
The plots are saved as an eps file named "zero_pad".
"""


def pad_values(
    events, target_shapes, value=0
):
    for (
        key,
        new_shape,
    ) in target_shapes.items():
        old_val = events.pop(key)
        old_shape = old_val.shape
        if "plot" in sys.argv:
            p = Plotter(
                projection="subplots"
            )
            fig, axes = plt.subplots(
                ncols=2,
                figsize=(20, 10),
            )
            p.fig, p.axes = (
                fig,
                axes[0],
            )
            p.Image(old_val[10])
        if old_shape[1] < new_shape[0]:
            diff = (
                new_shape[0]
                - old_shape[1]
            )
            x_new = np.full(
                (
                    old_shape[0],
                    diff,
                    old_shape[2],
                ),
                value,
            )
            old_val = np.concatenate(
                (
                    x_new[
                        :,
                        : int(
                            diff / 2
                        ),
                    ],
                    old_val,
                    x_new[
                        :,
                        int(
                            diff / 2
                        ) :,
                    ],
                ),
                axis=1,
            )
        old_shape = old_val.shape
        if old_shape[2] < new_shape[1]:
            diff = (
                new_shape[1]
                - old_shape[2]
            )
            x_new = np.full(
                (
                    old_shape[0],
                    old_shape[1],
                    diff,
                ),
                value,
            )
            # print (x_new.shape,old_val.shape)
            old_val = np.concatenate(
                (
                    x_new[
                        :,
                        :,
                        : int(
                            diff / 2
                        ),
                    ],
                    old_val,
                    x_new[
                        :,
                        :,
                        int(
                            diff / 2
                        ) :,
                    ],
                ),
                axis=2,
            )
        if "plot" in sys.argv:
            p.axes = axes[1]
            p.Image(old_val[10])
            p.save_fig(
                "zero_pad",
                extension="eps",
            )
        # print (old_val.shape)
        events[key] = old_val
    return events


"""
This function load_data loads the data for a machine learning model. 
The data is stored in different files for different classes, and the function 
loads the data from these files and creates a training and validation dataset. 
The function allows for different types of input data and different pre-processing techniques.
"""


def load_data(
    classes,
    length=30000,
    preprocess_tag=None,
    test_train_split=0.25,
    input_keys=["high_level"],
    return_array=False,
    function=None,
    run_io=False,
    **kwargs
):
    count = 0
    normalize = kwargs.get(
        "normalize", False
    )
    target_shapes = kwargs.get(
        "target_shapes", {}
    )
    if "sim_in" in input_keys:
        assert (
            "sim_central_values"
            in kwargs
        )
        random = kwargs.get(
            "random", True
        )
        sim_central_values = (
            kwargs.get(
                "sim_central_values"
            )
        )
    X = [[] for _ in input_keys]
    for item in classes:
        if not run_io:
            if function is None:
                if (
                    "bin_name"
                    in kwargs
                ):
                    folder = (
                        "/"
                        + kwargs.get(
                            "bin_name"
                        )
                    )
                else:
                    folder = "/all"
                events = Unpickle(
                    item + ".h",
                    load_path="./processed_events/"
                    + preprocess_tag
                    + folder,
                )
                load_path = os.path.abs(
                    "./processed_events/"
                    + preprocess_tag
                    + folder
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
        if "network_in" in input_keys:
            events[
                "network_in"
            ] = Unpickle(
                item + ".h",
                load_path="./processed_events/network_out",
            )[
                "network_out"
            ]
        if target_shapes:
            events = pad_values(
                events, target_shapes
            )
        for i, input_key in enumerate(
            input_keys
        ):
            if (
                input_key
                == "tower_image"
            ):
                X[i] = np.expand_dims(
                    events[input_key][
                        :length
                    ],
                    -1,
                )
                if kwargs.get(
                    "log", False
                ):
                    print(
                        "Calculating log of "
                        + input_key
                        + "...",
                        np.min(
                            X[i][
                                np.where(
                                    X[
                                        i
                                    ]
                                )
                            ]
                        ),
                        np.max(
                            X[i][
                                np.where(
                                    X[
                                        i
                                    ]
                                )
                            ]
                        ),
                    )
                    X[i][
                        np.where(X[i])
                    ] = np.log(
                        X[i][
                            np.where(
                                X[i]
                            )
                        ]
                    )
                    print(
                        "New: ",
                        np.min(
                            X[i][
                                np.where(
                                    X[
                                        i
                                    ]
                                )
                            ]
                        ),
                        np.max(
                            X[i][
                                np.where(
                                    X[
                                        i
                                    ]
                                )
                            ]
                        ),
                    )
            elif input_key == "sim_in":
                central_values = kwargs.get(
                    "sim_central_values"
                )
                if random:
                    X[
                        i
                    ] = np.random.normal(
                        loc=sim_central_values[
                            item
                        ][
                            "mean"
                        ],
                        scale=sim_central_values[
                            item
                        ][
                            "scale"
                        ],
                        size=tuple(
                            [length]
                            + list(
                                sim_central_values[
                                    "shape"
                                ]
                            )
                        ),
                    )
                else:
                    X[i] = np.full(
                        tuple(
                            [length]
                            + list(
                                sim_central_values[
                                    "shape"
                                ]
                            )
                        ),
                        sim_central_values[
                            item
                        ][
                            "mean"
                        ],
                    )
                print(
                    "sim_in shape: ",
                    X[i].shape,
                )
            elif (
                input_key == "num_bins"
            ):
                X[i] = np.zeros(
                    (
                        events[
                            "tower_image"
                        ][
                            :length
                        ].shape[
                            0
                        ],
                        1,
                    )
                )
                for (
                    ind,
                    item,
                ) in enumerate(
                    events[
                        "tower_image"
                    ][:length]
                ):
                    X[i][ind, 0] = len(
                        np.where(
                            events[
                                "tower_image"
                            ][ind]
                        )[0]
                    )
                    # print ("bin_in shape: ",X[i].shape,X[i][:10],np.where(events["tower_image"][ind]),len(np.where(events["tower_image"][ind])[0]))
            else:
                X[i] = events[
                    input_key
                ][:length]
            if normalize:
                print(
                    "Normalizing:\n min:",
                    np.min(X[i]),
                    " max: ",
                    np.max(X[i]),
                )
                for ind in range(
                    len(X[i])
                ):
                    X[i][ind] = X[i][
                        ind
                    ] / np.sqrt(
                        np.sum(
                            X[i][ind]
                        )
                    )
                print(
                    "\nNormalized:\n min:",
                    np.min(X[i]),
                    " max: ",
                    np.max(X[i]),
                )
            print(
                input_key, X[i].shape
            )
        Y = np.zeros(
            (len(X[0]), len(classes))
        )
        Y[:, count] = 1.0
        print(type(X), Y.shape)
        train_index = int(
            len(X)
            * (1 - test_train_split)
        )
        if count == 0:
            X_all, Y_all = [
                item[:] for item in X
            ], Y[:]
        else:
            X_all, Y_all = [
                np.concatenate(
                    (
                        prev_item,
                        item[:],
                    ),
                    axis=0,
                )
                for prev_item, item in zip(
                    X_all, X
                )
            ], np.concatenate(
                (Y_all, Y[:]), axis=0
            )
        print(item, Y[-10:], len(X))
        count += 1
    if len(input_keys) == 1:
        X_all = X_all[0]
        assert (
            X_all.shape[0]
            == Y_all.shape[0]
        )
        indices = np.arange(
            X_all.shape[0]
        )
        (
            X_train,
            X_val,
            Y_train,
            Y_val,
            train_indices,
            test_indices,
        ) = train_test_split(
            X_all,
            Y_all,
            indices,
            shuffle=True,
            random_state=12,
            test_size=0.25,
        )
    else:
        x_length = len(X_all)
        indices = np.arange(x_length)
        combined = X_all + []
        # combined.append(indices)
        combined.append(Y_all)
        if "debug" in sys.argv:
            print(
                "combined:",
                combined[-1][:10],
                combined[-1][10:],
            )
        combined = list(
            train_test_split(
                *combined,
                shuffle=True,
                random_state=12,
                test_size=0.25
            )
        )
        X_train, X_val = [], []
        for i in range(
            len(combined) - 2
        ):
            print(
                type(combined[i]),
                combined[i].shape,
            )
            if i % 2 == 0:
                X_train.append(
                    combined[i]
                )
            else:
                X_val.append(
                    combined[i]
                )
        train_indices = combined[-4]
        test_indices = combined[-3]
        Y_train = combined[-2]
        Y_val = combined[-1]
    # if "debug" in sys.argv:
    shape_print(
        X_train, Y_train
    ), shape_print(X_val, Y_val)
    train_dict = {
        "X": X_train,
        "Y": Y_train,
        "indices": train_indices,
    }
    test_dict = {
        "X": X_val,
        "Y": Y_val,
        "indices": test_indices,
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
            "path": load_path,
        }


def shape_print(X, Y):
    """
    This function takes input X and Y and prints their shapes along with the first 5 and last 5 elements of Y.

    Args:
    X: Input data of type numpy array or a list of numpy arrays.
    Y: Output data of type numpy array.

    Returns:
    None
    """
    if type(X) == np.ndarray:
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
        target_shapes={
            "tower_image": (48, 64)
        },
    )
