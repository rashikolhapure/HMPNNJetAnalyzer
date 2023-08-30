import sys
import os
from itertools import (
    combinations,
)
import re

import numpy as np
import matplotlib.pyplot as plt

from ..io.saver import (
    Unpickle,
    Pickle,
)
from ..genutils import (
    print_events,
    dir_ext_count,
)
from .classes import (
    Inference,
)
from ..plotter import Plotter


class Operator(object):
    """
    Attributes:
    operation: A callable object representing the operation to be performed on the dataset.

    Methods:
    iterator(original_data): Iterate over the dataset and apply the operation.
    """

    def __init__(self):
        pass

    def iterator(self, original_data):
        """
        Args:
        original_data: A dictionary representing the original dataset.

        Returns:
        A dictionary representing the updated dataset.
        """
        assert self.operation, " Set an operation first!"
        operated_array = np.zeros(
            original_data["X"].shape,
            dtype="float64",
        )
        for (
            i,
            array,
        ) in enumerate(original_data["X"]):
            operated_array[i] = self.operation(array)
        original_data["X"] = operated_array
        return original_data


class Deform(Operator):
    def __init__(self, *args, **kwargs):
        """soft=True=>local=False"""
        super().__init__()
        self.deform_radius = kwargs.get("deform_radius")
        self.num_pixels = kwargs.get("num_pixels")
        self.deform_scale = kwargs.get("deform_scale")
        self.soft = kwargs.get("soft", False)
        self.soft_scale = kwargs.get("soft_scale")
        self.hard_scale = kwargs.get("hard_scale")
        self.operation = None
        self.relevance = {
            "hard": ("hard_scale"),
            "soft_deform": (
                "num_pixels",
                "soft_scale",
                "deform_scale",
            ),
            "local_soft": (
                "num_pixels",
                "soft_scale",
                "deform_scale",
            ),
            "hard_deform": (
                "hard_scale",
                "num_pixels",
                "deform_scale",
            ),
            "local": (
                "num_pixels",
                "deform_radius",
                "deform_scale",
            ),
        }
        self.ht_local = False

    def unconstrained_deform(self, array, indices):
        deformed = np.zeros(
            array.shape,
            dtype="float64",
        )
        x, y, _ = np.where(array)
        try:
            def_inds = np.random.choice(
                indices,
                self.num_pixels,
                replace=False,
            )
        except ValueError:
            def_inds = indices
        undef_inds = np.array(
            [int(i) for i in range(len(x)) if i not in def_inds]
        )
        x_def, y_def = (
            x[def_inds],
            y[def_inds],
        )
        try:
            (
                x_undef,
                y_undef,
            ) = (
                x[undef_inds],
                y[undef_inds],
            )
        except IndexError:
            pass
        else:
            deformed[
                x_undef,
                y_undef,
            ] = array[
                x_undef,
                y_undef,
            ]
        x_deformed = [
            np.random.randint(
                i - self.deform_scale,
                i + self.deform_scale,
            )
            % array.shape[0]
            for i in x_def
        ]
        y_deformed = [
            np.random.randint(
                i - self.deform_scale,
                i + self.deform_scale,
            )
            % array.shape[1]
            for i in y_def
        ]
        deformed[
            x_deformed,
            y_deformed,
        ] = (
            deformed[
                x_deformed,
                y_deformed,
            ]
            + array[x_def, y_def]
        )
        if "debug" in sys.argv:
            debug_logger(
                array,
                deformed,
                x_def,
                y_def,
            )
        return deformed

    def constrained_deform(
        self,
        def_inds,
        allowed_indices,
    ):
        pass

    def local(self, array):
        x, y, _ = np.where(array)
        center_ind = np.where(array == np.max(array))
        inds = []
        count = 0
        for i in range(len(x)):
            if (
                np.sqrt(
                    (x[i] - center_ind[0]) ** 2 + (y[i] - center_ind[1]) ** 2
                )
                <= self.deform_radius + 0.2
            ):
                inds.append(i)
        ht_sum = np.sum(
            array[
                x[inds],
                y[inds],
            ]
        )
        deformed = self.unconstrained_deform(array, inds)
        return deformed

    def local_soft(self, array):
        x, y, _ = np.where(array)
        center_ind = np.where(array == np.argmax(array))
        inds = []
        count = 0
        for i in range(len(x)):
            if (
                np.sqrt(
                    (x[i] - x[center_ind]) ** 2 + (y[i] - y[center_ind]) ** 2
                )
                <= self.deform_radius + 0.2
            ):
                inds.append(i)
        print(inds)
        ht_sum = np.sum(
            array[
                x[inds],
                y[inds],
            ]
        )
        soft_inds = [
            i for i in inds if array[x[i], y[i], 0] <= self.soft_scale * ht_sum
        ]
        print(inds, soft_inds)
        sys.exit()
        try:
            def_inds = np.random.choice(
                soft_inds,
                self.num_pixels,
                replace=False,
            )
        except ValueError:
            def_inds = soft_inds
        deformed = self.unconstrained_deform(array, inds)

    def soft_deform(self, array):
        if self.num_pixels == 0:
            return array
        ht_sum = np.sum(array)
        if "debug" in sys.argv:
            print(
                "ht_sum: ",
                ht_sum,
                "cut_off :",
                self.soft_scale * ht_sum,
                array[np.where(array)],
            )
        x, y, _ = np.where(array)
        deformed = np.zeros(
            array.shape,
            dtype="float64",
        )
        soft_inds = [
            i
            for i in range(len(x))
            if array[x, y, 0][i] <= self.soft_scale * ht_sum
        ]
        try:
            def_inds = np.random.choice(
                soft_inds,
                self.num_pixels,
                replace=False,
            )
        except ValueError:
            def_inds = soft_inds
        undef_inds = np.array(
            [int(i) for i in range(len(x)) if i not in def_inds]
        )
        x_def, y_def = (
            x[def_inds],
            y[def_inds],
        )
        try:
            (
                x_undef,
                y_undef,
            ) = (
                x[undef_inds],
                y[undef_inds],
            )
        except IndexError:
            pass
        else:
            deformed[
                x_undef,
                y_undef,
            ] = array[
                x_undef,
                y_undef,
            ]
        x_deformed = [
            np.random.randint(
                i - self.deform_scale,
                i + self.deform_scale,
            )
            % array.shape[0]
            for i in x_def
        ]
        y_deformed = [
            np.random.randint(
                i - self.deform_scale,
                i + self.deform_scale,
            )
            % array.shape[1]
            for i in y_def
        ]
        deformed[
            x_deformed,
            y_deformed,
        ] = (
            deformed[
                x_deformed,
                y_deformed,
            ]
            + array[x_def, y_def]
        )
        return deformed

    def hard_deform(self, array):
        if self.num_pixels == 0:
            return array
        ht_sum = np.sum(array)
        # if "debug" in sys.argv: print (array[np.where(array)])
        x, y, _ = np.where(array)
        hard_indices = [
            i
            for i in range(len(x))
            if array[x, y, 0][i] >= self.hard_scale * ht_sum
        ]
        if "debug" in sys.argv:
            print(self.hard_scale * ht_sum)
        deformed = self.unconstrained_deform(
            array,
            hard_indices,
        )
        return deformed

    def hard(self, array):
        ht_sum = np.sum(array)
        # if "debug" in sys.argv: print (array[np.where(array)])
        x, y, _ = np.where(array)
        hard_array = np.zeros(
            array.shape,
            dtype="float64",
        )
        hard_indices = [
            i
            for i in range(len(x))
            if array[x, y, 0][i] >= self.hard_scale * ht_sum
        ]
        if "debug" in sys.argv:
            print(self.hard_scale * ht_sum)
        hard_array[
            x[hard_indices],
            y[hard_indices],
        ] = array[
            x[hard_indices],
            y[hard_indices],
        ]
        return hard_array


def debug_logger(
    array,
    deformed,
    x_def,
    y_def,
):
    print(
        "deform values: \n",
        array[x_def, y_def, 0],
        "\nvalues in same position of deformed array: \n",
        deformed[x_def, y_def, 0],
    )
    print("values adjacent to deform values\n")
    print(
        "west: \n    original \n    ",
        array[
            x_def - 1,
            y_def,
            0,
        ],
        x_def - 1,
        y_def,
        "\n    deformed :\n    ",
        deformed[
            x_def - 1,
            y_def,
            0,
        ],
        x_def - 1,
        y_def,
    )
    print(
        "east: \n    original \n    ",
        array[
            x_def + 1,
            y_def,
            0,
        ],
        x_def + 1,
        y_def,
        "\n    deformed :\n    ",
        deformed[
            x_def + 1,
            y_def,
            0,
        ],
        x_def + 1,
        y_def,
    )
    print(
        "south: \n    original \n    ",
        array[
            x_def,
            y_def - 1,
            0,
        ],
        x_def,
        y_def - 1,
        "\n    deformed :\n    ",
        deformed[
            x_def,
            y_def - 1,
            0,
        ],
        x_def,
        y_def - 1,
    )
    print(
        "north: \n    original \n    ",
        array[
            x_def,
            y_def + 1,
            0,
        ],
        x_def,
        y_def + 1,
        "\n    deformed :\n",
        deformed[
            x_def,
            y_def + 1,
            0,
        ],
        x_def,
        y_def + 1,
    )
    print(
        "north-east: \n    original \n    ",
        array[
            x_def + 1,
            y_def + 1,
            0,
        ],
        x_def,
        y_def + 1,
        "\n    deformed :\n    ",
        deformed[
            x_def + 1,
            y_def + 1,
            0,
        ],
    )
    print(
        "north-west: \n    original \n    ",
        array[
            x_def - 1,
            y_def + 1,
            0,
        ],
        x_def - 1,
        y_def + 1,
        "\n    deformed :\n    ",
        deformed[
            x_def - 1,
            y_def + 1,
            0,
        ],
        x_def - 1,
        y_def + 1,
    )
    print(
        "south-east: \n    original \n    ",
        array[
            x_def + 1,
            y_def - 1,
            0,
        ],
        x_def + 1,
        y_def - 1,
        "\n    deformed :\n    ",
        deformed[
            x_def + 1,
            y_def - 1,
            0,
        ],
        x_def + 1,
        y_def - 1,
    )
    print(
        "south-west: \n    original \n    ",
        array[
            x_def - 1,
            y_def - 1,
            0,
        ],
        x_def - 1,
        y_def - 1,
        "\n    deformed :\n    ",
        deformed[
            x_def - 1,
            y_def - 1,
            0,
        ],
        x_def - 1,
        y_def - 1,
    )
    check_plot(array, deformed)


def check_plot(array, deformed):
    print("Generating temp debugging plots...")
    p = Plotter(projection="image")
    p.Image(array)
    p.save_fig("temp_b")
    p.Image(deformed)
    p.save_fig("temp_a")
    sys.exit()


def operator(
    run_name,
    operation_name=None,
    operation_class=None,
    parameter_name=None,
    parameter_values=None,
    roc_plot=False,
    **kwargs
):
    """dir_name where the data and model_checkpoints are stored. operation_name is a class method belonging to an initiated operation_class,
    the class method must depend on the parameter_name, iterates the Inference.predict function over all parameter_values
    """
    assert parameter_name in dir(operation_class) and operation_name in dir(
        operation_class
    )
    assert parameter_name in operation_class.relevance[operation_name]
    if roc_plot:
        assert "roc_plot_values" in kwargs
        roc_plot_values = kwargs.pop("roc_plot_values")
    opt_args = {}
    for item in kwargs:
        assert item in dir(operation_class)
        operation_class.__setattr__(
            item,
            kwargs.get(item),
        )
        print(
            item,
            operation_class.__getattribute__(item),
        )
        opt_args[item] = operation_class.__getattribute__(item)
    opt_args["Operation"] = type(operation_class).__name__
    opt_args["name"] = operation_name
    opt_args["x_parameter"] = parameter_name
    I = Inference(run_name)
    tag = type(operation_class).__name__ + "_" + operation_name + "_"
    filename = "acc_dict_" + tag

    I.operation_name = operation_name
    operation_class.__setattr__(
        "operation",
        operation_class.__getattribute__(operation_name),
    )
    # print (dir(operation_class))
    for (
        i,
        parameter,
    ) in enumerate(parameter_values):
        operation_class.__setattr__(
            parameter_name,
            parameter,
        )
        print(
            operation_name,
            parameter_name,
            operation_class.__getattribute__(parameter_name),
        )
        if "debug" in sys.argv:
            operation_class.iterator(I.unoperated_data)
        if parameter in roc_plot_values:
            roc_plot = True
        else:
            roc_plot = False
        temp_dict = I.predict(
            operation=operation_class.iterator,
            roc_plot=roc_plot,
        )
        if roc_plot:
            roc_data = temp_dict.pop("roc")
            roc_data["specs"] = {
                "parameter_name": parameter_name,
                "parameter_value": parameter,
                "operation_name": type(operation_class).__name__,
            }
            for item in kwargs:
                roc_data["specs"][item] = kwargs.get(item)
            Pickle(
                roc_data,
                "roc_" + tag + "_" + str(parameter) + ".h",
                save_path=I.inference_data,
            )
        if i == 0:
            # break
            acc_dict = {item: [temp_dict[item][-1]] for item in temp_dict}
            print(acc_dict)
            # sys.exit()
        else:
            for _ in temp_dict:
                acc_dict[_].append(temp_dict[_][-1])

    acc_dict["opt_args"] = opt_args
    acc_dict["x_axis"] = parameter_values
    acc_dict["x_name"] = parameter_name
    acc_dict["classes"] = tuple(list(I.class_names) + ["combined"])
    # if "acc_dict_"+tag+".h" in os.listdir(I.inference_data):
    count = len(
        [
            item
            for item in os.listdir(I.inference_data)
            if item.startswith(filename) and item.endswith(".h")
        ]
    )
    Pickle(
        acc_dict,
        filename + str(count) + ".h",
        save_path=I.inference_data,
    )
    for item in acc_dict:
        print(
            item,
            acc_dict[item],
        )
    return os.path.join(
        I.inference_data,
        filename + str(count) + ".h",
    )
