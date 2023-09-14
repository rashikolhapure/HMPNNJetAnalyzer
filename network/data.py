import os
import sys

import numpy as np

from ..genutils import (
    check_dir,
)
from ..hep.data import (
    PreProcessedEvents,
)
from ..genutils import (
    merge_flat_dict,
    print_events,
)
from ..io.saver import (
    RunIO,
    Pickle,
    Unpickle,
    dict_hdf,
)
from .numpy_utils import (
    InputState,
    IndexKey,
)
from .keras_utils import (
    array_shuffle,
)
from .numpy_utils import (
    array_shuffle as nu_shuffle,
)


class ModelData(object):
    """
    Base class for handling data related to machine learning models.

    This class provides a foundation for managing data used in machine learning models. It includes attributes and methods to store, load, and preprocess data.

    Attributes:
        - _prefix_path (str): The prefix path where data is stored.
        - mode (str): The mode of operation, "w" for write, "r" for read.
        - save_as (str): The format in which data is saved (e.g., "numpy_array").
        - total_length (int): The total length of the data.
        - test_split (float): The split ratio for test data.
        - model_type (str): The type of the model.
        - preprocess_tag (str): A tag for data preprocessing.
        - class_names (list): List of class names.
        - input_states (list): List of input states.
        - save (bool): Flag to control saving data.
        - run_tag (str): A tag for the run.
        - tag_path (str): The path for run tags.
        - preprocessed_path (str): The path for preprocessed data.
        - run_name (str): The name of the run.
        - run_path (str): The path for the run.
        - data_save_path (str): The path for saved data.
        - model_checkpoints_path (str): The path for model checkpoints.
        - train_length (int): The length of the training data.
        - validation_length (int): The length of the validation data.
        - validation_tag (str): A tag for validation data.
        - train_data (dict): Training data dictionary.
        - val_data (dict): Validation data dictionary.
        - data (None): Placeholder for data.
        - index_dict (dict): Dictionary to store indices.
        - shuffled_index_dict (dict): Dictionary to store shuffled indices.
        - check (bool): Flag for data checking.
    """

    def __init__(self, *args, **kwargs):
        self._prefix_path = check_dir("./network_runs")
        self.mode = kwargs.get("mode", "w")
        self.save_as = kwargs.get(
            "save_as",
            "numpy_array",
        )
        compulsory_kwargs = {
            "input_states",
            "preprocess_tag",
            "class_names",
        }
        if self.mode == "w":
            assert compulsory_kwargs.issubset(set(kwargs.keys()))
        if "test_split" not in kwargs.keys():
            assert {
                "train_length",
                "validation_length",
            }.issubset(set(kwargs.keys()))
            self.load_data = self._load_data
        else:
            print("Load all data first! ")
            self.load_data = self._load_all_data
        self.total_length = kwargs.get(
            "total_length",
            100,
        )
        self.test_split = kwargs.get(
            "test_split",
            0.2,
        )
        self.model_type = kwargs.get(
            "model_type",
            "model",
        )
        self.preprocess_tag = kwargs.get("preprocess_tag")
        self.class_names = kwargs.get("class_names")
        self.input_states = kwargs.get("input_states")
        self.save = kwargs.get("save", False)
        if self.save:
            self.run_tag = kwargs.get(
                "run_tag",
                "no_tag",
            )
            self.tag_path = check_dir(
                os.path.join(
                    self._prefix_path,
                    self.run_tag,
                )
            )
            self.preprocessed_path = check_dir(
                os.path.join(
                    self.tag_path,
                    self.preprocess_tag,
                )
            )
            self.run_name = kwargs.get(
                "run_name",
                self.model_type + "_" + self.preprocess_tag,
            )
            self.run_path = check_dir(
                os.path.join(
                    self.preprocessed_path,
                    self.run_name,
                )
            )
            self.data_save_path = check_dir(
                os.path.join(
                    self.run_path,
                    "data",
                )
            )
            self.model_checkpoints_path = check_dir(
                os.path.join(
                    self.run_path,
                    "model_checkpoints",
                )
            )
        self.train_length = kwargs.get(
            "train_length",
            None,
        )
        self.validation_length = kwargs.get(
            "validation_length",
            None,
        )
        self.validation_tag = kwargs.get(
            "validation_tag",
            "",
        )
        self.train_data = {}
        self.val_data = {}
        self.data = None
        self.index_dict = {
            "train": {},
            "val": {},
        }
        self.shuffled_index_dict = {
            "train": {},
            "val": {},
        }
        self.check = False

    def _load_all_data(self):
        """
        Load all the data for each class and split it into training and validation sets.

        This method loads data for each class, splits it into training and validation sets, and prepares the data for training a machine learning model.

        Returns:
            Tuple: A tuple containing training and validation data, each represented as a dictionary with 'X' and 'Y' keys.
        """
        return_dict = {run_name: {} for run_name in self.class_names}
        train_dict = {}
        val_dict = {}
        temp_indices = {
            "split_point": {run_name: None for run_name in self.class_names},
            "class_interval": {
                run_name: None for run_name in self.class_names
            },
        }
        temp_indices["class_indices"] = {
            run_name: None for run_name in self.class_names
        }
        count = 0
        class_index = 0
        for run_name in self.class_names:
            in_data = PreProcessedEvents(
                run_name,
                mode="r",
                tag=self.preprocess_tag,
            )
            assert len(in_data) > 0, (
                "No matching preprocessed events with tag: "
                + self.preprocess_tag
                + "  in run: "
                + run_name
            )
            tot_length = len(in_data)
            print(tot_length)
            all_dict = {}
            class_count = 0
            for (
                data_count,
                item,
            ) in enumerate(in_data):
                final_state_dict = {
                    input_state: item[input_state.name]
                    for input_state in self.input_states
                }
                length = final_state_dict[self.input_states[0]].shape[0]
                if not self.check and self.model_type != "autoencoder":
                    self.check_consistency(final_state_dict)
                temp_indices[
                    IndexKey(
                        run_name,
                        count,
                        count + length,
                        "all",
                    )
                ] = (
                    in_data.current_run,
                    self.preprocess_tag,
                )
                count += length
                class_count += length
                all_dict = merge_flat_dict(
                    all_dict,
                    self.network_input(
                        final_state_dict,
                        class_index,
                    ),
                )
            self.index_dict = temp_indices
            return_dict[run_name] = all_dict
            if "debug" in sys.argv:
                print(run_name)
            split_point = int(
                len(return_dict[run_name]["Y"]) * (1 - self.test_split)
            )
            temp_indices["split_point"][run_name] = (
                count - class_count + split_point,
                split_point,
            )
            temp_indices["class_interval"][run_name] = (
                count - class_count,
                count,
            )
            temp_indices["class_indices"][run_name] = class_index
            for (
                key,
                value,
            ) in temp_indices.items():
                print(
                    key,
                    value,
                )
            print(
                "Return dict: ",
                return_dict[run_name]["X"][0].shape,
                return_dict[run_name]["Y"].shape,
                return_dict[run_name]["Y"][:-10],
            )
            if type(return_dict[run_name]["X"]) == list:
                temp_train = [
                    np_array[:split_point]
                    for np_array in return_dict[run_name]["X"]
                ]
                temp_val = [
                    np_array[split_point:]
                    for np_array in return_dict[run_name]["X"]
                ]
                if not train_dict:
                    (
                        train_dict["X"],
                        train_dict["Y"],
                    ) = (
                        temp_train,
                        return_dict[run_name]["Y"][:split_point],
                    )
                    (
                        val_dict["X"],
                        val_dict["Y"],
                    ) = (
                        temp_val,
                        return_dict[run_name]["Y"][split_point:],
                    )
                else:
                    assert len(train_dict["X"]) == len(temp_train) and len(
                        val_dict["X"]
                    ) == len(temp_val)
                    train_dict["X"] = [
                        np.concatenate(
                            (
                                all_data,
                                to_append,
                            ),
                            axis=0,
                        )
                        for all_data, to_append in zip(
                            train_dict["X"],
                            temp_train,
                        )
                    ]
                    val_dict["X"] = [
                        np.concatenate(
                            (
                                all_data,
                                to_append,
                            ),
                            axis=0,
                        )
                        for all_data, to_append in zip(
                            val_dict["X"],
                            temp_val,
                        )
                    ]
                    train_dict["Y"] = np.concatenate(
                        (
                            train_dict["Y"],
                            return_dict[run_name]["Y"][:split_point],
                        ),
                        axis=0,
                    )
                    val_dict["Y"] = np.concatenate(
                        (
                            val_dict["Y"],
                            return_dict[run_name]["Y"][split_point:],
                        ),
                        axis=0,
                    )
                if "debug" in sys.argv:
                    print("train: ")
                    [print(array.shape) for array in train_dict["X"]]
                    print(
                        train_dict["Y"].shape,
                        train_dict["Y"][-10:],
                    )
                    print("val:")
                    [print(array.shape) for array in val_dict["X"]]
                    print(val_dict["Y"].shape)
            else:
                if not train_dict:
                    (
                        train_dict["X"],
                        train_dict["Y"],
                    ) = (
                        return_dict[run_name]["X"][:split_point],
                        return_dict[run_name]["Y"][:split_point],
                    )
                    (
                        val_dict["X"],
                        val_dict["Y"],
                    ) = (
                        return_dict[run_name]["X"][split_point:],
                        return_dict[run_name]["Y"][split_point:],
                    )
                else:
                    train_dict["X"] = np.concatenate(
                        (
                            train_dict["X"],
                            return_dict[run_name]["X"][:split_point],
                        ),
                        axis=0,
                    )
                    train_dict["Y"] = np.concatenate(
                        (
                            train_dict["Y"],
                            return_dict[run_name]["Y"][:split_point],
                        ),
                        axis=0,
                    )
                    val_dict["X"] = np.concatenate(
                        (
                            val_dict["X"],
                            return_dict[run_name]["X"][split_point:],
                        ),
                        axis=0,
                    )
                    val_dict["Y"] = np.concatenate(
                        (
                            val_dict["Y"],
                            return_dict[run_name]["Y"][split_point:],
                        ),
                        axis=0,
                    )
            class_index += 1
        self.index_dict = temp_indices
        self.train_data = train_dict
        self.val_data = val_dict
        self.data = return_dict
        return

    def _load_data(self):
        """
        Load data for each class and split it into training and validation sets.

        This method loads data for each class, splits it into training and validation sets, and prepares the data for training a machine learning model.

        Returns:
            None
        """
        return_dict = {run_name: {} for run_name in self.class_names}
        train_dict = {}
        val_dict = {}
        (
            train_count,
            val_count,
        ) = (0, 0)
        class_index = 0
        class_indices = {}
        for run_name in self.class_names:
            in_data = PreProcessedEvents(
                run_name,
                mode="r",
                tag=self.preprocess_tag,
            )
            assert len(in_data) > 0, (
                "No matching preprocessed events with tag: "
                + self.preprocess_tag
                + "  in run: "
                + run_name
            )
            class_train_count = 0
            class_val_count = 0
            tot_length = len(in_data)
            print(tot_length)
            (
                val_full,
                train_full,
            ) = (
                False,
                False,
            )
            class_indices[run_name] = class_index
            for (
                data_count,
                item,
            ) in enumerate(in_data):
                final_state_dict = {
                    input_state: item[input_state.name]
                    for input_state in self.input_states
                }
                if not self.check and self.model_type != "autoencoder":
                    self.check_consistency(final_state_dict)
                return_dict[run_name] = merge_flat_dict(
                    return_dict[run_name],
                    final_state_dict,
                )
                length = final_state_dict[self.input_states[0]].shape[0]
                if self.validation_tag != "":
                    condition = (
                        self.validation_tag in os.listdir(in_data.current_run)
                        and not val_full
                    )
                else:
                    condition = not val_full
                if condition:
                    tag = "val"
                    if class_val_count + length > self.validation_length:
                        length = self.validation_length - class_val_count
                        val_dict = merge_flat_dict(
                            val_dict,
                            self.network_input(
                                final_state_dict,
                                class_index,
                            ),
                            append_length=length,
                        )
                        val_full = True
                    else:
                        val_dict = merge_flat_dict(
                            val_dict,
                            self.network_input(
                                final_state_dict,
                                class_index,
                            ),
                        )
                    self.index_dict["val"][
                        IndexKey(
                            run_name,
                            val_count,
                            val_count + length,
                            tag,
                        )
                    ] = (
                        in_data.current_run,
                        self.preprocess_tag,
                    )
                    val_count = val_count + length
                    class_val_count = class_val_count + length
                elif not train_full:
                    tag = "train"
                    if class_train_count + length > self.train_length:
                        length = self.train_length - class_train_count
                        train_dict = merge_flat_dict(
                            train_dict,
                            self.network_input(
                                final_state_dict,
                                class_index,
                            ),
                            append_length=length,
                        )
                        train_full = True
                    else:
                        train_dict = merge_flat_dict(
                            train_dict,
                            self.network_input(
                                final_state_dict,
                                class_index,
                            ),
                        )
                    self.index_dict["train"][
                        IndexKey(
                            run_name,
                            train_count,
                            train_count + length,
                            tag,
                        )
                    ] = (
                        in_data.current_run,
                        self.preprocess_tag,
                    )
                    train_count = train_count + length
                    class_train_count = class_train_count + length
                else:
                    break
            self.check = False
            class_index += 1
        self.index_dict["class_indices"] = class_indices
        for item in self.index_dict:
            for key in self.index_dict[item]:
                print(
                    item,
                    key,
                    self.index_dict[item][key],
                )
        self.data = return_dict
        self.train_data = train_dict
        self.val_data = val_dict
        return

    def network_input(
        self,
        final_state_dict,
        class_index,
    ):
        """
        Prepare the network input data for a single data point.

        This method takes the final_state_dict, which contains the preprocessed data for a single data point, and class_index, which is the index of the class for this data point, and prepares the network input data in the required format for training or inference.

        Parameters:
            final_state_dict (dict): A dictionary containing the preprocessed data for a single data point.
            class_index (int): The index of the class for this data point.

        Returns:
            dict: A dictionary containing the network input data, including 'X' for input features and 'Y' for class labels.
        """
        return_dict = {}
        if len(self.input_states) == 1:
            if type(self.input_states[0].index) == int:
                return_dict["X"] = final_state_dict[self.input_states[0]][
                    :,
                    self.input_states[0].index,
                ]
            else:
                return_dict["X"] = final_state_dict[self.input_states[0]]
            total_entries = len(return_dict["X"])
        else:
            listed = [0] * len(self.input_states)
            for f_state in self.input_states:
                if type(f_state.index) == int:
                    listed[f_state.network_input_index] = final_state_dict[
                        f_state
                    ][
                        :,
                        f_state.index,
                    ]
                else:
                    listed[f_state.network_input_index] = final_state_dict[
                        f_state
                    ]
            return_dict["X"] = listed
            total_entries = len(return_dict["X"][0])
        temp_y = np.zeros(
            (
                total_entries,
                len(self.class_names),
            ),
            dtype="float64",
        )
        temp_y[:, class_index] = 1.0
        return_dict["Y"] = temp_y
        return return_dict

    def write_to_disk(self):
        """
        Write data and dictionaries to disk.

        This method is used to save various data and dictionaries to disk, including training data, validation data, shuffled index dictionaries, and index dictionaries. It checks the mode of the class instance to ensure that it is in "write" mode before proceeding with saving the data.

        Returns:
        """
        assert self.mode == "w", "Trying to write in read instance of class"
        print(
            "Saving to: ",
            self.data_save_path,
        )
        save_list = [
            (
                self.train_data,
                "train.h",
            ),
            (
                self.val_data,
                "val.h",
            ),
            (
                self.shuffled_index_dict,
                "shuffled_index.h",
            ),
            (
                self.index_dict,
                "index_dict.h",
            ),
        ]
        for item in save_list:
            try:
                Pickle(
                    item[0],
                    item[1],
                    save_path=self.data_save_path,
                )
            except OverflowError as e:
                print(
                    e,
                    "Could not save ",
                    item[1],
                    "saving as .npy",
                )
                pwd = os.getcwd()
                os.chdir(self.data_save_path)
                try:
                    os.mkdir(item[1][-2:])
                except OSError:
                    pass
                os.chdir(item[1][-2:])
                for (
                    key,
                    val,
                ) in item[0].items():
                    np.save(
                        key,
                        val,
                    )
                os.chdir(pwd)
        return

    def check_consistency(
        self,
        final_states_dict,
    ):
        """
        Write data and dictionaries to disk.

        This method is used to save various data and dictionaries to disk, including training data, validation data, shuffled index dictionaries, and index dictionaries. It checks the mode of the class instance to ensure that it is in "write" mode before proceeding with saving the data.

        Returns:
            None
        """
        print("First load! Checking consistency...")
        for input_state in self.input_states:
            assert input_state in final_states_dict
            print(
                input_state.name,
                " found in dict keys!",
            )
            current_array = final_states_dict[input_state]
            print(
                current_array.shape[1:],
                input_state.shape,
            )
            if input_state.index is None:
                assert (
                    input_state.shape == current_array.shape[1:]
                ), "Wrong input shape!"
                i = 1
            else:
                assert (
                    input_state.shape == current_array.shape[2:]
                ), "Wrong input shape"
                i = 2
                assert input_state.index < current_array.shape[1], IndexError(
                    "Index out of range"
                )
            print(
                input_state.shape,
                "==",
                final_states_dict[input_state].shape[i:],
            )
        self.check = True
        return

    def multiple_input_check(
        self,
    ):
        """
        Check and shuffle multiple input data.

        This method is used to check and shuffle the multiple input data, including both the training and validation datasets. It ensures that the data is properly shuffled for training. If the data is in the form of a list, it shuffles each element separately and reassembles them.

        Returns:
            tuple: A tuple containing the shuffled training and validation data.

        Note:
            This method assumes that the data is organized as a dictionary with "X" and "Y" keys, where "X" may contain multiple input arrays to be shuffled separately.
        """
        print(self.train_data.keys())  # ,self.train_data["Y"][-10:])
        if type(self.train_data["X"]) != list:
            self.train_data = nu_shuffle(**self.train_data)
            self.val_data = nu_shuffle(**self.val_data)
        else:
            args_train = tuple(self.train_data["X"] + [self.train_data["Y"]])
            args_val = tuple(self.val_data["X"] + [self.val_data["Y"]])
            (
                args_train,
                _,
            ) = array_shuffle(*args_train)
            (
                args_val,
                _,
            ) = array_shuffle(*args_val)
            (
                self.train_data["X"],
                self.train_data["Y"],
            ) = (
                list(args_train[:-1]),
                args_train[-1],
            )
            (
                self.val_data["X"],
                self.val_data["Y"],
            ) = (
                list(args_val[:-1]),
                args_val[-1],
            )
        if self.model_type != "autoencoder":
            print(
                "Checking shuffled...",
                self.train_data["Y"][:10],
                self.train_data["Y"][-10:],
            )
        return (
            self.train_data,
            self.val_data,
        )

    def pre_inputs(self, total_length):
        """
        Initialize input arrays for preprocessing.

        This method initializes input arrays for preprocessing based on the specified `total_length`. It creates empty arrays for each input state and prepares a corresponding empty array for the target labels (`Y`) with the appropriate shape.

        Parameters:
            total_length (int): The total length of the input arrays.

        Note:
            The method assumes that `self.input_states` contains the descriptions of input states, and `self.class_names` contains the names of the target classes.

        Example:
            To initialize input arrays for a total length of 100 samples, you can call `pre_inputs(100)`.
        """
        X = [[] for i in range(len(self.input_states))]
        print(X)
        for input_state in self.input_states:
            shape = tuple([total_length] + list(input_state.shape))
            X[input_state.network_input_index] = np.zeros(shape)
            Y = np.zeros(
                (
                    total_length,
                    len(self.class_names),
                )
            )
        [print(item.shape) for item in X]
        return X, Y

    def load_from_index_dict(self, path):
        """currently writing for binary class, change later for n-class classification"""
        """
        Load data from the index dictionary.

        This method loads data based on the provided index dictionary from the specified `path`. It handles the loading of training and validation data, as well as their corresponding class indices.

        Parameters:
        path (str): The path to the directory containing the index dictionary and preprocessed data.

        Note:
        This method assumes that the index dictionary contains information about data splits and class indices.

        Example:
        To load data from the specified `path`, you can call `load_from_index_dict(path)`.
        """
        pwd = os.getcwd()
        print(path, pwd)
        index_dict = Unpickle(
            "index_dict.h",
            load_path=path,
        )
        try:
            split_points = index_dict.pop("split_point")
        except KeyError:
            split_points = {}
        try:
            class_intervals = index_dict.pop("class_interval")
        except KeyError:
            total_lengths = {}
        try:
            class_indices = index_dict.pop("class_indices")
            run_names = [key for key in class_indices]
            total_length = max(
                [class_intervals[name][-1] for name in run_names]
            )
            print(total_length)
            (
                X,
                Y,
            ) = self.pre_inputs(total_length)
        except KeyError:
            run_names = []
        (
            train_dict,
            val_dict,
        ) = ({}, {})
        all_count = 0
        if "train" not in index_dict and "val" not in index_dict:
            for (
                key,
                value,
            ) in index_dict.items():
                temp_dict = Unpickle(
                    "preprocessed_" + value[1] + ".h",
                    load_path=value[0],
                )
                for input_state in self.input_states:
                    loaded_shape = temp_dict[input_state.name].shape
                    if input_state.index is not None:
                        X[input_state.network_input_index][
                            key.start: key.end
                        ] = temp_dict[input_state.name][
                            :,
                            input_state.index,
                        ]
                    else:
                        X[input_state.network_input_index][
                            key.start: key.end
                        ] = temp_dict[input_state.name]
                    Y[
                        key.start: key.end,
                        class_indices[key.class_name],
                    ] = 1.0
                    print(
                        input_state.name,
                        key.start,
                        key.end,
                    )
                if "debug" in sys.argv:
                    print(
                        Y[key.start: key.start + 10],
                        key,
                    )
            for run_name in run_names:
                (
                    start,
                    end,
                ) = (
                    class_intervals[run_name][0],
                    class_intervals[run_name][1],
                )
                split = split_points[run_name][0]
                print(
                    start,
                    split,
                    end,
                )
                if "X" not in train_dict:
                    train_dict["X"] = [item[start:split] for item in X]
                    train_dict["Y"] = Y[start:split]
                    val_dict["X"] = [item[split:end] for item in X]
                    val_dict["Y"] = Y[split:end]
                else:
                    train_dict["X"] = [
                        np.concatenate(
                            (
                                prev_item,
                                item[start:split],
                            ),
                            axis=0,
                        )
                        for prev_item, item in zip(
                            train_dict["X"],
                            X,
                        )
                    ]
                    train_dict["Y"] = np.concatenate(
                        (
                            train_dict["Y"],
                            Y[start:split],
                        ),
                        axis=0,
                    )
                    val_dict["X"] = [
                        np.concatenate(
                            (
                                prev_item,
                                item[split:end],
                            ),
                            axis=0,
                        )
                        for prev_item, item in zip(
                            val_dict["X"],
                            X,
                        )
                    ]
                    val_dict["Y"] = np.concatenate(
                        (
                            val_dict["Y"],
                            Y[split:end],
                        ),
                        axis=0,
                    )
        else:
            lengths = {
                item: max(key.end for key in index_dict[item])
                for item in index_dict
            }
            data_dict = {
                item: {
                    space: array
                    for space, array in zip(
                        (
                            "X",
                            "Y",
                        ),
                        self.pre_inputs(lengths[item]),
                    )
                }
                for item in index_dict
            }
            if "debug" in sys.argv:
                print(lengths)
            try:
                class_indices = index_dict.pop("class_indices")
            except KeyError as e:
                print(
                    e,
                    "\n creating new class one-hot encoding, check correctness in case of inference or comparing pretrained models...",
                )
                class_indices = {
                    item: i
                    for item, i in zip(
                        self.class_names,
                        range(len(self.class_names)),
                    )
                }
            for (
                tag,
                tag_indices,
            ) in index_dict.items():
                for (
                    key,
                    value,
                ) in tag_indices.items():
                    temp_dict = Unpickle(
                        "preprocessed_" + value[1] + ".h",
                        load_path=value[0],
                    )
                    for input_state in self.input_states:
                        loaded_shape = temp_dict[input_state.name].shape
                        if input_state.index is not None:
                            data_dict[tag]["X"][
                                input_state.network_input_index
                            ][key.start: key.end] = temp_dict[
                                input_state.name
                            ][
                                : key.end - key.start,
                                input_state.index,
                            ]
                        else:
                            data_dict[tag]["X"][
                                input_state.network_input_index
                            ][key.start: key.end] = temp_dict[
                                input_state.name
                            ][
                                : key.end - key.start
                            ]
                        data_dict[tag]["Y"][
                            key.start: key.end,
                            class_indices[key.class_name],
                        ] = 1.0
                    if "debug" in sys.argv:
                        print(
                            data_dict[tag]["Y"][key.start: key.start + 10],
                            key,
                        )
            if "debug" in sys.argv:
                print(
                    data_dict["train"]["X"][0].shape,
                    data_dict["train"]["Y"].shape,
                    data_dict["val"]["X"][0].shape,
                    data_dict["val"]["Y"].shape,
                )
            (
                train_dict,
                val_dict,
            ) = (
                data_dict["train"],
                data_dict["val"],
            )
        if len(self.input_states) == 1:
            (
                train_dict["X"],
                val_dict["X"],
            ) = (
                train_dict["X"][0],
                val_dict["X"][0],
            )
        else:
            for item in train_dict["X"]:
                print(item.shape)
            print(train_dict["Y"].shape)
            for item in val_dict["X"]:
                print(item.shape)
            print(val_dict["Y"].shape)
        return (
            train_dict,
            val_dict,
        )

    def get_data(self):
        """
        Get the training and validation data.

        This method retrieves the training and validation data from stored files or loads them from index dictionaries. It also performs a consistency check on the loaded data and returns the data in the appropriate format.

        Returns:
        tuple or dict: A tuple containing training and validation data, or dictionaries containing data if in write mode.

        Example:
        To get the training and validation data, you can call `get_data()`.
        """
        try:
            self.train_data = Unpickle(
                "train.h",
                load_path=self.data_save_path,
            )
            self.val_data = Unpickle(
                "val.h",
                load_path=self.data_save_path,
            )
        except Exception as e:
            print(
                e,
                "\nTrying to reload from index dictionaries...",
            )
            try:
                (
                    self.train_data,
                    self.val_data,
                ) = self.load_from_index_dict(self.data_save_path)
            except Exception as e:
                print(
                    e,
                    "\n loading from madgraph events for the first time ...",
                )
                if self.mode != "w":
                    raise IOError("data not found")
                if not self.train_data:
                    self.load_data()
                if self.save:
                    self.write_to_disk()
        finally:
            (
                train_data,
                val_data,
            ) = self.multiple_input_check()
        self.train_data = train_data
        self.val_data = val_data
        if self.model_type != "autoencoder":
            if self.mode == "w":
                return (
                    self.train_data["X"],
                    self.train_data["Y"],
                    self.val_data["X"],
                    self.val_data["Y"],
                )
            else:
                return (
                    self.train_data,
                    self.val_data,
                )
        else:
            return (
                self.train_data["X"],
                self.train_data["X"],
                self.val_data["X"],
                self.val_data["X"],
            )


class AutoencoderData(ModelData):
    """
    Data handler class for training autoencoder models.

    This class extends the base `ModelData` class and provides methods for loading or generating training and validation data specific to autoencoder models.

    Attributes:
        - _prefix_path (str): The prefix path where data is stored.
        - mode (str): The mode of operation, "w" for write, "r" for read.
        - save_as (str): The format in which data is saved (e.g., "numpy_array").
        - total_length (int): The total length of the data.
        - test_split (float): The split ratio for test data.
        - model_type (str): The type of the model.
        - preprocess_tag (str): A tag for data preprocessing.
        - class_names (list): List of class names.
        - input_states (list): List of input states.
        - save (bool): Flag to control saving data.
        - run_tag (str): A tag for the run.
        - tag_path (str): The path for run tags.
        - preprocessed_path (str): The path for preprocessed data.
        - run_name (str): The name of the run.
        - run_path (str): The path for the run.
        - data_save_path (str): The path for saved data.
        - model_checkpoints_path (str): The path for model checkpoints.
        - train_length (int): The length of the training data.
        - validation_length (int): The length of the validation data.
        - validation_tag (str): A tag for validation data.
        - train_data (dict): Training data dictionary.
        - val_data (dict): Validation data dictionary.
        - data (None): Placeholder for data.
        - index_dict (dict): Dictionary to store indices.
        - shuffled_index_dict (dict): Dictionary to store shuffled indices.
        - check (bool): Flag for data checking.
        - data_handler (obj): Data handler object.
        - handler_kwargs (dict): Additional keyword arguments for the data handler.
        - load_path (None): Placeholder for load path.
        - train_indices (None): Placeholder for training indices.
        - val_indices (None): Placeholder for validation indices.
    """

    def __init__(*args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_data(self):
        """
        Get training and validation data for an autoencoder model.

        This method loads or generates training and validation data for an autoencoder model. If the data files "train.h" and "val.h" exist
        in the specified data save path, they are loaded. Otherwise, the data is generated and saved if the mode is "write" (w). The method
        also performs data shuffling and preprocessing.

        Returns:
        Tuple containing training and validation data:
            - train_X: Training input data
            - train_X: Training target data (same as input for autoencoder)
            - val_X: Validation input data
            - val_X: Validation target data (same as input for autoencoder)
        """
        if {
            "train.h",
            "val.h",
        }.issubset(set(os.listdir(self.data_save_path))):
            self.train_data = Unpickle(
                "train.h",
                load_path=self.data_save_path,
            )
            self.val_data = Unpickle(
                "val.h",
                load_path=self.data_save_path,
            )
        else:
            if self.mode != "w":
                raise IOError("data not found")
            if not self.train_data:
                self.load_data()
            print(
                self.train_data["Y"][:10],
                self.train_data["Y"][-10:],
            )
            (
                train_X,
                val_X,
            ) = (
                self.train_data["X"],
                self.val_data["X"],
            )
            if type(train_X) != list:
                train_data = array_shuffle(
                    X=np.expand_dims(
                        train_X,
                        -1,
                    ),
                    Y=self.train_data["Y"],
                )
                val_data = array_shuffle(
                    X=np.expand_dims(
                        val_X,
                        -1,
                    ),
                    Y=self.val_data["Y"],
                )
            else:
                container_train = {
                    "X"
                    + str(i): np.expand_dims(
                        item,
                        -1,
                    )
                    for i, item in enumerate(train_X)
                }
                container_train["Y"] = self.train_data["Y"]
                container_val = {
                    "X"
                    + str(i): np.expand_dims(
                        item,
                        -1,
                    )
                    for i, item in enumerate(val_X)
                }
                container_val["Y"] = self.val_data["Y"]
                container_train = array_shuffle(all=container_train)
                container_val = array_shuffle(all=container_val)
                train_data = {"Y": container_train["Y"]}
                val_data = {"Y": container_val["Y"]}
                train_data["X"] = [
                    container_train["X" + str(i)] for i in range(len(train_X))
                ]
                val_data["X"] = [
                    container_val["X" + str(i)] for i in range(len(val_X))
                ]
                train_data["ind_map"] = container_train["ind_map"]
                val_data["ind_map"] = container_val["ind_map"]
                self.shuffled_index_dict["train"] = train_data["ind_map"]
                self.shuffled_index_dict["val"] = val_data["ind_map"]
            self.train_data = train_data
            self.val_data = val_data
            if self.save:
                self.write_to_disk()
        self.train_data = train_data
        self.val_data = val_data
        return (
            self.train_data["X"],
            self.train_data["X"],
            self.val_data["X"],
            self.val_data["X"],
        )


class DataHandler(object):
    """
    Initialize a DataHandler object.

    Parameters:
        - run_name (str): Name of the run.
        - dir_name (str): Name of the directory to store data in.
        - re_initialize (bool): If True, overwrite existing directory.
        - mode (str): "w" for write mode, "r" for read mode.

    """

    def __init__(self, *args, **kwargs):
        self._prefix_path = check_dir("./network_runs")
        self.mode = kwargs.get("mode", "w")
        self.save_as = kwargs.get(
            "save_as",
            "numpy_array",
        )
        compulsory_kwargs = {
            "input_states",
            "preprocess_tag",
            "class_names",
        }
        assert compulsory_kwargs.issubset(set(kwargs.keys()))
        self.total_length = kwargs.get(
            "total_length",
            100,
        )
        self.test_split = kwargs.get(
            "test_split",
            0.2,
        )
        self.model_type = kwargs.get(
            "model_type",
            "model",
        )
        self.preprocess_tag = kwargs.get("preprocess_tag")
        self.class_names = kwargs.get("class_names")
        self.input_states = kwargs.get("input_states")
        self.save = kwargs.get("save", True)
        if self.save:
            self.run_tag = kwargs.get(
                "run_tag",
                "no_tag",
            )
            self.tag_path = check_dir(
                os.path.join(
                    self._prefix_path,
                    self.run_tag,
                )
            )
            self.preprocessed_path = check_dir(
                os.path.join(
                    self.tag_path,
                    self.preprocess_tag,
                )
            )
            self.run_name = kwargs.get(
                "run_name",
                self.model_type + "_" + self.preprocess_tag,
            )
            self.run_path = check_dir(
                os.path.join(
                    self.preprocessed_path,
                    self.run_name,
                )
            )
            self.data_save_path = check_dir(
                os.path.join(
                    self.run_path,
                    "data",
                )
            )
            self.model_checkpoints_path = check_dir(
                os.path.join(
                    self.run_path,
                    "model_checkpoints",
                )
            )
        self.train_length = kwargs.get(
            "train_length",
            None,
        )
        self.validation_length = kwargs.get(
            "validation_length",
            None,
        )
        self.validation_tag = kwargs.get(
            "validation_tag",
            "",
        )
        self.train_data = {}
        self.val_data = {}
        self.data = None
        self.index_dict = {
            "train": {},
            "val": {},
        }
        self.shuffled_index_dict = {
            "train": {},
            "val": {},
        }
        self.check = False
        self.data_handler = kwargs.get("data_handler")
        self.handler_kwargs = kwargs.get(
            "handler_kwargs",
            {},
        )
        self.load_path = None
        (
            self.train_indices,
            self.val_indices,
        ) = (None, None)

    def handler_load(self):
        """
        Load a DataHandler object from disk.

        Returns:
            data_handler: Loaded DataHandler object.
        """
        dictionary = Unpickle(
            "dictionary",
            load_path=self.data_save_path,
        )
        kwargs = dictionary["handler_kwargs"]
        for (
            save_key,
            item,
        ) in zip(
            dictionary["input_keys"],
            self.input_states,
        ):
            assert save_key == item.name
        for (
            name,
            real_name,
        ) in zip(
            dictionary["classes"],
            self.class_names,
        ):
            assert name == real_name
        assert self.preprocess_tag == dictionary["preprocess_tag"]
        kwargs["indices"] = Unpickle(
            "indices",
            load_path=self.data_save_path,
        )
        return self.data_handler(
            self.class_names, input_keys=dictionary["input_keys"], **kwargs
        )

    def get_data(self):
        """
        Get train and validation data.

        Returns:
            Tuple containing train and validation data.
        """
        if "dictionary" in os.listdir(self.data_save_path):
            data = self.handler_load()
            save = False
        else:
            data = self.data_handler(
                self.class_names,
                input_keys=[item.name for item in self.input_states],
                **self.handler_kwargs
            )
            save = True
        self.load_path = data.pop("path")
        (
            train,
            val,
        ) = data.pop(
            "train"
        ), data.pop("val")
        self.train_indices = train.pop("indices")
        self.val_indices = val.pop("indices")
        if save:
            self.write_to_disk()
        return (
            train["X"],
            train["Y"],
            val["X"],
            val["Y"],
        )

    def write_to_disk(self):
        """
        Write DataHandler object and indices to disk.
        """
        dictionary = {
            "classes": self.class_names,
            "handler_kwargs": self.handler_kwargs,
            "preprocess_tag": self.preprocess_tag,
            "input_keys": [item.name for item in self.input_states],
            "load_path": self.load_path,
        }
        print("Saving kwargs...")
        Pickle(
            dictionary,
            "dictionary",
            save_path=self.data_save_path,
        )
        print("Saving indices...")
        Pickle(
            {
                "train": self.train_indices,
                "val": self.val_indices,
            },
            "indices",
            save_path=self.data_save_path,
        )
        return


if __name__ == "__main__":
    fatjet1 = InputState(
        name="FatJet",
        shape=(32, 32),
        index=1,
        network_input_index=0,
    )
    fatjet2 = InputState(
        name="FatJet",
        shape=(2, 32, 32),
        index=None,
        network_input_index=1,
    )
    n = NetworkData(
        class_names=(
            "wwx",
            "dijet",
        ),
        input_shape=(
            2,
            32,
            32,
        ),
        preprocess_tag="try",
        input_states=[fatjet1],
        model_type="jet_image",
        run_name="trial",
        validation_tag="val",
        train_length=25000,
        validation_length=5000,
    )
    print(
        n._prefix_path,
        n.run_name,
        n.run_path,
        n.data_save_path,
        "\n",
        n.model_checkpoints_path,
    )
    (
        X,
        Y,
        X_val,
        Y_val,
    ) = n.get_data()
    try:
        print(
            X.shape,
            X_val.shape,
            Y_val.shape,
            Y[:10],
            Y[-10:],
        )
    except BaseException:
        print(
            X[0].shape,
            X[1].shape,
            X_val[0].shape,
            X_val[1].shape,
            Y_val.shape,
            Y[:10],
            Y[-10:],
        )
