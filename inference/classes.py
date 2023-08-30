from warnings import warn
import os
import sys

import numpy as np

if "operation_check" not in sys.argv:
    import tensorflow.keras as keras

from ..classes import (
    NetworkMethod,
)
from ..network.data import (
    ModelData,
)
from ..io.saver import (
    Unpickle,
    Pickle,
)
from ..genutils import (
    check_file,
    check_dir,
    pool_splitter,
    print_events,
)


class Error(Exception):
    pass


class ExecutionError(Error):
    pass


class Inference(NetworkMethod):
    def __init__(self, model_name, tag="val", **kwargs):
        self.tag = tag
        self.extra_handler_kwargs = kwargs.get(
            "extra_handler_kwargs",
            {},
        )
        self.model_name = model_name
        self.run_tag = kwargs.get(
            "run_tag",
            "no_tag",
        )
        self.preprocess_tag = kwargs.get("preprocess_tag")
        self._prefix_path = check_dir("./network_runs")
        self.best = kwargs.get(
            "best",
            "high",
        )
        self.run_path = os.path.join(
            self._prefix_path,
            self.run_tag,
            self.preprocess_tag,
            self.model_name,
        )
        self.data_path = os.path.join(
            self.run_path,
            "data",
        )
        self.model_checkpoints = os.path.join(
            self.run_path,
            "model_checkpoints",
        )
        # if "inference" in kwargs and kwargs["inference"] != "run":
        self.inference_path = check_dir(
            os.path.join(
                self.run_path,
                "inference",
            )
        )
        self.data_save_path = check_dir(
            os.path.join(
                self.inference_path,
                "data",
            )
        )
        assert (
            os.path.exists(self.run_path)
            and os.path.exists(self.data_path)
            and os.path.exists(self.model_checkpoints)
        )
        self.class_names = None
        self.tolerance = kwargs.get(
            "tolerance",
            0.02,
        )
        self.best_model_path = None
        self.unoperated_seperated_data = None
        self.operated_data = None
        self.operated_seperated_data = None
        self.model = None
        self.handler_kwargs = None
        self.replaced_data = None
        self.input_keys = None
        self.signal_prediction = None
        self.signal_acc = None
        self.get_model = kwargs.get(
            "get_model",
            False,
        )
        self.all_data = None
        self.per_run_model_paths = None
        self.per_run_models = None
        if not self.get_model:
            if "dictionary.pickle" in os.listdir(self.data_path):
                assert (
                    "data_handler" in kwargs
                ), "Provide data handler used while training model"
                self.load_data = kwargs.get("data_handler")
                self.unoperated_data = self.handler_load(
                    **self.extra_handler_kwargs
                )
            else:
                self.unoperated_data = self.model_data_load()

    def replace_data(self, data, **kwargs):
        """replace a particular class <out_key> with a new class <in_key> with <data_tag> loaded from ./processed_events/<data_tag>/all"""
        assert "new_class" in kwargs
        new_class = kwargs.get("new_class")
        out_key = kwargs.get(
            "out_key",
            self.class_names[-1],
        )
        preprocess_tag = kwargs.get(
            "data_tag",
            self.preprocess_tag,
        )
        input_keys = kwargs.get(
            "input_keys",
            self.input_keys,
        )
        new_data = Unpickle(
            new_class + ".h",
            load_path="./processed_events/" + preprocess_tag + "/all/",
        )
        # print_events(data),print_events(new_data)
        prev_data = data.pop(out_key)
        # print_events(prev_data)
        X_new = [[] for _ in self.input_keys]
        for (
            i,
            key,
        ) in enumerate(self.input_keys):
            if key == "tower_image":
                X_new[i] = np.expand_dims(
                    new_data["tower_image"],
                    -1,
                )
            else:
                X_new[i] = new_data[key]
            length = len(X_new[i])
        if len(input_keys) == 1:
            X_new = X_new[0]
        Y_index = np.where(prev_data["Y"][0])
        print(Y_index)
        Y_new = np.zeros((length, 2))
        Y_new[:, Y_index] = 1
        print(Y_new[:10])
        # print (prev_data.shape)
        data[new_class] = {
            "X": X_new,
            "Y": Y_new,
        }
        # print_events(data[in_key])
        return data

    def handler_load(self, **opt):
        # print (self.data_path,os.listdir(self.data_path))
        # sys.exit()
        dictionary = Unpickle(
            "dictionary.pickle",
            load_path=self.data_path,
        )
        self.class_names = dictionary["classes"]
        load_path = dictionary["load_path"]
        kwargs = dictionary["handler_kwargs"]
        kwargs["input_keys"] = dictionary["input_keys"]
        kwargs.update(opt)
        self.input_keys = dictionary["input_keys"]
        kwargs["preprocess_tag"] = dictionary["preprocess_tag"]
        assert kwargs["preprocess_tag"] == self.preprocess_tag
        self.handler_kwargs = kwargs
        kwargs["indices"] = Unpickle(
            "indices.pickle",
            load_path=self.data_path,
        )
        self.all_data = self.load_data(self.class_names, **kwargs)
        if opt.get(
            "return_all",
            False,
        ):
            print("Return all option active")
            warn(
                "Do not use return_all =True option while using function methods!"
            )
            return self.all_data
        else:
            return self.all_data[self.tag]

    def model_data_load(
        self,
    ):
        try:
            data = Unpickle(
                self.tag + ".pickle",
                load_path=self.data_path,
            )
        except Exception as e:
            print(
                e,
                "\nCould not load " + self.tag + ".pickle from ",
                self.data_path,
                "\ntrying to load through index dicts...",
            )
            data = self.load_from_index_dict(self.data_path)
        else:
            try:
                index = Unpickle(
                    "index_dict.pickle",
                    load_path=self.data_path,
                )
            except FileNotFoundError:
                self.class_names = (
                    "class 0",
                    "class 1",
                )
            else:
                class_names = []
                for key in index["val"]:
                    if key.class_name not in class_names:
                        class_names.append(key.class_name)
                self.class_names = tuple(class_names)
        finally:
            print(self.class_names)
        return data

    def seperate_classes(self, data):
        print("Seperating classes...")
        X, Y = (
            data["X"],
            data["Y"],
        )
        (
            class_0,
            class_1,
        ) = np.nonzero(
            Y[:, 0]
        ), np.nonzero(Y[:, 1])
        if "debug" in sys.argv:
            print(
                type(class_0),
                len(class_0[0]),
                class_0[0][:2],
                Y[class_0[0][:2]],
            )
        if not isinstance(X, list):
            X0 = X[class_0]
            X1 = X[class_1]
        else:
            X0 = [item[class_0] for item in X]
            X1 = [item[class_1] for item in X]
        Y0, Y1 = (
            Y[class_0],
            Y[class_1],
        )
        if not self.class_names:
            self.class_names = (
                "class_0",
                "class_1",
            )
        return {
            self.class_names[0]: {
                "X": X0,
                "Y": Y0,
            },
            self.class_names[1]: {
                "X": X1,
                "Y": Y1,
            },
        }

    def choose_model(self, model_files):
        val_acc = []
        remove_inds = []
        for (
            i,
            item,
        ) in enumerate(model_files):
            try:
                val = eval(item[: -len(".hdf5")].split("_")[-1])
            except NameError:
                print(
                    item,
                    " no val_acc in filename.",
                )
                remove_inds.append(i)
                continue
            else:
                val_acc.append(val)
        model_files = [
            item for i, item in enumerate(model_files) if i not in remove_inds
        ]
        # print (item,eval(item[:-len(".hdf5")].split("_")[-1]))
        val_acc = np.array(val_acc)
        if self.best == "high":
            ind = -1
        else:
            ind = 0
        model_path = model_files[val_acc.argsort()[ind]]
        return model_path

    def get_best_model(self):
        model_files = check_file(
            ".hdf5",
            self.model_checkpoints,
        )
        model_path = self.choose_model(model_files)
        print(
            "Loading best model from path : ",
            model_path,
        )
        model = keras.models.load_model(model_path)
        self.best_model_path = model_path
        self.model = model
        return model

    def get_best_from_seperate_runs(
        self,
    ):
        model_files = check_file(
            ".hdf5",
            self.model_checkpoints,
        )
        runs = np.unique([item.split("/")[-2] for item in model_files])
        per_run_files = [[] for _ in runs]
        for item in model_files:
            current_run = item.split("/")[-2]
            add_ind = np.where(current_run == runs)[0][0]
            per_run_files[add_ind].append(item)
        model_paths = []
        models = []
        for item in per_run_files:
            best_current_path = self.choose_model(item)
            print(
                "Loading model from path: ",
                best_current_path,
            )
            model_paths.append(best_current_path)
            models.append(keras.models.load_model(best_current_path))
        self.per_run_model_paths = model_paths
        self.per_run_models = models
        return self.per_run_models

    def per_run_predict(self, **kwargs):
        if self.per_run_models is None:
            self.get_best_from_seperate_runs()
        global_model = self.model
        return_list = []
        for (
            item,
            path,
        ) in zip(
            self.per_run_models,
            self.per_run_model_paths,
        ):
            self.model = item
            print(
                "\nPredicting with model loaded from: ",
                path,
            )
            print("Layer config of the model: ")
            for layer in item.layers:
                print(layer.get_config())
            item.summary()
            input("Press enter to continue: ")
            append_dict = self.predict(**kwargs)
            print_events(append_dict)
            return_list.append(append_dict)
        self.model = global_model
        return return_list

    def predict(
        self,
        operation="None",
        split=False,
        roc_plot=False,
        replace=None,
        save=False,
        batch_size=300,
    ):
        assert (
            self.unoperated_data is not None
        ), "Initialized for extracting model, set get_model to False during\
                                                 init or der self.unoperated_data for predicting data"
        if not self.model:
            self.get_best_model()
            # self.model.summary()
        return_dict = {}
        if operation == "None":
            combined_data = self.unoperated_data
        else:
            print("performing operation...")
            if (
                "debug" in sys.argv
                or "operation_check" in sys.argv
                or not split
            ):
                combined_data = operation(self.unoperated_data)
            combined_data = pool_splitter(
                operation,
                self.unoperated_data,
            )
        seperated = self.seperate_classes(combined_data)
        class_names = self.class_names
        if replace:
            print(
                "replacing ",
                class_names[-1],
                " with :",
                replace,
            )
            seperated = self.replace_data(
                seperated,
                new_class=replace,
            )
            class_names = (
                self.class_names[0],
                replace,
            )

        if roc_plot:
            roc_dict = {
                "combined": self.model.predict(
                    combined_data["X"],
                    verbose=2,
                )
            }
        if "prediction" in os.listdir(self.data_save_path) and save:
            saved_dict = Unpickle(
                "prediction",
                load_path=self.data_save_path,
            )
        else:
            saved_dict = {}
        if class_names[-1] == self.class_names[-1]:
            print("Evaluating combined validation data: ")
            return_dict["combined"] = self.model.evaluate(
                x=combined_data["X"],
                y=combined_data["Y"],
                batch_size=batch_size,
                verbose=1,
            )
            saved_dict["combined"] = return_dict["combined"]
        for (
            i,
            item,
        ) in enumerate(class_names):
            # if item==self.class_names[0] and self.signal_prediction is not
            # None: continue
            print(
                "Evaluating class: ",
                item,
            )
            self.model.evaluate(
                x=seperated[item]["X"],
                y=seperated[item]["Y"],
                batch_size=batch_size,
                verbose=1,
            )
            return_dict[item] = self.model.predict(
                seperated[item]["X"],
                verbose=1,
            )[:, 0]
            return_dict[item + "_true"] = seperated[item]["Y"][:, 0]
            if i == 0:
                (
                    y_true,
                    y_pred,
                ) = (
                    seperated[item]["Y"][:, 0],
                    return_dict[item],
                )
            else:
                (
                    y_true,
                    y_pred,
                ) = np.concatenate(
                    (
                        y_true,
                        seperated[item]["Y"][
                            :,
                            0,
                        ],
                    ),
                    axis=0,
                ), np.concatenate(
                    (
                        y_pred,
                        return_dict[item],
                    ),
                    axis=0,
                )
            saved_dict[item] = return_dict[item]
            if roc_plot:
                roc_dict[item] = self.model.predict(
                    seperated[item]["X"],
                    verbose=2,
                )
                if save:
                    Pickle(
                        roc_dict[item],
                        "roc_" + item,
                        save_path=self.data_save_path,
                    )
            if self.signal_acc is None and item == self.class_names[0]:
                self.signal_acc = return_dict[item]
                self.signal_prediction = self.model.predict(
                    seperated[item]["X"],
                    verbose=2,
                )
                # Pickle(self.signal_prediction,"roc_"+item,save_path=self.data_save_path)
        return_dict["y_true"] = y_true
        return_dict["y_pred"] = y_pred
        if "channel" in combined_data:
            return_dict["channel"] = combined_data["channel"]
        if save:
            Pickle(
                saved_dict,
                "prediction",
                save_path=self.data_save_path,
            )
        if roc_plot:
            return_dict["roc"] = roc_dict
        return return_dict
