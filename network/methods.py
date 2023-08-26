#!/home/vishal/anaconda3/envs/tf_gpu/bin/python


import os
import time
import sys

import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    TensorBoard,
)

from ..classes import (
    NetworkMethod,
)
from ..io.saver import Pickle
from ..genutils import (
    print_events,
    check_dir,
)
from .keras_utils import opt
from .numpy_utils import (
    InputState,
)
from .data import (
    ModelData,
    DataHandler,
)


"""
The __init__ method initializes some attributes and checks that all the compulsory arguments are provided. It also sets some default values for some attributes if they are not provided in the arguments.

The check_consistency method checks that the input and output shapes of the model match the expected shapes.

The compile method compiles the model, setting the optimizer, loss function, and metrics.

The set_checkpoints method sets up the checkpointing strategy for the model, which includes saving the model weights and optionally saving TensorBoard logs.

The fit method fits the compiled model to the training data. It checks if the training data has been set, and if not, it gets the data using get_data from the DataHandler or ModelData class, depending on how the object was initialized. The method also accepts some optional arguments, such as the number of epochs, batch size, and whether to shuffle the data. If encoder=True, the method trains the model as an autoencoder.
"""


class KerasModel(
    NetworkMethod
):
    def __init__(
        self, **kwargs
    ):
        compulsory_kwargs = {
            "class_names",
            "input_states",
            "preprocess_tag",
            "run_name",
        }
        assert compulsory_kwargs.issubset(
            set(
                kwargs.keys()
            )
        )
        self.input_states = kwargs.get(
            "input_states"
        )
        self.class_names = (
            kwargs.get(
                "class_names"
            )
        )
        self.num_classes = len(
            self.class_names
        )
        self.preprocess_tag = kwargs.get(
            "preprocess_tag",
            "",
        )
        self.model = None
        self.network_type = kwargs.get(
            "network_type",
            "jet_image",
        )
        self.compiled = False
        self.history = None
        self.data_handler = (
            None
        )
        self.lr = kwargs.get(
            "lr", 0.0001
        )
        self.loss = kwargs.get(
            "loss",
            "categorical_crossentropy",
        )
        self.opt_name = (
            kwargs.get(
                "opt",
                "Nadam",
            )
        )
        self.model_type = (
            kwargs.get(
                "model_type",
                "",
            )
        )
        if (
            self.model_type
            == "autoencoder"
        ):
            self.loss = "mean_squared_error"
        self.save = (
            kwargs.get(
                "save", False
            )
        )
        self.run_name = (
            kwargs.get(
                "run_name"
            )
        )
        if (
            "data_handler"
            in kwargs
        ):
            self.in_data = (
                DataHandler(
                    **kwargs
                )
            )
        else:
            self.in_data = (
                ModelData(
                    **kwargs
                )
            )
        self.network_file = (
            None
        )
        self.save_run = False
        self.save_dir = None
        self.train_data = (
            None
        )
        self.val_data = None
        self.history_save_path = (
            None
        )

    def check_consistency(
        self, model
    ):
        print(
            "Checking consistency..."
        )
        if (
            type(model.input)
            != list
        ):
            if (
                len(
                    model.input.shape
                )
                > 1
            ):
                i = None
            else:
                i = None
            assert (
                model.input.shape[
                    1:i
                ]
                == self.input_states[
                    0
                ].shape
            ), (
                ""
                + str(
                    model.input.shape[
                        1:i
                    ]
                )
                + " "
                + str(
                    self.input_states[
                        0
                    ].shape
                )
            )
        else:
            assert len(
                model.input
            ) == len(
                self.input_states
            )
            for (
                input_state,
                network_input,
            ) in zip(
                self.input_states,
                model.input,
            ):
                if (
                    len(
                        network_input.shape
                    )
                    > 1
                ):
                    i = -1
                else:
                    i = None
                # assert network_input.shape[:i]==input_state.shape
        if (
            self.network_type
            != "autoencoder"
        ):
            assert (
                self.num_classes
                == model.output.shape[
                    1
                ]
                and len(
                    model.output.shape
                )
                == 2
            )
        return

    def compile(
        self,
        model,
        check=True,
        **kwargs
    ):
        if (
            self.model_type
            != "autoencoder"
            and check
        ):
            self.check_consistency(
                model
            )
        self.model = model
        self.compiled = True
        if "loss" in kwargs:
            self.loss = (
                kwargs.get(
                    "loss"
                )
            )
        if (
            self.model_type
            == "autoencoder"
        ):
            default_metric = [
                "mean_squared_error"
            ]
        else:
            default_metric = [
                "acc"
            ]
        metrics = kwargs.get(
            "metrics",
            default_metric,
        )
        self.lr = kwargs.get(
            "lr", self.lr
        )
        self.opt_name = kwargs.get(
            "optimizer",
            self.opt_name,
        )
        opt_kwargs = (
            kwargs.get(
                "opt_kwargs",
                {},
            )
        )
        self.model.compile(
            loss=self.loss,
            metrics=metrics,
            optimizer=opt(
                self.lr,
                self.opt_name,
                **opt_kwargs
            ),
        )
        return self.model

    def set_checkpoints(
        self,
        include_tensorboard=False,
        early_stopping=False,
        **kwargs
    ):
        count = (
            len(
                os.listdir(
                    self.in_data.model_checkpoints_path
                )
            )
            + 1
        )
        checkpoints_path = check_dir(
            os.path.join(
                self.in_data.model_checkpoints_path,
                "run_"
                + str(count),
            )
        )
        self.history_save_path = (
            checkpoints_path
        )
        try:
            os.system(
                "cp network.py "
                + self.history_save_path
            )
        except (
            Exception
        ) as e:
            print(e)
        pwd = os.getcwd()
        os.chdir(
            self.history_save_path
        )
        with open(
            "network_specs.dat",
            "w+",
        ) as File:
            self.model.summary(
                print_fn=lambda x: File.write(
                    x + "\n"
                )
            )
            File.write(
                "\nClasses: "
            )
            for (
                item
            ) in (
                self.class_names
            ):
                File.write(
                    item
                    + "   "
                )
            File.write(
                "Optimizer: "
                + self.opt_name
                + "\n"
            )
            File.write(
                "Loss: "
                + self.loss
                + "\n"
            )
            File.write(
                "Learning rate: "
                + str(
                    self.lr
                )
                + "\n"
            )
            File.write(
                "Batch size: "
                + str(
                    kwargs.get(
                        "batch_size"
                    )
                )
            )
            File.write("\n")
        os.chdir(pwd)
        filename = "model"
        period = kwargs.get(
            "period", 1
        )
        if kwargs.get(
            "hyper_opt",
            False,
        ):
            checkpoint = []
        else:
            if (
                self.model_type
                != "autoencoder"
            ):
                checkpoint = [
                    ModelCheckpoint(
                        filepath=os.path.join(
                            checkpoints_path,
                            filename
                            + "_{epoch:02d}_{val_auc:.5f}.hdf5",
                        ),
                        monitor="val_auc",
                        save_best_only=True,
                        period=period,
                        mode="max",
                        verbose=0,
                    )
                ]
            else:
                checkpoint = [
                    ModelCheckpoint(
                        filepath=os.path.join(
                            checkpoints_path,
                            filename
                            + "_{epoch:02d}_{val_mean_squared_error:.5f}.hdf5",
                        ),
                        save_best_only=True,
                        period=period,
                    )
                ]
        if include_tensorboard:
            tensorboard_path = check_dir(
                self.in_data.run_path,
                "tensorboard",
            )
            checkpoin.append(
                TensorBoard(
                    log_dir=tensorboard_path,
                    write_grads=True,
                    write_graph=True,
                    write_images=True,
                    update_freq="batch",
                )
            )
        if early_stopping:
            checkpoint.append(
                keras.callbacks.EarlyStopping(
                    monitor=kwargs.get(
                        "monitor",
                        "val_acc",
                    ),
                    min_delta=kwargs.get(
                        "min_delta",
                        3,
                    ),
                    patience=kwargs.get(
                        "patience",
                        5,
                    ),
                    verbose=1,
                    mode=kwargs.get(
                        "mode",
                        "max",
                    ),
                    restore_best_weights=True,
                )
            )
        return checkpoint

    def fit(
        self,
        verbose=1,
        batch_size=300,
        shuffle=True,
        epochs=5,
        encoder=False,
        **kwargs
    ):
        if (
            self.train_data
            is None
        ):
            (
                X,
                Y,
                X_t,
                Y_t,
            ) = (
                self.in_data.get_data()
            )
            self.train_data = (
                X,
                Y,
            )
            self.val_data = (
                X_t,
                Y_t,
            )
        else:
            (
                X,
                Y,
            ) = (
                self.train_data
            )
            (
                X_t,
                Y_t,
            ) = self.val_data
        # print (X.shape,Y.shape,X_t.shape,Y_t.shape)
        # sys.exit()
        if self.save:
            checkpoints = self.set_checkpoints(
                batch_size=batch_size,
                **kwargs
            )
        else:
            checkpoints = (
                None
            )
        if epochs == 0:
            print(
                "Checked..."
            )
            sys.exit()
        # print (X.shape,Y.shape,X_t.shape,Y_t.shape)
        self.History = self.model.fit(
            X,
            Y,
            batch_size=batch_size,
            callbacks=checkpoints,
            verbose=verbose,
            epochs=epochs,
            shuffle=shuffle,
            validation_data=(
                X_t,
                Y_t,
            ),
        )
        if self.save:
            if encoder:
                encoder.save(
                    os.path.join(
                        self.history_save_path,
                        "encoder.h",
                    )
                )
            Pickle(
                self.History.history,
                "history",
                save_path=self.history_save_path,
            )
            self.model.save(
                os.path.join(
                    self.history_save_path,
                    "model.hdf5",
                )
            )
        return (
            self.History.history
        )
