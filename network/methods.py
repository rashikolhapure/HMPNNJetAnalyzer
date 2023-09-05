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


class KerasModel(NetworkMethod):
    def __init__(self, **kwargs):
        compulsory_kwargs = {
            "class_names",
            "input_states",
            "preprocess_tag",
            "run_name",
        }
        #         optional_kwargs = {"model"}
        """
        Initialize a KerasModel instance with the provided arguments.

        Args:
            **kwargs: Keyword arguments to configure the KerasModel.
                Compulsory arguments:
                    - "class_names": List of class names for classification.
                    - "input_states": List of input states for the model.
                    - "preprocess_tag": Tag for preprocessed data.
                    - "run_name": Name of the model run.

                Optional arguments:
                    - "network_type": Type of network (default: "jet_image").
                    - "lr": Learning rate (default: 0.0001).
                    - "loss": Loss function (default: "categorical_crossentropy").
                    - "opt": Optimizer name (default: "Nadam").
                    - "model_type": Type of the model (e.g., "autoencoder").
                    - "save": Flag to save model (default: False).
                    - "data_handler": DataHandler instance for data loading.
        
        Raises:
            AssertionError: If any of the compulsory arguments are missing.

        Attributes:
            - input_states (list): List of input states for the model.
            - class_names (list): List of class names for classification.
            - num_classes (int): Number of classes.
            - preprocess_tag (str): Tag for preprocessed data.
            - model (keras.models.Model): Keras model.
            - network_type (str): Type of network (default: "jet_image").
            - compiled (bool): Flag indicating if the model is compiled.
            - history (keras.callbacks.History): Training history.
            - data_handler (DataHandler or ModelData): Data handling instance.
            - lr (float): Learning rate (default: 0.0001).
            - loss (str): Loss function (default: "categorical_crossentropy").
            - opt_name (str): Optimizer name (default: "Nadam").
            - model_type (str): Type of the model (e.g., "autoencoder").
            - save (bool): Flag to save model (default: False).
            - run_name (str): Name of the model run.
            - in_data (DataHandler or ModelData): Data handling instance.
            - network_file: Path to the saved network file.
            - save_run (bool): Flag indicating if the run should be saved.
            - save_dir: Directory path for saving the run.
            - train_data: Training data.
            - val_data: Validation data.
            - history_save_path: Path for saving training history.
        """
        assert compulsory_kwargs.issubset(set(kwargs.keys()))
        self.input_states = kwargs.get("input_states")
        self.class_names = kwargs.get("class_names")
        self.num_classes = len(self.class_names)
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
        self.data_handler = None
        self.lr = kwargs.get("lr", 0.0001)
        self.loss = kwargs.get(
            "loss",
            "categorical_crossentropy",
        )
        self.opt_name = kwargs.get(
            "opt",
            "Nadam",
        )
        self.model_type = kwargs.get(
            "model_type",
            "",
        )
        if self.model_type == "autoencoder":
            self.loss = "mean_squared_error"
        self.save = kwargs.get("save", False)
        self.run_name = kwargs.get("run_name")
        if "data_handler" in kwargs:
            self.in_data = DataHandler(**kwargs)
        else:
            self.in_data = ModelData(**kwargs)
        self.network_file = None
        self.save_run = False
        self.save_dir = None
        self.train_data = None
        self.val_data = None
        self.history_save_path = None

    def check_consistency(self, model):
        """
    Check the consistency of the provided Keras model with the expected input and output shapes.

    Args:
        model (keras.models.Model): The Keras model to check.

    Raises:
        AssertionError: If the model's input and output shapes do not match the expected shapes.

    Returns:
        None
    """
        print("Checking consistency...")
        if not isinstance(model.input, list):
            if len(model.input.shape) > 1:
                i = None
            else:
                i = None
            assert model.input.shape[1:i] == self.input_states[0].shape, (
                ""
                + str(model.input.shape[1:i])
                + " "
                + str(self.input_states[0].shape)
            )
        else:
            assert len(model.input) == len(self.input_states)
            for (
                input_state,
                network_input,
            ) in zip(
                self.input_states,
                model.input,
            ):
                if len(network_input.shape) > 1:
                    i = -1
                else:
                    i = None
                # assert network_input.shape[:i]==input_state.shape
        if self.network_type != "autoencoder":
            assert (
                self.num_classes == model.output.shape[1]
                and len(model.output.shape) == 2
            )
        return

    def compile(self, model, check=True, **kwargs):
        """
        Compile the provided Keras model with specified optimizer, loss function, and metrics.

        Args:
            model (keras.models.Model): The Keras model to compile.
            check (bool, optional): Whether to check model consistency with input states. Defaults to True.
            **kwargs: Additional keyword arguments.
                - loss (str, optional): The loss function to use. Defaults to 'categorical_crossentropy' for classification
                tasks and 'mean_squared_error' for autoencoders.
                - metrics (list of str, optional): List of metrics to monitor during training. Defaults to ['acc'] for
                classification tasks and ['mean_squared_error'] for autoencoders.
                - lr (float, optional): Learning rate for the optimizer. Defaults to 0.0001.
                - optimizer (str, optional): The optimizer name. Defaults to 'Nadam'.
                - opt_kwargs (dict, optional): Additional keyword arguments to pass to the optimizer.

        Returns:
            keras.models.Model: The compiled Keras model.

        Raises:
            AssertionError: If model consistency check fails.
        """
        if self.model_type != "autoencoder" and check:
            self.check_consistency(model)
        self.model = model
        self.compiled = True
        if "loss" in kwargs:
            self.loss = kwargs.get("loss")
        if self.model_type == "autoencoder":
            default_metric = ["mean_squared_error"]
        else:
            default_metric = ["acc"]
        metrics = kwargs.get(
            "metrics",
            default_metric,
        )
        self.lr = kwargs.get("lr", self.lr)
        self.opt_name = kwargs.get(
            "optimizer",
            self.opt_name,
        )
        opt_kwargs = kwargs.get(
            "opt_kwargs",
            {},
        )
        self.model.compile(
            loss=self.loss,
            metrics=metrics,
            optimizer=opt(self.lr, self.opt_name, **opt_kwargs),
        )
        return self.model

    def set_checkpoints(
        self, include_tensorboard=False, early_stopping=False, **kwargs
    ):
        """
        Set up checkpointing strategy for the model, which includes saving model weights and optionally saving TensorBoard logs.

        Args:
            include_tensorboard (bool, optional): Whether to include TensorBoard logs. Defaults to False.
            early_stopping (bool, optional): Whether to enable early stopping. Defaults to False.
            **kwargs: Additional keyword arguments.
                - period (int, optional): Number of epochs between checkpoints. Defaults to 1.
                - hyper_opt (bool, optional): Whether this is a hyperparameter optimization run. Defaults to False.
                - batch_size (int, optional): Batch size. Defaults to None.
                - monitor (str, optional): Metric to monitor for early stopping. Defaults to 'val_acc'.
                - min_delta (int, optional): Minimum change in the monitored metric to qualify as an improvement. Defaults to 3.
                - patience (int, optional): Number of epochs with no improvement after which training will be stopped. Defaults to 5.
                - mode (str, optional): One of {'auto', 'min', 'max'}. In 'min' mode, training will stop when the monitored quantity stops decreasing. In 'max' mode, it will stop when the monitored quantity stops increasing. Defaults to 'max'.

        Returns:
            list: A list of Keras callbacks for checkpointing, early stopping, and TensorBoard, if enabled.

        Note:
            - When `hyper_opt` is True, `checkpoint` will be an empty list.
            - The checkpointed models are saved in the model checkpoints directory specified during initialization.

        Raises:
            AssertionError: If an invalid mode is provided.
        """
        count = len(os.listdir(self.in_data.model_checkpoints_path)) + 1
        checkpoints_path = check_dir(
            os.path.join(
                self.in_data.model_checkpoints_path,
                "run_" + str(count),
            )
        )
        self.history_save_path = checkpoints_path
        try:
            os.system("cp network.py " + self.history_save_path)
        except Exception as e:
            print(e)
        pwd = os.getcwd()
        os.chdir(self.history_save_path)
        with open(
            "network_specs.dat",
            "w+",
        ) as File:
            self.model.summary(print_fn=lambda x: File.write(x + "\n"))
            File.write("\nClasses: ")
            for item in self.class_names:
                File.write(item + "   ")
            File.write("Optimizer: " + self.opt_name + "\n")
            File.write("Loss: " + self.loss + "\n")
            File.write("Learning rate: " + str(self.lr) + "\n")
            File.write("Batch size: " + str(kwargs.get("batch_size")))
            File.write("\n")
        os.chdir(pwd)
        filename = "model"
        period = kwargs.get("period", 1)
        if kwargs.get(
            "hyper_opt",
            False,
        ):
            checkpoint = []
        else:
            if self.model_type != "autoencoder":
                checkpoint = [
                    ModelCheckpoint(
                        filepath=os.path.join(
                            checkpoints_path,
                            filename + "_{epoch:02d}_{val_auc:.5f}.hdf5",
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
        """
        Fit the compiled model to the training data.

        Args:
            verbose (int, optional): Verbosity mode (0, 1, or 2). Defaults to 1.
            batch_size (int, optional): Number of samples per gradient update. Defaults to 300.
            shuffle (bool, optional): Whether to shuffle the training data before each epoch. Defaults to True.
            epochs (int, optional): Number of epochs to train the model. Defaults to 5.
            encoder (bool, optional): Whether to train the model as an autoencoder. Defaults to False.
            **kwargs: Additional keyword arguments.
                - Additional arguments to be passed to the set_checkpoints method.

        Returns:
            dict: A dictionary containing training and validation metrics.

        Note:
            - If `self.train_data` and `self.val_data` have not been set, this method uses `get_data` from the DataHandler or ModelData class to fetch the data.
            - If `self.save` is True, the method sets up checkpointing based on the specified keyword arguments.
            - If `encoder` is True, the trained encoder is saved.
            - The training history, model weights, and encoder (if applicable) are saved in the history save path specified during initialization.

        Example:
            To fit the model with custom settings:
            ```
            model.fit(verbose=2, batch_size=128, shuffle=True, epochs=10, encoder=True, monitor='val_loss', patience=3)
            ```
        """
        if self.train_data is None:
            (
                X,
                Y,
                X_t,
                Y_t,
            ) = self.in_data.get_data()
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
            ) = self.train_data
            (
                X_t,
                Y_t,
            ) = self.val_data
        # print (X.shape,Y.shape,X_t.shape,Y_t.shape)
        # sys.exit()
        if self.save:
            checkpoints = self.set_checkpoints(batch_size=batch_size, **kwargs)
        else:
            checkpoints = None
        if epochs == 0:
            print("Checked...")
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
        return self.History.history
