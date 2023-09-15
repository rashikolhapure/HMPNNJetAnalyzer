import os
import multiprocessing
import __main__
import sys
import numpy as np
import pickle
import pandas as pd

from ..genutils import (
    merge_flat_dict,
    print_events,
)


def dict_hdf(
    python_dict,
    filename,
    save_path=".",
    key="data",
):
    """
    Converts a Python dictionary to a pandas DataFrame and saves it to an HDF
    file.

    Args:
        python_dict (dict): A dictionary containing the data to be saved.
        filename (str): The name of the file to be saved (without extension).
        save_path (str, optional): The directory where the file should be
            saved (default is the current directory).
        key (str, optional): The name of the HDF key where the data
            should be stored (default is 'data').

    Returns:
        None
    """
    dataframe = pd.DataFrame.from_dict(python_dict)
    pwd = os.getcwd()
    os.chdir(save_path)
    dataframe.to_hdf(
        filename + ".hdf",
        key=key,
        mode="w",
    )
    os.chdir(pwd)
    print(
        filename,
        " saved at ",
        os.path.abspath(save_path),
    )
    return


def Unpickle(
    filename,
    load_path=".",
    verbose=True,
    keys=None,
    extension=".pickle",
    path=None,
    **kwargs,
):
    """load <python_object> from <filename> at location <load_path>"""
    # if len(filename.split('.')) != 1: filename=filename+extension
    """
    Load a Python object from a file using pickle or numpy.load.

    Arguments:
        filename: the name of the file to be loaded
            (with or without extension)
        load_path: the directory where the file is located
            (default is the current directory)
        verbose: if True, print information about the loading process
            (default is True)
        keys: a list of keys to extract from the loaded dictionary
            (only used if the file is a folder)
        extension: the file extension (default is '.pickle')
        path: an optional argument that overrides the value of load_path
            (for backwards compatibility)
        **kwargs: additional keyword arguments to pass to the pickle.load
            function (e.g., encoding)

    Returns:
        the loaded Python object

    """
    if path is not None:
        load_path = path
    if verbose:
        print(
            "Reading {} from {}".format(
                filename,
                load_path,
            )
        )
    if "." not in filename:
        filename = filename + extension
    pwd = os.getcwd()
    if load_path != ".":
        os.chdir(load_path)
    if filename[-4:] == ".npy":
        ret = np.load(
            filename,
            allow_pickle=True,
        )
        if verbose:
            print(
                filename,
                " loaded from ",
                os.getcwd(),
            )
        os.chdir(pwd)
        return ret
    try:
        with open(filename, "rb") as File:
            return_object = pickle.load(
                File,
                **kwargs,
            )
    except Exception as e:
        print(
            e,
            " checking if folder with ",
            filename.split(".")[0],
            " exists..",
        )
        try:
            os.chdir(filename.split(".")[0])
        except Exception as e:
            os.chdir(pwd)
            raise e
        print("exists! loading...")
        return_object = folder_load(keys=keys)
    if verbose:
        print(
            filename,
            " loaded from ",
            os.getcwd(),
        )
    os.chdir(pwd)
    return return_object


def Pickle(
    python_object,
    filename,
    save_path=".",
    verbose=True,
    overwrite=True,
    path=None,
    append=False,
    extension=".pickle",
):
    """save <python_object> to <filename> at location <save_path>"""
    """
    Save a Python object to a file using pickle or numpy.save

    Arguments:
        python_object: the object to be saved
        filename: the name of the file to be saved (with or without extension)
        save_path: the directory where the file should be saved
            (default is the current directory)
        verbose: if True, print information about the saving process
            (default is True)
        overwrite: if False, raise an IOError if the file already exists
            (default is True)
        path: an optional argument that overrides the value of save_path
            (for backwards compatibility)
        append: if True and the file is a dictionary, merge the saved
            dictionary with the previous one (default is False)
        extension: the file extension (default is '.pickle')

    Returns: None

    """
    if "." not in filename and not isinstance(python_object, np.ndarray):
        filename = filename + extension
    if path is not None:
        save_path = path
    if verbose:
        print(
            "Saving {} at {}".format(
                filename,
                save_path,
            )
        )
    pwd = os.getcwd()
    if save_path != ".":
        os.chdir(save_path)
    if not overwrite:
        if filename in os.listdir("."):
            raise IOError("File already exists!")
    if append:
        assert isinstance(python_object, dict)
        prev = Unpickle(filename)
        print_events(prev, name="old")
        python_object = merge_flat_dict(
            prev,
            python_object,
        )
        print_events(
            python_object,
            name="appended",
        )
    if isinstance(python_object, np.ndarray):
        np.save(
            filename,
            python_object,
        )
        suffix = ".npy"
    else:
        try:
            File = open(
                filename,
                "wb",
            )
            pickle.dump(
                python_object,
                File,
            )
        except OverflowError as e:
            File.close()
            os.system("rm " + filename)
            os.chdir(pwd)
            print(
                e,
                "trying to save as numpy arrays in folder...",
            )
            folder_save(
                python_object,
                filename.split(".")[0],
                save_path,
            )
            return
        suffix = ""
    if verbose:
        print(
            filename + suffix,
            " saved at ",
            os.getcwd(),
        )
    os.chdir(pwd)
    return


def folder_save(
    events,
    folder_name,
    save_path,
    append=False,
):
    """
    Save a dictionary of numpy arrays to a folder.

    Args:
        events (dict): A dictionary with string keys and numpy array
            values to be saved to the folder.
        folder_name (str): A string specifying the name of the folder
            to be created.
        save_path (str): A string specifying the path where the folder
            should be created.
        append (bool, optional): A boolean indicating whether to append
            to existing numpy arrays in the folder if they exist
            (default: False).

    Side effects:
    - Creates a folder with the specified name at the specified path.
    - Saves the numpy arrays in the dictionary to the folder using their
        respective keys as filenames.

    Returns:
        None
    """
    pwd = os.getcwd()
    os.chdir(save_path)
    try:
        os.mkdir(folder_name)
    except FileExistsError as e:
        print(
            e,
            "Overwriting...",
        )
    finally:
        os.chdir(folder_name)
    for item in events:
        if append:
            print("appending...")
            events[item] = np.concatenate(
                (
                    np.load(
                        item + ".npy",
                        allow_pickle=True,
                    ),
                    events[item],
                ),
                axis=0,
            )
        if isinstance(events[item], list):
            print("list type found as val, creating directory...")
            os.mkdir(item)
            os.chdir(item)
            for (
                i,
                array,
            ) in enumerate(events[item]):
                np.save(
                    item + str(i),
                    array,
                    allow_pickle=True,
                )
                print(
                    array.shape,
                    "saved at ",
                    os.getcwd(),
                )
            os.chdir("..")
        else:
            np.save(
                item,
                events[item],
                allow_pickle=True,
            )
            print(
                item + ".npy saved at ",
                os.getcwd(),
                "shape = ",
                events[item].shape,
            )
    os.chdir(pwd)
    return


def folder_load(keys=None, length=None):
    """
    Load all the .npy files in the current directory as numpy arrays
    and return a dictionary containing the arrays with keys equal to the
    filenames without the .npy extension.

    Args:
        keys (list, optional): If provided, only load the arrays corresponding
            to the filenames specified in this list.
        length (int, optional): If provided, only load the first `length`
            elements of each array.

    Returns:
        events (dict): A dictionary containing the numpy arrays loaded
            from the .npy files.
    """
    events = dict()
    pwd = os.getcwd()
    for filename in os.listdir("."):
        if os.path.isdir(filename):
            os.chdir(filename)
            events[filename] = [
                np.load(
                    array_files,
                    allow_pickle=True,
                )
                for array_files in os.listdir(".")
            ]
            os.chdir("..")
            continue
        if keys is not None:
            if filename[:-4] not in keys:
                continue
        try:
            events[filename[:-4]] = np.load(
                filename,
                allow_pickle=True,
            )[:length]
        except IOError as e:
            os.chdir(pwd)
            raise e
        else:
            print(
                filename[:-4],
                " loaded to python dictionary...",
            )
    return events


class RunIO:
    def __init__(
        self,
        run_name,
        dir_name,
        re_initialize=False,
        mode="w",
    ):
        """Initialize RunIO object.

        Args:
            run_name (str): Name of the run.
            dir_name (str): Name of the directory to store data in.
            re_initialize (bool): If True, overwrite existing directory.
            mode (str): "w" for write mode, "r" for read mode.
        """
        _MASTER_DIR = "./python_pickles"
        self._mode = mode
        self._pwd = os.getcwd()
        self.run_name = run_name
        try:
            os.chdir(_MASTER_DIR)
        except OSError:
            os.mkdir(_MASTER_DIR)
            os.chdir(_MASTER_DIR)
        if run_name not in os.listdir():
            assert mode == "w", run_name + " directory not found!"
            os.mkdir(self.run_name)
        os.chdir(self.run_name)
        if dir_name not in os.listdir("."):
            assert mode == "w", dir_name + " not found!"
            os.mkdir(dir_name)
        os.chdir(dir_name)
        self.__path = os.path.abspath(".")
        out_dict = {
            "w": "saving",
            "r": "loading",
        }
        print(
            self.__path,
            f" initialized for {out_dict[mode]} run: ",
            self.run_name,
            " with attributes: ",
            dir_name,
        )
        os.chdir(self._pwd)

    def append_to_text(
        self,
        filename,
        re_initialize=False,
    ):
        """
        Create or append to a text file in the directory.

        Parameters:
            filename (str): Name of the text file.
            re_initialize (bool): If True, overwrite existing file.

        Returns:
            file object: The file object corresponding to the opened file.
        """
        pwd = os.getcwd()
        os.chdir(self.__path)
        if filename not in os.listdir(".") or re_initialize:
            File = open(filename, "w")
        else:
            File = open(filename, "a")
        os.chdir(pwd)
        return File

    def save_events(
        self,
        events,
        append=False,
    ):
        """
        Save events as numpy arrays in the directory.

        Parameters:
            events (dict): Dictionary containing events to save.
            append (bool): If True, append to existing array.
        """
        if self._mode == "r":
            raise IOError(
                "Attempting to write in read-only instance of RunIO object"
            )
        pwd = os.getcwd()
        os.chdir(self.__path)
        for item in events:
            if append:
                print("appending...")
                events[item] = np.concatenate(
                    (
                        np.load(
                            item + ".npy",
                            allow_pickle=True,
                        ),
                        events[item],
                    ),
                    axis=0,
                )
            np.save(
                item,
                events[item],
            )
            print(
                item + ".npy saved at ",
                os.getcwd(),
                "shape = ",
                events[item].shape,
            )
        os.chdir(pwd)
        return

    def load_events(self, length=None):
        """
        Load events as numpy arrays from the directory.

        Parameters:
            length (int): Maximum number of items to load.

        Returns:
            dict: Dictionary containing loaded events.
        """
        pwd = os.getcwd()
        os.chdir(self.__path)
        events = dict()
        for filename in os.listdir("."):
            try:
                events[filename[:-4]] = np.load(
                    filename,
                    allow_pickle=True,
                )[:length]
            except IOError as e:
                os.chdir(pwd)
                raise e
            else:
                print(
                    filename[:-4],
                    " loaded to python dictionary...",
                )
        os.chdir(pwd)
        return events


if __name__ == "__main__":
    a = np.random.normal(
        loc=0.0,
        scale=5.0,
        size=(40, 3),
    )
    print(a.shape)
    print(
        crop(a),
        crop(a).shape,
    )
