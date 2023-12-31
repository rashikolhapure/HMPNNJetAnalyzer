import os
import sys
import __main__
from typing import Dict, List, Union

import numpy as np

try:
    import uproot
except ModuleNotFoundError as e:
    print(
        e,
        "\nMight create problem in reading root files...",
    )

from ..genutils import (
    check_dir,
    check_file,
    print_events,
)
from .config import (
    Paths,
    FinalStates,
)
from ..io.saver import (
    Unpickle,
    Pickle,
)
from ..classes import (
    PhysicsData,
)


class RootEvents(PhysicsData):
    def __init__(
        self,
        run_name: str,
        *args: tuple,
        **kwargs: dict
    ) -> None:
        """
        Initialize a RootEvents object.

        Parameters:
        ----------
        run_name : str
            The name of the run.
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Attributes:
        ----------
        both_dirs : bool
            Flag indicating whether both directories should be considered.
        select_runs : list
            List of runs to select.
        ignore_runs : list
            List of runs to ignore.
        runs : list
            List of root file paths.
        mg_event_path : str
            MadGraph event path.
        count : int
            Count of processed runs.
        max_count : int
            Maximum count of runs.
        Events : None
            Placeholder for event data.
        return_vals : str
            Return values type.

        Notes:
        -----
        This class initializes a RootEvents object, which is used to manage
        and process data from root files.
        """
        print(
            self.__class__.__name__,
            run_name,
            args,
            kwargs,
        )
        # sys.exit()
        super().__init__(
            run_name=run_name,
            main_dir="madgraph",
            reader_method="DelphesNumpy",
            writer_method="Madgraph",
            extension=".root",
        )
        self.both_dirs = kwargs.get(
            "both_dirs",
            False,
        )
        self.select_runs = kwargs.get("select_runs", [])
        self.ignore_runs = kwargs.get("ignore_runs", [])
        if "root_file_path" in kwargs:
            self.runs = [kwargs.get("root_file_path")]
            self.mg_event_path = os.path.abspath(
                os.path.dirname(kwargs.get("root_file_path"))
            )
            assert os.access(
                self.mg_event_path,
                os.F_OK,
            )
        else:
            self.runs = check_file(
                ".root",
                self.mg_event_path,
                tag=kwargs.get("tag", ""),
                run_tag=kwargs.get(
                    "run_tag",
                    "None",
                ),
            )
            self.runs = [item for item in self.runs if "delphes" in item]
            if self.both_dirs:
                self.runs.append(
                    check_file(
                        ".root",
                        Paths.other_dir,
                        tag=kwargs.get(
                            "tag",
                            "",
                        ),
                        run_tag=kwargs.get(
                            "run_tag",
                            "None",
                        ),
                    )
                )
            if len(self.select_runs) != 0:
                new = [
                    item
                    for item in self.runs
                    if item.split("/")[-2].split("_")[1] in self.select_runs
                ]
                self.runs = new
                print(self.runs)
            if len(self.ignore_runs) != 0:
                new = [
                    item
                    for item in self.file
                    if item.split("/")[-2].split("_")[1]
                    not in self.ignore_runs
                ]
                self.files = new
        (
            self.count,
            self.max_count,
        ) = 0, len(self.runs)
        print(self.max_count)
        self.Events = None
        self.return_vals = kwargs.get(
            "return_vals",
            "Delphes",
        )

    def read(self, root_file: str) -> uproot.tree.TTree:
        """class method to directly select <final_state> with list of
        <attributes> at <indices> from <root_file>. returns a numpy
        array of either len(indices) with variable shape depending
        on the <final_state>

        Class method to directly select <final_state> with list of
        <attributes> at <indices> from <root_file>.

        Parameters:
        ----------
        root_file : str
            The path to the root file to read.

        Returns:
        -------
        numpy.ndarray
            A numpy array with the selected data.

        Notes:
        -----
        This method reads data from a root file and selects a specific
        final state with a list of attributes at given indices. The resulting
        data is returned as a numpy array with variable shape depending on
        the final state.
        """
        print(
            "root_file_path:",
            root_file,
        )
        self.Events = uproot.open(root_file)["Delphes"]
        return self.Events

    def __next__(self) -> Union[np.ndarray, Dict[str, Union[np.ndarray, int, str]]]:
        """
        Retrieve the next item from the iterator.

        Returns:
        -------
        numpy.ndarray or dict
            Depending on the configuration, it returns either a numpy array or
            a dictionary containing selected data.

        Raises:
        ------
        StopIteration
            When there are no more items to iterate over.
        """
        if self.count < self.max_count:
            self.count += 1
            splitted = os.path.split(self.runs[self.count - 1])
            print(splitted[0])
            return_dict = {
                "Delphes": self.read(self.runs[self.count - 1]),
                "path": splitted[0],
                "filename": splitted[1],
                "index": self.count - 1,
            }
            if isinstance(self.return_vals, str):
                return return_dict[self.return_vals]
            else:
                return {item: return_dict[item] for item in self.return_vals}
        else:
            raise StopIteration


class NumpyEvents(PhysicsData):
    def __init__(
        self,
        run_name: str,
        *args: tuple,
        **kwargs: dict
    ) -> None:
        """
        Initialize a NumpyEvents instance.

        Parameters:
        -----------
        run_name : str
            The name of the run.
        kwargs : dict
            Additional keyword arguments.

        Keyword Args:
        -------------
        mode : str
            The mode of operation, either 'r' for reading or 'w' for writing.
        prefix : str, optional
            A prefix to be added to the filename, default is 'no_tag'.
        select_runs : list, optional
            List of run names to include (default is empty).
        ignore_runs : list, optional
            List of run names to ignore (default is empty).
        both_dirs : bool, optional
            Whether to search in both madgraph directories (default is False).

        Returns:
        --------
        None
        """
        assert "mode" in kwargs and kwargs.get("mode") in {"r", "w"}
        print(
            self.__class__.__name__,
            run_name,
            args,
            kwargs,
        )
        self.select_runs = kwargs.get("select_runs", [])
        self.ignore_runs = kwargs.get("ignore_runs", [])
        self.mode = kwargs.get("mode")
        self.prefix = kwargs.get(
            "prefix",
            "no_tag",
        )
        super().__init__(
            run_name=run_name,
            main_dir="madgraph",
            reader_method="BaselineCuts",
            writer_method="DelphesNumpy",
            **kwargs
        )
        self.count = 0
        self.exception = False
        self.both_dirs = kwargs.get(
            "both_dirs",
            False,
        )
        if self.mode == "r":
            self.files = check_file(
                self.prefix
                + "_"
                + kwargs.get(
                    "filename",
                    "delphes",
                )
                + ".pickle",
                self.mg_event_path,
                full_name=True,
                run_tag=kwargs.get(
                    "run_tag",
                    "None",
                ),
            )
            if self.both_dirs:
                self.files.append(
                    check_file(
                        self.prefix
                        + "_"
                        + kwargs.get(
                            "filename",
                            "delphes",
                        )
                        + ".pickle",
                        Paths.other_dir,
                        full_name=True,
                        run_tag=kwargs.get(
                            "run_tag",
                            "None",
                        ),
                    )
                )
            if len(self.select_runs) != 0:
                new = [
                    item
                    for item in self.files
                    if item.split("/")[-2].split("_")[1] in self.select_runs
                ]
                self.files = new
            if len(self.ignore_runs) != 0:
                new = [
                    item
                    for item in self.file
                    if item.split("/")[-2].split("_")[1]
                    not in self.ignore_runs
                ]
                self.files = new
            print(self.files)
            self.current_run = None
            self.max_count = len(self.files)
        else:
            self.current_run = None
            self.current_events = None
            try:
                self.max_count = kwargs["max_count"]
            except KeyError:
                raise KeyError(
                    "For write instance of NumpyEvents, provide max_count"
                )

    def __next__(self) -> None:
        if self.count < self.max_count:
            self.count += 1
            if self.mode == "r":
                splitted = os.path.split(self.files[self.count - 1])
                self.current_run = splitted[0]
                return Unpickle(
                    splitted[1],
                    load_path=splitted[0],
                )
            else:
                if self.exception:
                    Pickle(
                        self.current_events,
                        self.prefix + "_" + "delphes.pickle",
                        save_path=os.path.join(
                            self.mg_event_path,
                            self.current_run,
                        ),
                    )
                return
        else:
            raise StopIteration


class PassedEvents(PhysicsData):
    """
    A class for handling passed events data.

    This class provides methods for reading and writing passed events data.

    Parameters:
    ----------
    run_name : str
        The name of the run.
    kwargs : dict
        Additional keyword arguments.

    Attributes:
    ----------
    mode : str
        The mode of operation, either 'r' for reading or 'w' for writing.
    select_runs : List[str]
        A list of selected runs to include.
    tag : str
        A tag used for filenames.
    save : bool
        A flag indicating whether to save the data.
    exception : bool
        A flag indicating whether an exception occurred.
    current_events : Dict[str, Union[int, float, List[Union[int, float]]]]
        The current events data.
    current_run : str
        The path of the current run.
    count : int
        A counter for the current iteration.
    both_dirs : bool
        A flag indicating whether to search in both directories.
    remove_keys : List[str]
        A list of keys to remove from the data.

    Methods:
    -------
    __next__(): Dict[str, Union[int, float, List[Union[int, float]]]]
        Retrieve the next item from the iterator.

    Raises:
    ------
    StopIteration
        When there are no more items to iterate over.
    """


    def __init__(self, run_name: str, *args: tuple, **kwargs: dict) -> None:
        super().__init__(
            run_name=run_name,
            reader_method="Preprocess",
            writer_method="BaselineCuts",
        )
        self.select_runs = kwargs.get("select_runs", [])
        assert "mode" in kwargs
        self.tag = kwargs.get("tag", "")
        self.mode = kwargs.get("mode")
        self.save = kwargs.get("save", False)
        assert self.mode in (
            "r",
            "w",
        )
        self.exception = False
        self.current_events = None
        self.current_run = None
        self.count = 0
        self.both_dirs = kwargs.get(
            "both_dirs",
            False,
        )
        self.remove_keys = kwargs.get("remove_keys", [])
        if self.mode == "w":
            assert "max_count" in kwargs
            self.max_count = kwargs["max_count"]
        else:
            self.files = check_file(
                kwargs.get(
                    "filename",
                    "passed_",
                )
                + self.tag,
                self.mg_event_path,
                full_name=True,
                run_tag=kwargs.get(
                    "run_tag",
                    "None",
                ),
            )
            if self.both_dirs:
                self.files.append(
                    check_file(
                        kwargs.get(
                            "filename",
                            "passed_",
                        )
                        + self.tag,
                        Paths.other_dir,
                        full_name=True,
                        run_tag=kwargs.get(
                            "run_tag",
                            "None",
                        ),
                    )
                )
            if len(self.select_runs) != 0:
                new = [
                    item
                    for item in self.files
                    if item.split("/")[-2].split("_")[1] in self.select_runs
                ]
                self.files = new
            self.max_count = len(self.files)

    def __next__(self) -> Dict[str, Union[int, float, List[Union[int, float]]]]:
        if self.count < self.max_count:
            self.count += 1
            if self.mode == "w":
                try:
                    print_events(self.current_events)
                except Exception:
                    pass
                if self.save:
                    Pickle(
                        self.current_events,
                        "passed_" + self.tag,
                        save_path=os.path.join(
                            self.mg_event_path,
                            self.current_run,
                        ),
                    )
            else:
                splitted = os.path.split(self.files[self.count - 1])
                self.current_run = splitted[0]
                self.current_events = Unpickle(
                    splitted[1],
                    load_path=splitted[0],
                )
                for item in self.remove_keys:
                    try:
                        self.current_events.pop(item)
                    except KeyError as e:
                        print(
                            e,
                            item,
                            " not in dict. Careful for cut consistency!",
                        )
                return self.current_events
        else:
            raise StopIteration


class PreProcessedEvents(PhysicsData):
    """
    A class for handling preprocessed events data.

    This class provides methods for reading and writing preprocessed
    events data.

    Parameters:
    ----------
    run_name : str
        The name of the run.
    kwargs : dict
        Additional keyword arguments.

    Attributes:
    ----------
    mode : str
        The mode of operation, either 'r' for reading or 'w' for writing.
    select_runs : List[str]
        A list of selected runs to include.
    tag : str
        A tag used for filenames.
    exception : bool
        A flag indicating whether an exception occurred.
    current_events : Dict[str, Union[int, float, List[Union[int, float]]]]
        The current events data.
    current_run : str
        The path of the current run.
    count : int
        A counter for the current iteration.
    both_dirs : bool
        A flag indicating whether to search in both directories.

    Methods:
    -------
    __next__(): Dict[str, Union[int, float, List[Union[int, float]]]]
        Retrieve the next item from the iterator.

    Raises:
    ------
    StopIteration
        When there are no more items to iterate over.
    """
    def __init__(self, run_name: str, *args: tuple, **kwargs: dict) -> None:
        super().__init__(
            run_name=run_name,
            reader_method="Network",
            writer_method="PreProcess",
        )
        assert "mode" in kwargs
        self.select_runs = kwargs.get("select_runs", [])
        self.tag = kwargs.get("tag", "")
        self.mode = kwargs.get("mode")
        assert self.mode in (
            "r",
            "w",
        )
        self.exception = False
        self.current_events = None
        self.current_run = None
        self.count = 0
        self.both_dirs = kwargs.get(
            "both_dirs",
            False,
        )
        if self.mode == "w":
            assert "max_count" in kwargs
            self.max_count = kwargs["max_count"]
        else:
            self.files = check_file(
                kwargs.get(
                    "filename",
                    "preprocessed_",
                )
                + self.tag
                + ".h",
                self.mg_event_path,
                full_name=True,
                run_tag=kwargs.get(
                    "run_tag",
                    "None",
                ),
            )
            if self.both_dirs:
                self.files.append(
                    check_file(
                        kwargs.get(
                            "filename",
                            "preprocessed_",
                        )
                        + self.tag
                        + ".h",
                        Paths.other_dir,
                        full_name=True,
                        run_tag=kwargs.get(
                            "run_tag",
                            "None",
                        ),
                    )
                )
            if len(self.select_runs) != 0:
                new = [
                    item
                    for item in self.files
                    if item.split("/")[-2].split("_")[1] in self.select_runs
                ]
                self.files = new
            self.max_count = len(self.files)

    def __next__(self) -> Dict[str, Union[int, float, List[Union[int, float]]]]:
        if self.count < self.max_count:
            self.count += 1
            if self.mode == "w":
                print_events(self.current_events)
                Pickle(
                    self.current_events,
                    "preprocessed_" + self.tag + ".h",
                    save_path=os.path.join(
                        self.mg_event_path,
                        self.current_run,
                    ),
                )
            else:
                splitted = os.path.split(self.files[self.count - 1])
                self.current_run = splitted[0]
                self.current_events = Unpickle(
                    splitted[1],
                    load_path=splitted[0],
                )
                return self.current_events
        else:
            raise StopIteration


if __name__ == "__main__":
    Events = PreProcessedEvents(
        "dijet",
        mode="r",
        tag="try",
    )
    for item in Events:
        print_events(item)
