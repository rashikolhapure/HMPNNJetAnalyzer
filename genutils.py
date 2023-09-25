import os
import multiprocessing
import sys
from optparse import (
    OptionParser,
)
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from hep.config import (
    gen_particle_index,
    particle_to_PID,
    FinalStates,
    PID_to_particle,
)
import pickle
import numpy as np


def print_vectors(
    vectors: Union[np.ndarray, Iterable],
    format: str = "lorentz"
):
    """
    Print arrays of vectors, either in (pt, eta, phi, mass) or
    (px, py, pz, E) format.

    Args:
        vectors (array or iterable): The array or iterable containing the
            vectors to print.
        format (str, optional): The format in which to print the vectors.
            Default is 'lorentz'.

    Returns:
        None
    """

    def print_single(vector):
        if format == "lorentz":
            print(
                f"Px: {vector.px:10.8f}  Py: {vector.py:10.8f}  \
                Pz: {vector.pz:10.8f}  E: {vector.e:10.8f}"
            )
        else:
            print(
                f"Pt: {vector.pt:.8f} Eta: {vector.eta:.8f}  \
                Phi: {vector.phi:.8f} Mass: {vector.mass:.8f}"
            )

    if hasattr(vectors, "__iter__"):
        list(
            map(
                print_single,
                vectors,
            )
        )
    else:
        print_single(vectors)
    return


def rescale(array: np.ndarray) -> np.ndarray:
    """
    Rescale an input array to have zero mean and unit variance along each
    feature dimension.

    Args:
        array (numpy.ndarray): The input array to be rescaled.

    Returns:
        numpy.ndarray: The rescaled array.
    """
    print(array.shape)
    print(
        "Rescaling:\n mean values:",
        np.mean(array, axis=0),
        " Standard Deviations: ",
        np.std(array, axis=0),
    )
    array = (array - np.mean(array, axis=0)) / np.std(array, axis=0)
    print(
        "Rescaling:\n mean values:",
        np.mean(array, axis=0),
        " Standard Deviations: ",
        np.std(array, axis=0),
    )
    return array


def print_particle(
    particle: np.ndarray,
    four_vec: bool = False,
    ind: int = None,
    mass: bool = True,
):
    """
    Print information about a particle.

    Args:
        particle (numpy.ndarray): The particle information array.
        four_vec (bool, optional): Whether to print the 4-vector components.
            Default is False.
        ind (int, optional): The index of the particle (if applicable).
            Default is None.
        mass (bool, optional): Whether to print the mass of the particle.
            Default is True.

    Returns:
        int: Always returns 0.
    """
    order = FinalStates.attributes["Particle"]
    start, stop = 4, -1
    # print (order,particle)
    if ind is not None:
        print(
            "Ind: ",
            ind,
            " ",
            end="",
        )
    [
        print(
            name,
            ": ",
            item,
            "  ",
            end="",
        )
        for name, item in zip(
            order[start:stop],
            particle[start:stop],
        )
    ]
    try:
        print(
            "Particle name:",
            PID_to_particle[particle[gen_particle_index.PID]],
        )
    except KeyError:
        print("Particle name not in SM particles")
    if four_vec:
        print("4-vec:", end="")
        [
            print(
                "  ",
                round(item, 4),
                end="",
            )
            for item in particle[:4]
        ]
        print()
    if mass:
        print(
            "Mass: ",
            particle[gen_particle_index.Mass],
        )
    return 0


def choose_bin(
    events: Dict[str, Union[np.ndarray, Dict]],
    bin_var: str,
    Range: Tuple[float, float],
    var_key: str = "Jet",
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Select events within a specified range for a given variable.

    Args:
        events (dict): Dictionary containing event data.
        bin_var (str): Variable for which to choose events within the
            specified range.
        Range (tuple): Range of values to select events from (e.g.,
            (min_value, max_value)).
        var_key (str, optional): Key specifying the variable to use
            from the event data dictionary. Default is "Jet".
        **kwargs: Additional keyword arguments.

    Returns:
        dict: Dictionary containing selected events within the specified range.
    """
    print(
        "Chossing events in range: ",
        Range,
        " for variable: ",
        bin_var,
    )
    jets = events[var_key]
    if var_key == "FatJet":
        print("Binning to be performed with FatJet!\n")
        input("Press enter to sum Fatjet constituents, or Ctrl + X to abort: ")
        jets = np.expand_dims(
            np.array([np.sum(item) for item in jets]),
            -1,
        )
    print(Range, jets.shape)
    if bin_var == "pt_j0":
        true_indices = [
            i
            for i in range(len(jets))
            if jets[i][0].Pt() >= Range[0] and jets[i][0].Pt() <= Range[1]
        ]
    elif bin_var == "pt_j1":
        true_indices = [
            i
            for i in range(len(jets))
            if jets[i][1].Pt() >= Range[0] and jets[i][1].Pt() <= Range[1]
        ]
    elif bin_var == "eta_j0":
        true_indices = [
            i
            for i in range(len(jets))
            if jets[i][0].Eta() >= Range[0] and jets[i][0].Eta() <= Range[1]
        ]
    elif bin_var == "eta_j1":
        true_indices = [
            i
            for i in range(len(jets))
            if jets[i][1].Eta() >= Range[0] and jets[i][1].Eta() <= Range[1]
        ]
    elif bin_var == "delta_eta":
        true_indices = [
            i
            for i in range(len(jets))
            if abs(jets[i][0].Eta() - jets[i][1].Eta()) >= Range[0]
            and abs(jets[i][0].Eta() - jets[i][1].Eta()) <= Range[1]
        ]
    elif bin_var == "delta_phi":
        true_indices = [
            i
            for i in range(len(jets))
            if abs(jets[i][0].DeltaPhi(jets[i][1])) >= Range[0]
            and abs(jets[i][0].DeltaPhi(jets[i][1])) <= Range[1]
        ]
    elif bin_var == "m_jj":
        true_indices = [
            i
            for i in range(len(jets))
            if np.sum(jets[i]).M() >= Range[0]
            and np.sum(jets[i]).M() <= Range[1]
        ]
    else:
        raise ValueError
    print(true_indices)
    binned_name = bin_var + "_" + str(Range[0]) + "_" + str(Range[1])
    binned_events = {}
    for (
        key,
        val,
    ) in events.items():
        try:
            if key == "EventAttribute":
                val = np.array(val)
            binned_events[key] = val[true_indices]
        except TypeError as e:
            print(e, key)
            if key in {"cut_flow"}:
                binned_events[key] = val
    return binned_events


def cut_counter(
    prev_cut_flow: Dict[str, int],
    current_cut_flow: Dict[str, int]
) -> Dict[str, int]:
    """
    Counts the number of events passing each cut by accumulating the
    current cut flow into the previous one.

    Args:
        prev_cut_flow (dict): The previous cut flow dictionary.
        current_cut_flow (dict): The current cut flow dictionary to be
            accumulated.

    Returns:
        dict: The updated cut flow dictionary.
    """
    if len(prev_cut_flow.keys()) == 0:
        return current_cut_flow
    else:
        for (
            key,
            val,
        ) in current_cut_flow.items():
            if not isinstance(val, int):
                print(
                    key,
                    " has value ",
                    type(val),
                    " skipping...",
                )
                continue
            prev_cut_flow[key] += val
        return prev_cut_flow


def cut_efficiency(
    cut_flow: Dict[str, int],
    verbose: Optional[bool] = False
) -> Dict[str, float]:
    """
    Calculates the efficiency of each cut in the cut flow.

    Args:
        cut_flow (dict): The cut flow dictionary containing the number
            of events for each cut.
        verbose (bool, optional): Whether to print detailed efficiency
            information. Default is False.

    Returns:
        dict: A dictionary containing the efficiency of each cut, including the
            total efficiency.
    """
    order = cut_flow["order"]
    tot = cut_flow["total"]
    efficiencies = {"total_efficiency": cut_flow["passed"] / tot}
    if verbose:
        print(
            "total_efficiency",
            efficiencies["total_efficiency"],
            "Total: ",
            cut_flow["total"],
            "Passed: ",
            cut_flow["passed"],
        )
    count = 0
    for key in order:
        # count+=cut_flow[key]
        efficiencies[key] = cut_flow[key] / cut_flow["total"]
        if verbose:
            print(
                key,
                " :",
                efficiencies[key],
            )
        # tot=tot-cut_flow[key]
    # if verbose: print ("Total rejected: ","\nRemaining: ",
    # tot,"\nSum: ",count+tot)
    # assert count+tot==cut_flow["total"]
    return efficiencies


def dir_ext_count(
    ext: str,
    dir_path: str,
    prefix: str = "",
    suffix: str = ""
) -> List[str]:
    """
    Count files with a specific extension in a directory that match a given
    prefix and suffix.

    Args:
        ext (str): The file extension to count.
        dir_path (str): The directory path to search for files.
        prefix (str, optional): A prefix that files must start with.
            Default is an empty string.
        suffix (str, optional): A suffix that files must end with.
            Default is an empty string.

    Returns:
        list: A list of file paths that match the specified criteria.
    """
    path = []
    for item in os.listdir(dir_path):
        if item.startswith(prefix) and item.endswith(ext):
            print(item, prefix)
            if item[-len(ext) - len(suffix):] == suffix:
                path.append(
                    os.path.join(
                        dir_path,
                        item,
                    )
                )
    return path


def workaround_concatenate(
    to_return: Union[List[Any], np.ndarray],
    to_append: Union[List[Any], np.ndarray],
    return_type: str = "array"
) -> Union[List[Any], np.ndarray]:
    """
    Concatenate two lists or arrays and return the result.

    Args:
        to_return (list or numpy.ndarray): The initial list or array.
        to_append (list or numpy.ndarray): The list or array to append
            to the initial one.
        return_type (str, optional): The return type, either 'array'
            or 'list'. Default is 'array'.

    Returns:
        list or numpy.ndarray: The concatenated list or array.
    """
    return_array = []
    for item in to_return:
        return_array.append(item)
    for item in to_append:
        return_array.append(item)
    if return_type != "array":
        return return_array
    else:
        return np.array(return_array)


def merge_flat_dict(
    append: Dict[str, Union[np.ndarray, List[Any]]],
    temp: Dict[str, Union[np.ndarray, List[Any]]],
    append_length: int = None,
    keys: Union[str, List[str]] = "all",
    exclude: List[str] = []
) -> Dict[str, Union[np.ndarray, List[Any]]]:
    """Merge two dictionaries with numpy arrays as values, combining values
    with matching keys.

    Args:
        append (dict): The initial dictionary to which values will be
            appended.
        temp (dict): The dictionary containing values to append to the initial
            one.
        append_length (int, optional): The length of values to append.
            Default is None.
        keys (list or str, optional): The keys to consider for merging.
            Default "all".
        exclude (list, optional): The keys to exclude from merging.
            Default is an empty list.

    Returns:
        dict: The merged dictionary.
    """
    if len(list(append.keys())) == 0:
        if keys == "all":
            return temp
        else:
            return_dict = {}
            for item in keys:
                return_dict[item] = temp[item]
    if keys == "all":
        keys = temp.keys()
    for item in append:
        if item == "EventAttribute":
            print("Found EventAttribute, concatenating as list...")
            append[item] = workaround_concatenate(
                append[item],
                temp[item][:append_length],
                return_type="list",
            )
            continue
        if item not in keys or item in exclude:
            continue
        if isinstance(append[item], np.ndarray) and isinstance(
            temp[item], np.ndarray
        ):
            try:
                append[item] = np.concatenate(
                    (
                        append[item],
                        temp[item][:append_length],
                    ),
                    axis=0,
                )
            except ValueError as e:
                print(
                    e,
                    "trying workaround method",
                )
                append[item] = workaround_concatenate(
                    append[item],
                    temp[item][:append_length],
                )
        elif isinstance(append[item], list) and isinstance(temp[item], list):
            if "debug" in sys.argv:
                print(
                    "list",
                    item,
                    len(append[item]),
                    len(temp[item]),
                    append[item][0].shape,
                    append[item][1].shape,
                    temp[item][0].shape,
                    temp[item][1].shape,
                )
            assert len(append[item]) == len(temp[item])
            for i in range(len(append[item])):
                append[item][i] = np.concatenate(
                    (
                        append[item][i],
                        temp[item][i][:append_length],
                    ),
                    axis=0,
                )
            if "debug" in sys.argv:
                print(
                    "list",
                    item,
                    len(append[item]),
                    len(temp[item]),
                    append[item][0].shape,
                    append[item][1].shape,
                    temp[item][0].shape,
                    temp[item][1].shape,
                )
            # sys.exit()
    return append


def print_events(
    events: dict,
    name: Optional[str] = None
) -> None:
    """Print nested dictionaries with up to 3 levels, with the final
    value being a numpy.ndarray.

    Args:
        events (dict): The nested dictionary to be printed.
        name (str, optional): A name or label for the printed dictionary.
            Default is None.
    """
    if name:
        print(name)
    for channel in events:
        if hasattr(
            events[channel],
            "shape",
        ) or hasattr(
            events[channel],
            "__len__",
        ):
            if hasattr(
                events[channel],
                "shape",
            ):
                print(
                    "    Final State:",
                    channel,
                    events[channel].shape,
                    f" dtype: {type(events[channel])}",
                )
            elif channel == "EventAttribute":
                print(
                    "    Final State: ",
                    channel,
                    np.array(events[channel]).shape,
                    " dtype: EventAttribute",
                )
            else:
                # try: print ("    Final State:", channel,
                # [item.shape for item in events[channel]],
                # f' dtype: {type(events[channel])}')
                # except AttributeError:
                print(
                    "    Final State:",
                    channel,
                    len(events[channel]),
                    f" dtype: {type(events[channel])}",
                )
            continue
        print(
            "Specs: ",
            channel,
            "\n     Content: ",
            events[channel],
        )
    return


def check_file(
    name: str,
    event_folder: str,
    tag: Optional[str] = "",
    full_name: bool = False,
    suffix: bool = False,
    run_tag: str = "None",
    target_file: Optional[str] = None,
) -> List[str]:
    """Check for files in a specified directory and its subdirectories.

    This function navigates through a directory and its subdirectories and
    searches for files with specific criteria such as name, tag, and suffix.

    Args:
        name (str): The name of the file to search for.
        event_folder (str): The main directory where the search will start.
        tag (str, optional): A tag to match with the beginning of the file
            name.Default is an empty string.
        full_name (bool, optional): If True, matches the full file name. If
            False, only matches the suffix. Default is False.
        suffix (bool, optional): If True, matches the suffix of the file name.
            If False, matches the entire file name. Default is False.
        run_tag (str, optional): A tag to match with the beginning of
            subdirectory names. Default is "None".
        target_file (str, optional): If provided, the function will skip
            directories containing this specific file. Default is None.

    Returns:
        list: A list of file paths that match the specified criteria.
    """
    path = []
    pwd = os.getcwd()
    if pwd != os.path.abspath(event_folder):
        os.chdir(event_folder)
    # print (event_folder,os.listdir(event_folder))
    for item in os.listdir(event_folder):
        # print (os.getcwd()+"/"+item,os.path.isdir(os.getcwd()+"/"+item))
        if os.path.isdir(item):
            files = os.listdir(item)
            if run_tag != "None":
                if run_tag not in {item[: len(run_tag)] for item in files}:
                    continue
            os.chdir(item)
            assert os.access(".", os.W_OK), os.getcwd()
            if target_file is not None:
                if target_file in files:
                    print(
                        target_file,
                        " found in ",
                        os.getcwd(),
                        " skipping ",
                    )
                    os.chdir("..")
                    continue
            for filename in files:
                if full_name:
                    if filename == name:
                        path.append(
                            os.path.join(
                                event_folder,
                                item,
                                filename,
                            )
                        )
                    continue
                if not suffix:
                    if (
                        filename[-len(name):] == name
                        and filename[: len(tag)] == tag
                    ):
                        path.append(
                            os.path.join(
                                event_folder,
                                item,
                                filename,
                            )
                        )
                    continue
                if (
                    filename[-len(name):] == name
                    and filename[len(tag):] == tag
                ):
                    path.append(
                        os.path.join(
                            event_folder,
                            item,
                            filename,
                        )
                    )
            os.chdir("..")
    os.chdir(pwd)
    path.sort()
    return path


def check_dir(path: str) -> str:
    """Check if a directory at the specified path exists. If it doesn't,
    create the directory.

    Args:
        path (str): The path to the directory to check/create.

    Returns:
        str: The absolute path to the created directory.
    """
    pwd = os.getcwd()
    try:
        os.chdir(path)
    except OSError:
        os.mkdir(path)
        os.chdir(path)
    path = os.getcwd()
    os.chdir(pwd)
    return path


def arg_split(
    args: Union[Dict, List, np.ndarray],
    num_cores: int,
    ignore_keys: List[str] = ["cut_flow"],
    verbose: bool = False,
) -> List[Union[Dict, List, np.ndarray]]:
    """to fix: not splitting args with len(array)==num_cores \
        correctly into iterables of single length

    Split the arguments for parallel processing.

    Args:
        args (dict, list, or ndarray): The input arguments to be split.
        num_cores (int): The number of CPU cores to split the arguments for.
        ignore_keys (list, optional): List of keys to ignore during splitting.
            Defaults to ["cut_flow"].
        verbose (bool, optional): If True, print information about the
            splitting process. Defaults to False.

    Returns:
        list: A list of argument dictionaries or arrays, one for each core.
    """
    print(
        "Arge type: ",
        type(args),
    )
    if isinstance(args, (np.ndarray, list)):
        step = int(len(args) / num_cores)
        arg = []
        for i in range(
            0,
            len(args),
            step,
        ):
            try:
                arg.append(args[i: i + step])
            except IndexError:
                arg.append(args[i:])
    elif isinstance(args, dict):
        arg = [dict() for i in range(num_cores)]
        if verbose:
            print(
                "Splitting argument into: ",
                len(arg),
            )
            print_events(
                args,
                name="Original:",
            )
        start_inds = []
        for key in args:
            if key in ignore_keys:
                print(
                    "Ignoring key: ",
                    key,
                )
                continue
            if isinstance(args[key], np.ndarray):
                step = int(len(args[key]) / num_cores) + 1
            else:
                step = int(len(args[key]) / num_cores)
            count = 0
            # print (key,step,np.array(args[key]).shape)
            for key in args:
                if key in ignore_keys:
                    print(
                        "Ignoring key: ",
                        key,
                    )
                    continue
                step = int(len(args[key]) / num_cores) + 1
                count = 0
                for i in range(
                    0,
                    len(args[key]),
                    step,
                ):
                    if i not in start_inds:
                        start_inds.append(i)
                    arg[count][key] = args[key][i: i + step]
                    count += 1

        if verbose:
            core_ind = 0
            print("Splitted dictionaries with: Arg_<start_ind>_<core_ind>")
            for (
                start,
                item,
            ) in zip(
                start_inds,
                arg,
            ):
                print_events(
                    item,
                    name="Arg_" + str(start) + "_" + str(core_ind),
                )
            core_ind += 1
    else:
        raise TypeError(
            "No algorithm for splitting arguments of type: " + type(args)
        )
    return arg


def init_lock(l: multiprocessing.Lock) -> None:
    global lock
    lock = l


def pool_splitter(
    function: Callable,
    args: Union[Dict, List, np.ndarray],
    num_cores: int = multiprocessing.cpu_count(),
    exclude: List[str] = [],
    ignore_keys: List[str] = ["cut_flow"],
    add_keys: List[str] = [],
    with_lock: bool = False,
    verbose: bool = False,
) -> Union[Dict, np.ndarray]:
    """Utility function for multiprocessing any function with a single
    argument of either numpy.ndarray or a flat dictionary with
    numpy.ndarray values.

    Args:
        function (callable): The function to be parallelized.
        args (dict, list, or ndarray): The input arguments to be split.
        num_cores (int, optional): The number of CPU cores to use for
            parallel processing. Defaults to the number of CPU cores
            available.
        exclude (list, optional): List of keys to exclude from the result
            dictionary. Defaults to an empty list.
        ignore_keys (list, optional): List of keys to ignore during argument
            splitting. Defaults to ["cut_flow"].
        add_keys (list, optional): List of keys to add to each argument
            dictionary. Defaults to an empty list.
        with_lock (bool, optional): If True, use a lock for multiprocessing.
            Defaults to False.
        verbose (bool, optional): If True, print information about the
            splitting process. Defaults to False.

    Returns:
        dict or ndarray: The result of the parallelized function.
    """
    add_dict = {}
    for item in ignore_keys + add_keys:
        if item in args:
            add_dict[item] = args.pop(item)
    arg = arg_split(
        args,
        num_cores,
        ignore_keys=ignore_keys,
        verbose=verbose,
    )
    arg = [item for item in arg if item]
    for (
        key,
        val,
    ) in add_dict.items():
        if key in add_keys:
            for item in arg:
                item[key] = val
    if len(arg) < num_cores:
        print(
            "Not enough arguments to split in :",
            num_cores,
            " cores\nReducing number of cores",
        )
        num_cores = len(arg)
    if with_lock:
        print("Initialising lock...")
        l = multiprocessing.Lock()
        p = multiprocessing.Pool(
            processes=num_cores,
            initializer=init_lock,
            initargs=(l,),
        )
    else:
        p = multiprocessing.Pool(processes=num_cores)
    if num_cores > 1:
        print(
            "Splitting "
            + function.__name__
            + " on "
            + str(num_cores)
            + " cores..."
        )
    try:
        data = p.map(function, arg)
    except KeyboardInterrupt as ki:
        p.close()
        raise ki
    finally:
        p.close()
    print("Done!")
    if isinstance(data[0], np.ndarray):
        return np.concatenate(data, axis=0)
    elif isinstance(data[0], dict):
        # if len(data)>16:
        #    print ("Splitting concatenation on 8 cores...")
        #    p=multiprocessing.Pool(processes=8)
        #    new_args=arg_split(data,8)
        #    data=p.map(combine_dict,new_args)
        #    p.close()
        data = combine_dict(
            data,
            exclude=exclude,
        )
        data.update(add_dict)
        return data
    else:
        return data


def seperate_classes(
    data: Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]],
    class_names: List[str],
) -> Dict[str, Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]]:
    """
    Separate data into different classes based on class labels.

    Args:
        data (dict): The input data dictionary containing features
            and labels.
        class_names (list): List of class names to assign to the separated
            classes.

    Returns:
        dict: A dictionary containing separate classes with their respective
            features and labels.
    """
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
    X0, Y0 = (
        X[class_0],
        Y[class_0],
    )
    X1, Y1 = (
        X[class_1],
        Y[class_1],
    )
    # if not class_names: self.class_names=("class_0","class_1")
    return {
        class_names[0]: {
            "X": X0,
            "Y": Y0,
        },
        class_names[1]: {
            "X": X1,
            "Y": Y1,
        },
    }


def concatenate_list(
    data: List[Dict[str, Any]],
    verbose: bool = False,
    **kwargs: Any,
) -> Dict[str, np.ndarray]:
    """
    Concatenate a list of dictionaries into a single dictionary.

    Args:
        data (list): A list of dictionaries to be concatenated.
        verbose (bool, optional): If True, print information about the
            concatenated data. Default is False.
        **kwargs: Additional keyword arguments to be included in the
            concatenated dictionary.

    Returns:
        dict: A dictionary containing concatenated data from the input list
            of dictionaries.
    """
    count = 0
    if verbose:
        for item in data:
            print("-------------------------------------------------")
            for particle in item:
                try:
                    print(
                        particle,
                        item[particle].shape,
                    )
                except AttributeError:
                    print(
                        particle,
                        len(item[particle]),
                    )
    return_dict = {}
    for item in data:
        for particle in item:
            if particle not in return_dict:
                return_dict[particle] = []
            for event in list(item[particle]):
                return_dict[particle].append(event)
    for item in return_dict:
        return_dict[item] = np.array(return_dict[item])
    if "cut_flow" in kwargs:
        return_dict["cut_flow"] = kwargs.get("cut_flow")
    return return_dict


def combine_dict(data: List[Dict[str, Any]], exclude: List[str] = []) -> Dict[str, Any]:
    """
    Combine a list of dictionaries into a single dictionary.

    Args:
        data (list): A list of dictionaries to be combined.
        exclude (list, optional): A list of keys to be excluded from the
            combination. Default is an empty list.

    Returns:
        dict: A dictionary containing the combined data from the input list of
            dictionaries.
    """
    print("Combining dictionaries...")
    # sys.exit()
    return_dict = dict()
    cont, count = False, 0
    if "cut_flow" in data[0]:
        sum_dicts = [data[i].pop("cut_flow") for i in range(len(data))]
        cut_flow = {}
        for item in sum_dicts:
            cut_flow = cut_counter(
                cut_flow,
                item,
            )
        add = True
    else:
        add = False
    for i in range(len(data)):
        if len(data[i].keys()) == 0:
            continue
        for key in data[i]:
            if key in exclude:
                continue
            if count == 0:
                return_dict[key] = data[i][key]
            else:
                if isinstance(data[i][key], list):
                    return_dict[key] = return_dict[key] + data[i][key]
                else:
                    try:
                        return_dict[key] = np.concatenate(
                            (
                                return_dict[key],
                                data[i][key],
                            ),
                            axis=0,
                        )
                    except ValueError as E:
                        print(
                            E,
                            "\nCould not concatenate as numpy array, appending as lists...",
                        )
                        if add:
                            kwargs = {"cut_flow": cut_flow}
                        return concatenate_list(
                            data,
                            **kwargs,
                        )
        count += 1
    for key in exclude:
        return_dict[key] = data[0][key]
    if add:
        return_dict["cut_flow"] = cut_flow
    return return_dict
