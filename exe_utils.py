from .hep.methods import (
    DelphesNumpy,
)
from .hep.data import (
    NumpyEvents,
)
from .genutils import (
    print_events,
    merge_flat_dict,
)


def get_from_indices(
    events,
    indices,
    keys=None,
):
    """
    Extract selected keys and corresponding indices from a dictionary of
    events.

    Args:
        events (dict): A dictionary containing event data.
        indices (array-like): Indices to select from the event data.
        keys (list of str, optional): A list of keys to extract from the
            events. If None, all keys are extracted.

    Returns:
        dict: A new dictionary containing the selected keys and their
        corresponding values for each index in `indices`. The returned value
        is
    """
    if keys is None:
        keys = list(events.keys())
    return_array = {}
    for (
        key,
        val,
    ) in events.items():
        if key not in keys:
            continue
        return_array[key] = val[indices]
    return return_array


def get_delphes(run_names, **kwargs):
    """
    Load and print Delphes event data from one or more run names.

    Args:
        run_names (str or list of str): The name(s) of the Delphes run(s) to
            load.
        **kwargs: Additional keyword arguments for DelphesNumpy
            initialization.

    Returns:
        None
    """
    if isinstance(run_names, str):
        run_names = [run_names]
    for run_name in run_names:
        now = DelphesNumpy(run_name, **kwargs)
        for events in now:
            print_events(events)
    return


def get_numpy_events(run_name, runs="first", **kwargs):
    """
    Load and return NumpyEvents data from a run name.

    Args:
        run_name (str): The name of the run to load.
        runs (str, optional): Specify how to handle multiple runs if present
            ('first' to return the first run, 'merge' to merge all runs).
            Default is 'first'.
        **kwargs: Additional keyword arguments for NumpyEvents initialization.

    Returns:
        dict: A dictionary containing the loaded event data.
    """
    now = NumpyEvents(run_name, mode="r", **kwargs)
    return_dict = {}
    for item in now:
        if runs == "first":
            return item
        else:
            return_dict = merge_flat_dict(
                return_dict,
                item,
            )
    return return_dict
