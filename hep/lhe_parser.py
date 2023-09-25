from typing import (
    Dict, 
    Optional, 
    Tuple, 
    Union,
)
from .config import (
    EventAttribute,
)
import hep_ml.hep.utils.root_utils as ru
import gzip
from ROOT import (
    TLorentzVector,
)
import numpy as np
from tqdm import tqdm
import os
import sys
from collections import (
    namedtuple,
)

lhe_particle = namedtuple(
    "lhe_particle",
    [
        "PID",
        "Status",
        "Px",
        "Py",
        "Pz",
        "E",
        "Mass",
        "Name",
    ],
)


PID_to_particle_dict = {
    1: "u",
    2: "d",
    3: "s",
    4: "c",
    5: "b",
    6: "t",
    11: "e-",
    12: "ve",
    13: "mu-",
    14: "vm",
    15: "ta-",
    16: "vt",
    -1: "u~",
    -2: "d~",
    -3: "s~",
    -4: "c~",
    -5: "b~",
    -6: "t~",
    -11: "e+",
    -12: "ve~",
    -13: "mu+",
    -14: "vm~",
    -15: "ta+",
    -16: "vt~",
    9: "g",
    21: "g",
    22: "gamma",
    23: "Z",
    24: "W+",
    -24: "W-",
    25: "h",
    51: "sk",
}
charge_dict = {
    "e+": +1,
    "e-": -1,
    "mu+": +1,
    "mu-": -1,
}
MAP = PID_to_particle_dict
# charged_leptons={"e+","e-","mu+","mu-","ta+","ta-"}
neutrinos = {
    "ve",
    "vm",
    "vt",
    "ve~",
    "vm~",
    "vt~",
}
jets = {
    "u",
    "d",
    "c",
    "s",
    "u~",
    "d~",
    "c~",
    "s~",
    "g",
}  # ,'b','b~'}
b = {"b", "b~"}
h = {"h"}
t = {"t", "t~"}
a = {"gamma"}
Electrons = {"e+", "e-"}
Muons = {"mu+", "mu-"}
Leptons = Electrons.union(Muons)


# leptons=charged_leptons.union(neutrinos)
def read_lhe(
    path: str = ".",
    filename: str = "unweighted_events.lhe",
    final_state_only: bool = True,
    return_weights: bool = False,
    exclude_initial: bool = True,
    return_structured: bool = False,
    length: Optional[int] = None,
    add_attribute: bool = False,
    run_name: Optional[str] = None,
) -> Union[
    np.ndarray,
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    np.ndarray,
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    print(
        "Reading from file:",
        os.path.join(path, filename),
    )
    if filename.endswith(".gz"):
        #
        lhe_file = gzip.open(
            os.path.join(
                path,
                filename,
            ),
            "rt",
        )
        # print (lhe_file.readlines())

        # sys.exit()
        # pwd=os.getcwd()
        # os.chdir(path)
        # os.system('gzip -d '+filename)
        # os.chdir(pwd)
        # filename=filename[:-len('.gz')]
    else:
        lhe_file = open(
            os.path.join(
                path,
                filename,
            ),
            "r",
        )
    event_count, flag = 0, 0
    events = []
    count = 0
    event = False
    lines = lhe_file.readlines()
    lhe_file.close()
    event_weights = []
    for line in tqdm(lines):
        splitted = line.split()
        if "</event>" in splitted:
            if return_structured:
                event_particles = np.array(
                    event_particles,
                    dtype=[
                        (
                            "PID",
                            "i4",
                        ),
                        (
                            "Status",
                            "i4",
                        ),
                        (
                            "Px",
                            "f4",
                        ),
                        (
                            "Py",
                            "f4",
                        ),
                        (
                            "Pz",
                            "f4",
                        ),
                        (
                            "E",
                            "f4",
                        ),
                        (
                            "Mass",
                            "f4",
                        ),
                        (
                            "Name",
                            "string",
                        ),
                    ],
                )
            events.append(event_particles)
            event = False
            count += 1
            if length is not None:
                if count == length:
                    break
        if event:
            if line.startswith("<"):
                continue
            evaluated = [eval(x) for x in splitted]
            if len(evaluated) == 6:
                event_weight = evaluated[2]
                # event_weight=np.sign(evaluated[2])
                event_weights.append(event_weight)
            if len(evaluated) != 13:
                event_particles = []
            else:
                if final_state_only:
                    if evaluated[1] == 1:
                        particle = lhe_particle(
                            PID=evaluated[0],
                            Status=evaluated[1],
                            Px=evaluated[6],
                            Py=evaluated[7],
                            Pz=evaluated[8],
                            E=evaluated[9],
                            Mass=evaluated[10],
                            Name=MAP[evaluated[0]],
                        )
                        event_particles.append(particle)
                        if "debug" in sys.argv:
                            print(particle)
                else:
                    if exclude_initial:
                        if evaluated[1] != -1:
                            particle = lhe_particle(
                                PID=evaluated[0],
                                Status=evaluated[1],
                                Px=evaluated[6],
                                Py=evaluated[7],
                                Pz=evaluated[8],
                                E=evaluated[9],
                                Mass=evaluated[10],
                                Name=MAP[evaluated[0]],
                            )
                            event_particles.append(particle)
                    else:
                        particle = lhe_particle(
                            PID=evaluated[0],
                            Status=evaluated[1],
                            Px=evaluated[6],
                            Py=evaluated[7],
                            Pz=evaluated[8],
                            E=evaluated[9],
                            Mass=evaluated[10],
                            Name=MAP[evaluated[0]],
                        )
                        event_particles.append(particle)
                    if "debug" in sys.argv:
                        print(particle)
        if "<event>" in splitted:
            event = True
    event_weights = np.array(event_weights)
    assert len(event_weights) == len(
        events
    ), "Incompatible weights and events!"
    if add_attribute:
        assert run_name is not None, "Provide run_name to add to attribute!"
        event_attributes = [
            EventAttribute(
                run_name=run_name,
                tag=filename,
                path=os.path.join(
                    path,
                    filename,
                ),
                index=_,
            )
            for _ in range(len(events))
        ]
    if return_structured:
        events = np.array(events)
        event_attributes = np.array(event_attributes)
    if return_weights:
        if add_attribute:
            return (
                events,
                event_weights,
                event_attributes,
            )
        else:
            return (
                events,
                event_weights,
            )
    else:
        if add_attribute:
            return (
                events,
                event_attributes,
            )
        else:
            return events


def get_cross_section(
    path_to_file: str,
) -> Tuple[float, float]:
    """
    Read LHE (Les Houches Event) file and parse its contents into structured data.

    Parameters:
    ----------
    path : str, optional
        The path to the directory containing the LHE file, defaults to the
        current directory.
    filename : str, optional
        The name of the LHE file to read, defaults to "unweighted_events.lhe".
    final_state_only : bool, optional
        Whether to include only final-state particles in the parsed data,
        defaults to True.
    return_weights : bool, optional
        Whether to return event weights, defaults to False.
    exclude_initial : bool, optional
        Whether to exclude initial-state particles in the parsed data,
        defaults to True.
    return_structured : bool, optional
        Whether to return the parsed data as structured numpy arrays,
        defaults to False.
    length : int, optional
        The maximum number of events to read, defaults to None
        (read all events).
    add_attribute : bool, optional
        Whether to add event attributes like run_name, tag, path, and
        index, defaults to False.
    run_name : str, optional
        The name of the run for adding event attributes, required if
        add_attribute is True.

    Returns:
    -------
    events : list or numpy.ndarray of LHEParticle objects
        A list or structured numpy array containing parsed LHEParticle
        objects representing events.
    event_weights : numpy.ndarray, optional
        An array containing event weights if return_weights is True.
    event_attributes : list or numpy.ndarray of EventAttribute objects,optional
        A list or structured numpy array containing event attributes
        if add_attribute is True.

    Notes:
    ------
    - LHEParticle objects must be defined elsewhere in the code for
    this function to work.
    - The function returns parsed LHE events, optional weights,
    and event attributes.
    """
    f = open(path_to_file, "r")
    imp = []
    append = False
    for line in f:
        if "</init>" in line:
            append = False
            break
        if append:
            imp.append(line.split())
        if "<init>" in line:
            append = True
    f.close()
    cross_section = eval(imp[-1][0])
    error = eval(imp[-1][1])
    # print (imp,cross_section,error)
    return (
        cross_section,
        error,
    )


def reverse_dict(dictionary: Dict[str, iter]) -> Dict[iter, str]:
    """dictionary with iterable values, with empty intersection between
    different values, builds return dictionary with all items in value
    and returns the key of the particular iter_val key.

    Reverse a dictionary with iterable values, ensuring no overlap
    between different values.

    Parameters:
    -----------
    dictionary : dict
        A dictionary with iterable values and no overlapping elements.

    Returns:
    --------
    dict
        A new dictionary where the elements from the original values
        become keys, and the keys from the original dictionary
        become values.

    Raises:
    -------
    AssertionError
        If there is an overlap between the elements in the original
        dictionary values.

    """
    return_dict = {}
    for (
        key,
        val,
    ) in dictionary.items():
        assert hasattr(val, "__iter__")
        for item in val:
            assert (
                item not in return_dict
            ), "Found degeneracy, cannot build unique map!"
            return_dict[item] = key
    return return_dict


def convert_to_dict(
    events,
    final_states=None,
    return_vector=False,
    name=True,
    sort=False,
    add_charge=None,
    attributes=None,
):
    """
    Convert a list of LHE events into a dictionary of final states
    with optional attributes.

    Parameters:
    -----------
    events : list
        A list of LHE events to be converted.
    final_states : list
        A list of final states to include in the dictionary.
    return_vector : bool, optional
        Whether to return vectors (e.g., TLorentzVectors) for particles,
        defaults to False.
    name : bool, optional
        Whether to include the particle name in the dictionary keys,
        defaults to True.
    sort : bool, optional
        Whether to sort particles in each final state
        (only for return_vector=True), defaults to False.
    add_charge : list, optional
        A list of particle names for which charge should be added
        to the dictionary, defaults to None.
    attributes : list, optional
        A list of additional particle attributes to include in
        the dictionary when return_vector is False, defaults to None.

    Returns:
    --------
    dict
        A dictionary containing final states and optional attributes.

    Notes:
    ------
    - If return_vector is True, final states will contain TLorentzVector
    objects for each particle.
    - If return_vector is False, attributes must be provided, and
    final states will contain lists of particle attributes.
    - Particle charge can be added to the dictionary for specific
    particles specified in add_charge.

    """
    assert final_states is not None
    if return_vector is False:
        assert (
            attributes is not None
        ), "Provide attributes array for return_vector=False"
    print(
        "Converting to final_states: ",
        final_states,
    )
    return_dict = {item: [] for item in final_states}
    if add_charge is not None:
        for item in add_charge:
            return_dict[item + "_charge"] = []
    reverse_map = reverse_dict(final_states)
    for event in events:
        current = {item: [] for item in final_states}
        if add_charge is not None:
            for item in add_charge:
                current[item + "_charge"] = []
        for particle in event:
            append_key = reverse_map[particle.Name]
            if return_vector:
                current[append_key].append(
                    TLorentzVector(
                        particle.Px,
                        particle.Py,
                        particle.Pz,
                        particle.E,
                    )
                )
                if append_key in add_charge:
                    # print (particle,charge_dict[particle.Name])
                    current[append_key + "_charge"].append(
                        charge_dict[particle.Name]
                    )
            else:
                array = []
                for item in attributes:
                    array.append(
                        getattr(
                            particle,
                            item,
                        )
                    )
                current[append_key].append(array)
                # print (array,particle,attributes)
        for (
            key,
            val,
        ) in current.items():
            if sort and return_vector:
                if not key.endswith("charge"):
                    val = ru.Sort(np.array(val))
            # ru.Print(val,name=key)
            return_dict[key].append(val)
    for (
        key,
        val,
    ) in return_dict.items():
        return_dict[key] = np.array(val)
    print(
        "Returning as numpy.ndarray of {}!".format(
            "TLorentzVector" if return_vector else attributes
        )
    )
    return_dict["final_states"] = final_states
    # sys.exit()
    return return_dict
