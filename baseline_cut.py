import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from hep.data import (
    NumpyEvents,
)
from hep.methods import (
    DelphesNumpy,
    BaselineCuts,
    PreProcess,
)
from hep.utils.preprocess_utils import (
    regularize_fatjet,
    translate,
    rotate,
    reflect,
    binner,
)
from genutils import (
    print_events,
    merge_flat_dict,
    check_dir,
)
from io.saver import (
    Pickle,
    Unpickle,
)
from hep.config import (
    FinalStates,
)
from hep.utils.fatjet import (
    FatJet,
)
from hep.utils import (
    root_utils as ru,
)
from exe_utils import (
    get_from_indices,
)


def dijet_cut(
    events: dict,
    logging: bool = False
) -> dict:
    """Apply dijet event selection criteria to the given events.

    The dijet criteria include requiring at least two jets with specific
    transverse momenta and pseudorapidities.


    Parameters:
        events (dict): A dictionary containing event data including "Jet",
            "Track", and "Tower" information.
        logging (bool): Whether to print log messages during processing
            (default is False).

    Returns:
        dict: A dictionary containing selected events that meet the dijet
            criteria.
    """
    (
        passed_met,
        passed_tower,
    ) = ([], [])
    Jets, Tracks, Towers = (
        events["Jet"],
        events["Track"],
        events["Tower"],
    )
    (
        event_index,
        passed_event_index,
    ) = (
        0,
        0,
    )
    ind = FinalStates.index_map
    return_dict = dict()
    passed_indices = []
    for (
        jets,
        tower,
        track,
    ) in zip(Jets, Towers, Tracks):
        event_index += 1
        if event_index % 5000 == 0 and logging:
            print(
                "Event count: ",
                event_index,
                "Passed Events: ",
                passed_event_index,
            )
        array_jets = jets
        if len(jets) < 2:
            continue
        jets = ru.GetTLorentzVector(jets[:, :4])
        if jets[0].Pt() < 150.0:
            continue
        if jets[1].Pt() < 130.0:
            continue
        if abs(jets[0].Eta()) > 4.7:
            continue
        if abs(jets[1].Eta()) > 4.7:
            continue

        if passed_event_index == 0:
            (
                return_dict["Jet"],
                return_dict["Tower"],
            ) = np.array(
                [jets[:2]]
            ), np.array([tower])
            return_dict["jet_delphes"] = [array_jets[:]]
            return_dict["Tower"] = [tower]
            return_dict["Track"] = [track]
        else:
            return_dict["Jet"] = np.concatenate(
                (
                    return_dict["Jet"],
                    np.array([jets[:2]]),
                ),
                axis=0,
            )
            return_dict["Tower"].append(tower)
            return_dict["Track"].append(track)
            return_dict["jet_delphes"].append(array_jets[:])
        passed_indices.append(event_index - 1)
        passed_event_index += 1
    if passed_event_index > 1:
        return_dict["Tower"] = np.array(return_dict["Tower"])
        return_dict["jet_delphes"] = np.array(return_dict["jet_delphes"])
        return_dict["passed_indices"] = np.array(passed_indices)
        return_dict["Track"] = np.array(return_dict["Track"])
    if passed_event_index == 1:
        return {}
    return return_dict
