import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from hep_ml.hep.data import (
    NumpyEvents,
)
from hep_ml.hep.methods import (
    DelphesNumpy,
    BaselineCuts,
    PreProcess,
)
from hep_ml.hep.utils.preprocess_utils import (
    regularize_fatjet,
    translate,
    rotate,
    reflect,
    binner,
)
from hep_ml.genutils import (
    print_events,
    merge_flat_dict,
    check_dir,
)
from hep_ml.io.saver import (
    Pickle,
    Unpickle,
)
from hep_ml.hep.config import (
    FinalStates,
)
from hep_ml.hep.utils.fatjet import (
    FatJet,
)
from hep_ml.hep.utils import (
    root_utils as ru,
)
from hep_ml.exe_utils import (
    get_from_indices,
)


def dijet_cut(
    events, logging=False
):
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
    ind = (
        FinalStates.index_map
    )
    return_dict = dict()
    passed_indices = []
    for (
        jets,
        tower,
        track,
    ) in zip(
        Jets, Towers, Tracks
    ):
        event_index += 1
        if (
            event_index
            % 5000
            == 0
            and logging
        ):
            print(
                "Event count: ",
                event_index,
                "Passed Events: ",
                passed_event_index,
            )
        array_jets = jets
        if len(jets) < 2:
            continue
        jets = ru.GetTLorentzVector(
            jets[:, :4]
        )
        if (
            jets[0].Pt()
            < 150.0
        ):
            continue
        if (
            jets[1].Pt()
            < 130.0
        ):
            continue
        if (
            abs(
                jets[0].Eta()
            )
            > 4.7
        ):
            continue
        if (
            abs(
                jets[1].Eta()
            )
            > 4.7
        ):
            continue

        if (
            passed_event_index
            == 0
        ):
            (
                return_dict[
                    "Jet"
                ],
                return_dict[
                    "Tower"
                ],
            ) = np.array(
                [jets[:2]]
            ), np.array(
                [tower]
            )
            return_dict[
                "jet_delphes"
            ] = [
                array_jets[:]
            ]
            return_dict[
                "Tower"
            ] = [tower]
            return_dict[
                "Track"
            ] = [track]
        else:
            return_dict[
                "Jet"
            ] = np.concatenate(
                (
                    return_dict[
                        "Jet"
                    ],
                    np.array(
                        [
                            jets[
                                :2
                            ]
                        ]
                    ),
                ),
                axis=0,
            )
            return_dict[
                "Tower"
            ].append(tower)
            return_dict[
                "Track"
            ].append(track)
            return_dict[
                "jet_delphes"
            ].append(
                array_jets[:]
            )
        passed_indices.append(
            event_index - 1
        )
        passed_event_index += (
            1
        )
    if (
        passed_event_index
        > 1
    ):
        return_dict[
            "Tower"
        ] = np.array(
            return_dict[
                "Tower"
            ]
        )
        return_dict[
            "jet_delphes"
        ] = np.array(
            return_dict[
                "jet_delphes"
            ]
        )
        return_dict[
            "passed_indices"
        ] = np.array(
            passed_indices
        )
        return_dict[
            "Track"
        ] = np.array(
            return_dict[
                "Track"
            ]
        )
    if (
        passed_event_index
        == 1
    ):
        return {}
    return return_dict
