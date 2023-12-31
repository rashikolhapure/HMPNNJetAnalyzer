import os
import pickle
import sys


from collections import (
    namedtuple,
)


Index = namedtuple(
    "Index",
    [
        "MET",
        "ET",
        "PT",
        "Eta",
        "Phi",
        "Mass",
        "Charge",
        "BTag",
        "TauTag",
        "E",
        "Eem",
        "Ehad",
    ],
)
jet_index = Index(
    PT=0,
    MET=None,
    ET=None,
    Eta=1,
    Phi=2,
    Mass=3,
    Charge=6,
    BTag=4,
    TauTag=5,
    E=None,
    Eem=None,
    Ehad=None,
)
tower_index = Index(
    ET=0,
    MET=None,
    PT=None,
    Eta=1,
    Phi=2,
    Mass=None,
    Charge=3,
    BTag=None,
    TauTag=None,
    E=3,
    Eem=4,
    Ehad=5,
)
met_index = Index(
    MET=0,
    PT=None,
    ET=None,
    Eta=None,
    Phi=2,
    Mass=None,
    Charge=3,
    BTag=None,
    TauTag=None,
    E=None,
    Eem=None,
    Ehad=None,
)
lepton_index = Index(
    PT=0,
    MET=None,
    ET=None,
    Eta=1,
    Phi=2,
    Mass=None,
    Charge=3,
    BTag=None,
    TauTag=None,
    E=None,
    Eem=None,
    Ehad=None,
)
track_index = Index(
    PT=0,
    MET=None,
    ET=None,
    Eta=1,
    Phi=2,
    Mass=None,
    E=None,
    Charge=None,
    BTag=None,
    TauTag=None,
    Eem=None,
    Ehad=None,
)
photon_index = Index(
    PT=0,
    MET=None,
    ET=None,
    Eta=1,
    Phi=2,
    Mass=None,
    Charge=None,
    BTag=None,
    TauTag=None,
    E=3,
    Eem=None,
    Ehad=None,
)
GenParticleIndex = namedtuple(
    "GenParticleIndex",
    [
        "PT",
        "Eta",
        "Phi",
        "E",
        "PID",
        "M1",
        "M2",
        "D1",
        "D2",
        "Status",
        "Mass",
    ],
)
gen_particle_index = GenParticleIndex(
    PT=0,
    Eta=1,
    Phi=2,
    E=3,
    PID=4,
    M1=5,
    M2=6,
    D1=7,
    D2=8,
    Status=9,
    Mass=10,
)


EventAttribute = namedtuple(
    "EventAttribute",
    [
        "run_name",
        "tag",
        "path",
        "index",
    ],
)


PID_to_particle = {
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
    23: "z",
    24: "w+",
    -24: "w-",
    25: "h",
}
particle_to_PID = {val: key for key, val in PID_to_particle.items()}
charged_leptons = {
    "e+",
    "e-",
    "mu+",
    "mu-",
    "ta+",
    "ta-",
}
neutrinos = {
    "ve",
    "vm",
    "vt",
    "ve~",
    "vm~",
    "vt~",
}


class FinalStates:
    """
    A class that defines attributes and mappings for various
    final states in particle physics data.
    """
    _attributes = [
        "PT",
        "Eta",
        "Phi",
    ]
    index_map = {
        "PT": 0,
        "ET": 0,
        "MET": 0,
        "Eta": 1,
        "Phi": 2,
        "Charge": 3,
        "JetMass": 3,
        "BTag": 4,
        "E": 3,
        "Eem": 4,
        "Ehad": 5,
        "TauTag": 6,
    }
    attributes = {
        "Jet": _attributes
        + [
            "Mass",
            "BTag",
            "TauTag",
            "Charge",
        ],
        "Muon": _attributes + ["Charge"],
        "Electron": _attributes + ["Charge"],
        "MissingET": [
            "MET",
            "Eta",
            "Phi",
        ],
        "Tower": [
            "ET",
            "Eta",
            "Phi",
            "E",
            "Eem",
            "Ehad",
        ],
        "EFlow": [
            "ET",
            "Eta",
            "Phi",
            "E",
            "Eem",
            "Ehad",
        ],
        "Photon": [
            "PT",
            "Eta",
            "Phi",
            "E",
        ],
        "Track": [
            "PT",
            "Eta",
            "Phi",
        ],
        "Particle": [
            "PT",
            "Eta",
            "Phi",
            "E",
            "PID",
            "M1",
            "M2",
            "D1",
            "D2",
            "Status",
            "Mass",
        ],
        "Event": ["Weight"],
    }


class Paths:
    """
    A class for defining file paths based on conditions.
    """
    # if "ab" in sys.argv:
    # madgraph_dir=os.path.abspath("/home/sweety/MG5_aMC_v2_5_5/")
    # else: madgraph_dir=os.path.abspath("/home/rajveer/MG5_aMC_v2_6_5/")
    if "ab" in sys.argv:
        madgraph_dir = os.environ["mdgraph_dir_AB"]
        other_dir = os.environ["mdgraph_dir"]
    else:
        madgraph_dir = os.environ["mdgraph_dir"]
        other_dir = os.environ["mdgraph_dir_AB"]


class Bins:
    """
    A class for defining bin edges or ranges.
    """
    cms_tower_full = [
        -5.00, -4.83, -4.66, -4.49, -4.32, -4.15, -3.98, -3.81, -3.64, -3.47,
        -3.30, -3.13, -2.96, -2.79, -2.62, -2.45, -2.28, -2.11, -1.94, -1.77,
        -1.60, -1.52, -1.44, -1.36, -1.28, -1.20, -1.12, -1.04, -0.96, -0.88,
        -0.80, -0.72, -0.64, -0.56, -0.48, -0.40, -0.32, -0.24, -0.16, -0.08,
        0.00, 0.08, 0.16, 0.24, 0.32, 0.40, 0.48, 0.56, 0.64, 0.72, 0.80, 0.88,
        0.96, 1.04, 1.12, 1.20, 1.28, 1.36, 1.44, 1.52, 1.60, 1.77, 1.94, 2.11,
        2.28, 2.45, 2.62, 2.79, 2.96, 3.13, 3.30, 3.47, 3.64, 3.81, 3.98, 4.15,
        4.32, 4.49, 4.66, 4.83, 5.00,
    ]
    cms_central_bins_half_range = [
        0.00, 0.08, 0.16, 0.24, 0.32, 0.40, 0.48, 0.56, 0.64, 0.72,
        0.80, 0.88, 0.96, 1.04, 1.12, 1.20, 1.28, 1.36, 1.44, 1.52,
    ]
    
    cms_forward_bins_half_range = [
        1.60, 1.77, 1.94, 2.11, 2.28, 2.45, 2.62, 2.79, 2.96, 3.13,
        3.30, 3.47, 3.64, 3.81, 3.98, 4.15, 4.32, 4.49, 4.66, 4.83, 5.0,
    ]
    