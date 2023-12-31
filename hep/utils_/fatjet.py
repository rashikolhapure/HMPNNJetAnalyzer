#!/home/vishal/anaconda3/envs/scikit_hep/bin/python
from typing import List, Union
from pyjet import (
    cluster,
)
import numpy as np
from . import (
    root_utils as ru,
)
from ROOT import (
    TLorentzVector,
)


class FatJet(object):
    """
    This code defines a class called FatJet that uses the pyjet package to
    cluster jets from a set of input particles (towers), and to perform
    subjet clustering. The resulting jets are stored as a list of
    PseudoJet objects.

    The FatJet class has the following methods:

    __init__: Initializes the class with the input parameters, including the
    clustering algorithm (antikt, CA, or kt), the distance parameter R, the
    minimum transverse momentum pt_min for jet constituents, and a verbose
    flag to print output.

    ConstructVector: A helper function that converts a list of particles
    (each with pt, eta, phi, and mass) to an array of four-vectors.
    ClusterJets: Uses PyJet's ClusterSequence object to find clusters in the
    vectorized.

    RemoveElectron: A function that removes fatjets formed with energy
    deposits from electrons.

    Get: Clusters jets from the input Tower particles using the specified
    algorithm and R parameter. Returns a list of PseudoJet objects
    representing the fatjets.

    GetConstituents: Given a list of fatjets, returns an array of particles
    (in TLorentzVector format) that make up each jet.

    Recluster: Performs subjet clustering on a given fatjet using the
    specified algorithm, R parameter, and dcut parameter. Returns a list
    of PseudoJet objects representing the subjets. Overall, this code
    provides a set of tools for jet clustering and subjet analysis,
    which are useful in particle physics analyses.
    """

    def __init__(
        self,
        tower: np.ndarray = None,
        algorithm: str = "antikt",
        r: float = 1.2,
        pt_min: float = 200.0,
        verbose: bool = False,
    ):
        """
        <tower> should be numpy array with shape (constituents,3/4) with
        the second dimension being [pt,eta,phi,(mass)]
        """
        self.Tower = tower
        self.Verbose = verbose
        self.Algorithm = algorithm
        self.R = r
        self.PtMin = pt_min
        self.AlgorithmDict = {
            "antikt": -1,
            "CA": 0,
            "kt": 1,
        }
        self.Vectors = None
        self.fatjets = None
        self.cluster_sequence = None

    def ConstructVector(
        self, fatjet: Union[List[TLorentzVector], np.ndarray]
    ) -> np.ndarray:
        if isinstance(fatjet[0], TLorentzVector):
            fatjet = ru.GetNumpy(
                fatjet,
                format="lhc",
                observable_first=False,
            )
        vectors = []
        for item in fatjet:
            vectors.append(
                np.array(
                    (
                        item[0],
                        item[1],
                        item[2],
                        item[3],
                    ),
                    dtype=[
                        (
                            "pT",
                            "f8",
                        ),
                        (
                            "eta",
                            "f8",
                        ),
                        (
                            "phi",
                            "f8",
                        ),
                        (
                            "mass",
                            "f8",
                        ),
                    ],
                )
            )
        return np.array(vectors)

    def RemoveElectron(self, lepton: np.ndarray):
        """
        Remove fatjets formed with energy deposit of electrons in the event.

        Parameters:
        ----------
        lepton : numpy.ndarray
            A numpy array containing information about electrons, where each row
            represents an electron and should have the format [pt, eta, phi].

        Returns:
        -------
        list
            A list of fatjets after removing those formed with electron energy deposits.
        """
        if self.Verbose:
            print(
                lepton,
                self.fatjets,
            )
        electron_eta_phi = []
        for item in lepton:
            electron_eta_phi.append(
                [
                    item[1],
                    item[2],
                ]
            )
        indices = []
        for i in range(len(self.fatjets)):
            for item in electron_eta_phi:
                R = np.sqrt(
                    (self.fatjets[i].eta - item[0]) ** 2
                    + (self.fatjets[i].phi - item[1]) ** 2
                )
                if R < self.R:
                    if i not in indices:
                        indices.append(i)
        FatJets = []
        for i in range(len(self.fatjets)):
            if i not in indices:
                FatJets.append(self.fatjets[i])
        self.fatjets = FatJets
        if self.Verbose:
            print(self.fatjets)
        return self.fatjets

    def Get(self) -> list:
        """
        Get a list of fatjets in PseudoJet class.

        Returns:
        -------
        list
            A list of fatjets represented as PseudoJet objects.
        """
        if isinstance(self.Tower[0], np.ndarray):
            temp = np.concatenate(
                (
                    self.Tower,
                    np.zeros(
                        (
                            len(self.Tower),
                            1,
                        )
                    ),
                ),
                axis=1,
            )
        else:
            temp = ru.GetNumpy(
                self.Tower,
                format="lhc",
                observable_first=False,
            )
        vectors = []
        for item in temp:
            vectors.append(
                np.array(
                    (
                        item[0],
                        item[1],
                        item[2],
                        item[3],
                    ),
                    dtype=[
                        (
                            "pT",
                            "f8",
                        ),
                        (
                            "eta",
                            "f8",
                        ),
                        (
                            "phi",
                            "f8",
                        ),
                        (
                            "mass",
                            "f8",
                        ),
                    ],
                )
            )
        vectors = np.array(vectors)
        self.Vectors = vectors
        sequence = cluster(
            vectors,
            p=self.AlgorithmDict[self.Algorithm],
            R=self.R,
        )
        self.cluster_sequence = sequence
        self.fatjets = sequence.inclusive_jets(ptmin=self.PtMin)
        return self.fatjets

    def GetConstituents(
        self,
        fatjets: list,
        format: str = "root"
    ) -> np.ndarray:
        """
        Get a numpy array of len(fatjets) containing TLorentzVector of
        the constituents of each fatjet.

        Parameters:
        ----------
        fatjets : list
            A list of fatjets represented as PseudoJet objects.
        format : str, optional
            The format for returning the constituents data
            ("root" or "image", default is "root").

        Returns:
        -------
        np.ndarray
            A numpy array containing the constituents of each fatjet.

        Notes:
        -----
        This function returns the constituents of each fatjet as a numpy
        array of TLorentzVector or as a numpy array of eta, phi, and pt
        values based on the chosen format.
        """
        return_array = []
        if format == "image":
            for item in fatjets:
                return_array.append(
                    np.swapaxes(
                        np.array(
                            [
                                [
                                    particle.eta,
                                    particle.phi,
                                    particle.pt,
                                ]
                                for particle in item.constituents()
                            ],
                            dtype="float64",
                        ),
                        0,
                        1,
                    )
                )
            return return_array
        for item in fatjets:
            if format == "root":
                return_array.append(
                    ru.GetTLorentzVector(
                        item,
                        format="fatjet",
                    )
                )
            else:
                return_array.append(
                    np.array(
                        [
                            item.eta,
                            item.phi,
                            item.pt,
                        ],
                        dtype="float64",
                    )
                )
        return np.array(return_array)

    def Recluster(
        self,
        fatjet: object,
        r: float = 0.4,
        dcut: float = 0.5,
        algorithm: str = "CA",
        subjets: int = 3,
    ) -> list:
        """
        Recluster a fatjet into subjets using a specified jet clustering
        algorithm.

        Parameters:
        ----------
        fatjet : object
            The input fatjet to be reclustered.
        r : float, optional
            The jet radius parameter for the clustering algorithm
            (default is 0.4).
        dcut : float, optional
            The distance parameter for the clustering algorithm
            (default is 0.5).
        algorithm : str, optional
            The clustering algorithm to be used (default is "CA").
        subjets : int, optional
            The maximum number of subjets to be returned
            (default is 3).

        Returns:
        -------
        list
            A list of subjets obtained after reclustering the fatjet.

        Notes:
        -----
        This method uses a clustering algorithm to break down a fatjet into
        smaller subjets. The `pt_min` is determined based on the fatjet's
        transverse momentum.
        """
        pt_min = 0.03 * np.sum(fatjet).Pt()
        vectors = self.ConstructVector(fatjet)
        sequence = cluster(
            vectors,
            R=r,
            p=self.AlgorithmDict[algorithm],
        )
        pt_min = 0.03 * np.sum(fatjet).Pt()
        vectors = self.ConstructVector(fatjet)
        # print (vectors)
        sequence = cluster(
            vectors,
            R=r,
            p=self.AlgorithmDict[algorithm],
        )
        return sequence.inclusive_jets(ptmin=pt_min)[:subjets]


def Print(
    fatjet: object,
    name: str = None,
    format: str = "lhc",
    constituents: bool = False,
) -> None:
    """
    Print information about a fatjet or its constituents.

    Parameters:
    ----------
    fatjet : object
        The fatjet object to be printed.
    name : str, optional
        A name or label for the fatjet (default is None).
    format : str, optional
        The format to use for printing ("lhc" or "other")
        (default is "lhc").
    constituents : bool, optional
        Whether to print information about the constituents of the fatjet
        (default is False).

    Returns:
    -------
    None

    Notes:
    -----
    This function prints information about the given fatjet or its
    constituents in the specified format. If 'name' is provided, it will
    be printed as a label for the fatjet. If 'constituents' is set to True,
    the function will print information about each constituent of the fatjet.
    """
    if name is not None:
        print(name)
    if not constituents:
        if format == "lhc":
            print(
                f"    Eta: {fatjet.eta:20.16f}        "
                f"Phi: {fatjet.phi:20.16f}        "
                f"Pt : {fatjet.pt:20.16f}        "
                f"Mass: {fatjet.mass:20.16f}"
            )
        else:
            print(
                f"    Px: {fatjet.px:20.16f}        "
                f"Py: {fatjet.py:20.16f}        "
                f"Pz : {fatjet.pz:20.16f}        "
                f"E: {fatjet.e:20.16f}"
            )
    else:
        print("Constituents of array: ")
        for item in fatjet.constituents():
            if format == "lhc":
                print(
                    f"    Eta: {item.eta:20.16f}        "
                    f"Phi: {item.phi:20.16f}        "
                    f"Pt : {item.pt:20.16f}        "
                    f"Mass: {item.mass:20.16f}"
                )
            else:
                print(
                    f"    Px : {item.px:20.16f}        "
                    f"Py: {item.py:20.16f}        "
                    f"Pz: {item.pz:20.16f}        "
                    f"E : {item.E:20.16f}"
                )
    return
