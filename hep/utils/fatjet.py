#!/home/vishal/anaconda3/envs/scikit_hep/bin/python
from pyjet import cluster, PseudoJet
import numpy as np
from hep_ml.hep.utils import root_utils as ru
import sys
from ROOT import TLorentzVector


class FatJet(object):
    def __init__(
        self,
        tower=None,
        algorithm="antikt",
        r=1.2,
        pt_min=200.0,
        verbose=False,
    ):
        """<tower> should be numpy array with shape (constituents,3/4) with the second dimension being [pt,eta,phi,(mass)]"""
        self.Tower = tower
        self.Verbose = verbose
        self.Algorithm = algorithm
        self.R = r
        self.PtMin = pt_min
        self.AlgorithmDict = {"antikt": -1, "CA": 0, "kt": 1}
        self.Vectors = None
        self.fatjets = None
        self.cluster_sequence = None

    def ConstructVector(self, fatjet, maximum_particles=None):
        if type(fatjet) == PseudoJet:
            if self.Verbose:
                [print(item) for item in fatjet]
                print("\n")
            fatjet = np.array([[item.pt, item.eta, item.phi, item.mass] for item in fatjet])
        elif type(fatjet[0]) == TLorentzVector:
            fatjet = ru.GetNumpy(fatjet, format="lhc", observable_first=False)
        vectors = []
        for item in fatjet[:maximum_particles]:
            vectors.append(
                np.array(
                    (item[0], item[1], item[2], item[3]),
                    dtype=[
                        ("pT", "f8"),
                        ("eta", "f8"),
                        ("phi", "f8"),
                        ("mass", "f8"),
                    ],
                )
            )
        return np.array(vectors)

    def RemoveElectron(self, lepton):
        """If event contains electrons, remove fatjets formed with energy deposit of electrons"""
        if self.Verbose:
            print(lepton, self.fatjets)
        electron_eta_phi = []
        for item in lepton:
            electron_eta_phi.append([item[1], item[2]])
        indices = []
        for i in range(len(self.fatjets)):
            for item in electron_eta_phi:
                R = np.sqrt(
                    (self.fatjets[i].eta - item[0]) ** 2 + (self.fatjets[i].phi - item[1]) ** 2
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

    def Get(self):
        """get list of fatjet in PseudoJet class"""
        if type(self.Tower[0]) == np.ndarray:
            temp = np.concatenate((self.Tower, np.zeros((len(self.Tower), 1))), axis=1)
        else:
            temp = ru.GetNumpy(self.Tower, format="lhc", observable_first=False)
        vectors = []
        for item in temp:
            vectors.append(
                np.array(
                    (item[0], item[1], item[2], item[3]),
                    dtype=[
                        ("pT", "f8"),
                        ("eta", "f8"),
                        ("phi", "f8"),
                        ("mass", "f8"),
                    ],
                )
            )
        vectors = np.array(vectors)
        self.Vectors = vectors
        sequence = cluster(vectors, p=self.AlgorithmDict[self.Algorithm], R=self.R)
        self.cluster_sequence = sequence
        self.fatjets = sequence.inclusive_jets(ptmin=self.PtMin)
        return self.fatjets

    def GetConstituents(self, fatjets, format="root"):
        """get a numpy array of len(fatjets) containing TLorentzVector of the constituents of each fatjet"""
        return_array = []
        if format == "image":
            for item in fatjets:
                return_array.append(
                    np.swapaxes(
                        np.array(
                            [
                                [particle.eta, particle.phi, particle.pt]
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
                return_array.append(ru.GetTLorentzVector(item, format="fatjet"))
            else:
                return_array.append(np.array([item.eta, item.phi, item.pt], dtype="float64"))
        return np.array(return_array)

    # @methods
    def Recluster(
        self,
        fatjet,
        r=0.4,
        dcut=0.5,
        algorithm="CA",
        pt_min=None,
        subjets=None,
        return_val="inc_jets",
        maximum_particles=None,
    ):
        vectors = self.ConstructVector(fatjet, maximum_particles=maximum_particles)
        length = len(vectors)
        sequence = cluster(vectors, R=r, p=self.AlgorithmDict[algorithm])
        if return_val == "sequence":
            return sequence, length
        else:
            if pt_min is None:
                pt_min = 0.03 * np.sum(fatjet).Pt()
            return np.array(sequence.inclusive_jets(ptmin=pt_min)[:subjets])


def Print(fatjet, name=None, format="lhc", constituents=False):
    if name != None:
        print(name)
    if not constituents:
        if format == "lhc":
            print(
                f"    Eta: {fatjet.eta:20.16f}        Phi: {fatjet.phi:20.16f}        Pt : {fatjet.pt:20.16f}        Mass: {fatjet.mass:20.16f}"
            )
        else:
            print(
                f"    Px: {fatjet.px:20.16f}        Py: {fatjet.py:20.16f}        Pz : {fatjet.pz:20.16f}        E: {fatjet.e:20.16f}"
            )
    else:
        print("Constituents of array: ")
        for item in fatjet.constituents():
            if format == "lhc":
                print(
                    f"    Eta: {item.eta:20.16f}        Phi: {item.phi:20.16f}        Pt : {item.pt:20.16f}        Mass: {item.mass:20.16f}"
                )
            else:
                print(
                    f"    Px : {item.px:20.16f}        Py: {item.py:20.16f}        Pz: {item.pz:20.16f}        E : {item.E:20.16f}"
                )
    return
