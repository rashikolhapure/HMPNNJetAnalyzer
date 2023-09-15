from ..config import (
    tower_index,
)
from ...plotter import (
    Plotter,
)
from ...plot_utils import (
    seperate_image_plot,
    plot_tower_jets,
)
from . import (
    root_utils as ru,
)
from .fatjet import FatJet
from ROOT import (
    TVector3,
    TLorentzVector,
)
import matplotlib.pyplot as plt
import os
import sys
import math

import numpy as np

np.set_printoptions(precision=16)


##################################### IMAGE PREPROCESSING#################
def translate(*args, **kwargs):
    """array of elements numpy array or float with coordinate in (X,Y),
    return X-x and Y-y"""
    for item in args:
        # print (item.shape)
        item[0] = item[0] - kwargs["x"]
        item[1] = item[1] - kwargs["y"]
    if len(args) == 1:
        return args[0]
    return args


def rotate(*args, **kwargs):
    """array of elements TVector3 and theta, imtem.RotateZ(-theta)"""
    x, y = (
        kwargs["x"],
        kwargs["y"],
    )
    theta = np.arccos(x / np.sqrt(x**2 + y**2))
    if y < 0:
        theta = -theta
    for item in args:
        # print (item.shape)
        (
            item[0],
            item[1],
        ) = item[0] * np.cos(
            theta
        ) + item[1] * np.sin(
            theta
        ), -item[0] * np.sin(theta) + item[1] * np.cos(theta)
    if len(args) == 1:
        return args[0]
    return args


def reflect(*args):
    """Reflect along x axis"""
    for item in args:
        # print(item.shape)
        item[1] = -item[1]
    if len(args) == 1:
        return args[0]
    return args


##########################################################################


################################################# FAT JET ################
def process_fatjets(fatjets, operation="all", subparts="subjets", **kwargs):
    """Regularize tower/fatjet in (eta,phi) plane wih translation to
    subpart[0],rotate such the subpart[1] is at eta=0, and reflect such
    that subpart[2] is at the positive phi"""
    # print_events(events)
    x_interval = kwargs.get(
        "x_interval",
        (-1.6, 1.6),
    )
    y_interval = kwargs.get(
        "y_interval",
        (-1.6, 1.6),
    )
    shape = kwargs.get("shape", (32, 32))
    return_shape = tuple([len(fatjets)] + list(shape))
    return_array = np.zeros(
        return_shape,
        dtype="float64",
    )
    for (
        fatjet_index,
        fatjet,
    ) in enumerate(fatjets):
        (
            fatjet,
            subjets,
        ) = regularize_fatjet(fatjet)
        if subparts != "subjets":
            subjets = subparts
        (
            fatjet,
            subjets,
        ) = translate(
            fatjet,
            subjets,
            x=subjets[0][0],
            y=subjets[1][0],
        )
        try:
            (
                fatjet,
                subjets,
            ) = rotate(
                fatjet,
                subjets,
                x=subjets[0][1],
                y=subjets[1][1],
            )
        except IndexError:
            pass
        try:
            if subjets[1][2] < 0:
                (
                    fatjet,
                    subjets,
                ) = reflect(
                    fatjet,
                    subjets,
                )
        except IndexError:
            pass
        return_array[fatjet_index] = binner(
            fatjet,
            shape=(32, 32),
            x_interval=x_interval,
            y_interval=y_interval,
        )
    return return_array


def shift_phi(
    phi,
    shift,
    range=(-np.pi, np.pi),
    side="right",
    tolerance=10e-8,
):
    assert phi >= (range[0] - tolerance) and phi <= (range[1] + tolerance), (
        str(phi)
        + " not in prescribed range "
        + str(range[0])
        + " to "
        + str(range[1])
    )
    if abs(shift) > 2 * np.pi:
        sign = np.sign(shift)
        _, shift = divmod(
            abs(shift),
            2 * np.pi,
        )
        shift = sign * shift
        print(shift)
    if side == "right":
        shifted = phi + shift
    else:
        shifted = phi - shift
    # print (shifted)
    if shifted > range[1]:
        shifted = -np.pi + (shifted - range[1])
    if shifted < range[0]:
        shifted = np.pi + (range[0] - shifted)
    return shifted


def regularize_fatjet(
    fatjet,
    r=1.2,
    inclusive=True,
    subjet_fourvec=False,
    force=True,
    check=False,
    **kwargs
):
    """<fatjet> has constituents as TLorentzVector return array f TVector3
    with (eta,phi,pt) axes, regulates phi such that all components lie inside
    fatjet radius R in the Euclidean (eta,phi) plane, reclusters the fatjet
    with CA algorithm with r=0.4 and returns them in the same (eta,phi,pt)
    format
    """
    phi, eta = (
        np.sum(fatjet).Phi(),
        np.sum(fatjet).Eta(),
    )
    recl_algo = kwargs.get(
        "recluster_algorithm",
        "CA",
    )
    n_subjets = kwargs.get("subjets", 3)
    if inclusive:
        recl_r = kwargs.get(
            "recluster_r",
            0.4,
        )
        subjets = np.array(
            [
                TLorentzVector(
                    item.px,
                    item.py,
                    item.pz,
                    item.e,
                )
                for item in FatJet().Recluster(
                    fatjet,
                    r=recl_r,
                    algorithm=recl_algo,
                    subjets=n_subjets,
                    pt_min=5,
                )
            ]
        )
    else:
        recl_r = kwargs.get("recluster_r", r)
        (
            sequence,
            _,
        ) = FatJet().Recluster(
            fatjet,
            r=recl_r,
            algorithm=recl_algo,
            subjets=n_subjets,
            return_val="sequence",
        )
        subjets = np.swapaxes(
            np.array(
                [
                    [
                        item.eta,
                        item.phi,
                        item.pt,
                    ]
                    for item in sequence.exclusive_jets(n_subjets)
                ]
            ),
            0,
            1,
        )
    num_fat = ru.GetNumpy(
        fatjet,
        format="image",
    )
    # print (num_fat.shape,fatjet)
    if check:
        p = Plotter(
            projection="subplots",
            set_range=False,
        )
        (
            p.fig,
            axes,
        ) = plt.subplots(
            ncols=2,
            figsize=(20, 10),
        )
        p.axes = axes[0]
        p.scatter_plot(num_fat)
    fj_sum = np.sum(fatjet)
    delta_num = np.array(
        [
            [
                item.Eta() - fj_sum.Eta(),
                item.DeltaPhi(fj_sum),
                item.Pt(),
            ]
            for item in fatjet
        ]
    )
    if force and len(subjets) < n_subjets:
        assert (
            len(fatjet) >= n_subjets
        ), "Can't force, less number of jet ({}) constituents!".format(
            len(fatjet)
        )
        subjets = np.swapaxes(
            delta_num[:n_subjets],
            0,
            1,
        )
    else:
        subjets = np.swapaxes(
            np.array(
                [
                    [
                        item.Eta() - fj_sum.Eta(),
                        item.DeltaPhi(fj_sum),
                        item.Pt(),
                    ]
                    for item in subjets
                ]
            ),
            0,
            1,
        )
    # print
    if check:
        p.axes = axes[1]
        p.scatter_plot(
            np.swapaxes(
                delta_num,
                0,
                1,
            )
        )
        p.axes = axes
        p.save_fig(
            "reg_check",
            dpi=100,
        )
        # sys.exit()
    return (
        np.swapaxes(delta_num, 0, 1),
        subjets,
    )


def __regularize_fatjet(fatjet, r=1.2, inclusive=False, **kwargs):
    """<fatjet> has constituents as TLorentzVector return array f TVector3
    with (eta,phi,pt) axes, regulates phi such that all components lie
    inside fatjet radius R in the Euclidean (eta,phi) plane, reclusters
    the fatjet with CA algorithm with r=0.4 and returns them in the same
    (eta,phi,pt) format
    """
    phi, eta = (
        np.sum(fatjet).Phi(),
        np.sum(fatjet).Eta(),
    )
    recl_algo = kwargs.get(
        "recluster_algorithm",
        "CA",
    )
    n_subjets = kwargs.get("subjets", 2)
    if inclusive:
        recl_r = kwargs.get(
            "recluster_r",
            0.4,
        )
        subjets = np.swapaxes(
            np.array(
                [
                    [
                        item.eta,
                        item.phi,
                        item.pt,
                    ]
                    for item in FatJet().Recluster(
                        fatjet,
                        r=recl_r,
                        algorithm=recl_algo,
                        subjets=n_subjets,
                    )
                ]
            ),
            0,
            1,
        )
    else:
        recl_r = kwargs.get("recluster_r", r)
        (
            sequence,
            _,
        ) = FatJet().Recluster(
            fatjet,
            r=recl_r,
            algorithm=recl_algo,
            subjets=n_subjets,
            return_val="sequence",
        )
        subjets = np.swapaxes(
            np.array(
                [
                    [
                        item.eta,
                        item.phi,
                        item.pt,
                    ]
                    for item in sequence.exclusive_jets(n_subjets)
                ]
            ),
            0,
            1,
        )
    num_fat = ru.GetNumpy(
        fatjet,
        format="image",
    )
    delta = np.pi - abs(phi)
    if delta < r:
        # ----To check uncomment all----#
        # p=Plotter(set_range=False)
        # p.axes.scatter(num_fat[0],num_fat[1],label="before",alpha=0.5)
        for item in (
            num_fat,
            subjets,
        ):
            d = r - delta
            if phi < 0:
                indices = item[1] > 0
                item[
                    1,
                    indices,
                ] = (
                    -2 * np.pi
                    + item[
                        1,
                        indices,
                    ]
                )
            else:
                indices = item[1] < 0
                item[
                    1,
                    indices,
                ] = (
                    2 * np.pi
                    + item[
                        1,
                        indices,
                    ]
                )
        # p.axes.scatter(num_fat[0],num_fat[1],label="After",marker='s',alpha=0.5)
        # p.axes.legend()
        # p.Show()
        # sys.exit()
    return num_fat, subjets


def _regularize_fatjet(fatjet, r=1.2):
    """<fatjet> has constituents as TLorentzVector return array f TVector3
    with
    (eta,phi,pt) axes, regulates phi such that all components lie inside
    fatjet radius R in the Euclidean (eta,phi) plane, reclusters the fatjet
    with CA algorithm with r=0.4 and returns them in the same (eta,phi,pt)
    format
    """
    phi, eta = (
        np.sum(fatjet).Phi(),
        np.sum(fatjet).Eta(),
    )
    subjets = np.swapaxes(
        np.array(
            [
                [
                    item.eta,
                    item.phi,
                    item.pt,
                ]
                for item in FatJet().Recluster(
                    fatjet,
                    r=0.4,
                    algorithm="CA",
                    subjets=3,
                )
            ]
        ),
        0,
        1,
    )
    num_fat = ru.GetNumpy(
        fatjet,
        format="image",
    )
    delta = np.pi - abs(phi)
    if delta < r:
        for item in (
            num_fat,
            subjets,
        ):
            d = r - delta
            if phi < 0:
                indices = item[1] > 0
                item[
                    1,
                    indices,
                ] = (
                    -2 * np.pi
                    + item[
                        1,
                        indices,
                    ]
                )
            else:
                indices = item[1] < 0
                item[
                    1,
                    indices,
                ] = (
                    2 * np.pi
                    + item[
                        1,
                        indices,
                    ]
                )
    return num_fat, subjets


def _remove_jets(
    lorentz_tower,
    lorentz_jets,
    r=0.4,
    return_jets=False,
    shift_jets=True,
    **kwargs
):
    if kwargs.get("verbose", False):
        print("Removing jet constituents...")
        print(lorentz_tower.shape)
    shifted_jets = []
    for jet in lorentz_jets:
        del_r = np.array([item.DeltaR(jet) for item in lorentz_tower])
        shifted_jet = TLorentzVector()
        # shifted_phi=
        shifted_jet.SetPtEtaPhiM(
            jet.Pt(),
            jet.Eta(),
            shift_phi(
                jet.Phi(),
                np.pi,
            ),
            jet.M(),
        )
        collect_indices = np.array(
            [
                i
                for i, vect in enumerate(lorentz_tower)
                if vect.DeltaR(shifted_jet) <= r
            ]
        )
        valid_indices = np.where(del_r > r)
        collected_vectors = np.array(
            [TLorentzVector() for _ in collect_indices]
        )
        for i, item in zip(
            collect_indices,
            collected_vectors,
        ):
            item.SetPtEtaPhiM(
                lorentz_tower[i].Pt(),
                lorentz_tower[i].Eta(),
                lorentz_tower[i].Phi() - np.pi,
                lorentz_tower[i].M(),
            )
        lorentz_tower = lorentz_tower[valid_indices]
        if shift_jets:
            lorentz_tower = ru.Sort(
                np.concatenate(
                    (
                        lorentz_tower,
                        collected_vectors,
                    ),
                    axis=0,
                )
            )
        shifted_jets.append(shifted_jet)
        # print (valid_indices)
        # ru.Print(lorentz_tower),ru.Print(collected_vectors)
    # sys.exit()
    if not return_jets:
        return lorentz_tower
    else:
        return (
            lorentz_tower,
            shifted_jets,
        )


def keep_jets(lorentz_tower, lorentz_jets, r=0.4, **kwargs):
    if kwargs.get("verbose", False):
        print("Keeping only jet constituents...")
        print(lorentz_tower.shape)
    keep_indices = []
    for jet in lorentz_jets:
        del_r = np.array([item.DeltaR(jet) for item in lorentz_tower])
        keep_indices.append(np.where(del_r < r)[0])
    keep_indices = np.concatenate(
        keep_indices,
        axis=0,
    )
    return lorentz_tower[keep_indices]


def remove_jets(lorentz_tower, lorentz_jets, r=0.4, **kwargs):
    if kwargs.get("verbose", False):
        print("Removing jet constituents...")
        print(lorentz_tower.shape)
    return_array = []
    other_array = []
    removed_constituents = []
    for item in lorentz_tower:
        add = True
        for jet in lorentz_jets:
            if item.DeltaR(jet) <= r:
                add = False
                break
        if add:
            removed_constituents.append(item)
    if kwargs.get("central_only", False) or kwargs.get(
        "seperate_center",
        False,
    ):
        assert len(lorentz_jets) == 2
        region = []
        for jet in lorentz_jets:
            if jet.Eta() < 0:
                region.append(jet.Eta() + r)
            else:
                region.append(jet.Eta() - r)
        region.sort()
        assert lorentz_jets[0].Eta() * lorentz_jets[1].Eta() < 0
        # assert region[0]*region[1]<0
        for item in removed_constituents:
            if item.Eta() >= region[0] and item.Eta() <= region[1]:
                return_array.append(item)
            else:
                other_array.append(item)
        assert len(removed_constituents) == (
            len(return_array) + len(other_array)
        )
    else:
        return_array = removed_constituents
    return_array = np.array(return_array)
    if kwargs.get("sorted_by_pt", False):
        return_array = ru.Sort(return_array)
    if kwargs.get(
        "seperate_center",
        False,
    ):
        if kwargs.get(
            "sorted_by_pt",
            False,
        ):
            other_array = ru.Sort(np.array(other_array))
            # ru.Print(return_array),ru.Print(other_array)
        return (
            return_array,
            np.array(other_array),
        )
    else:
        return return_array


##########################################################################
# --------------------------------------------BINNING UTILS-------------------------------------------------#
##########################################################################


def image_to_var(
    images,
    eta_axis=2,
    phi_axis=1,
    eta_range=(-5, 5),
    phi_range=(
        -np.pi,
        np.pi,
    ),
):
    if images.shape[-1] == 1:
        images = np.squeeze(images)
    eta_interval = abs((eta_range[1] - eta_range[0])) / images.shape[eta_axis]
    phi_interval = abs((phi_range[1] - phi_range[0])) / images.shape[phi_axis]
    eta_centers = np.linspace(
        eta_range[0] + eta_interval / 2,
        eta_range[1] - eta_interval / 2,
        images.shape[eta_axis],
    )
    phi_centers = np.linspace(
        phi_range[0] + phi_interval / 2,
        phi_range[1] - phi_interval / 2,
        images.shape[phi_axis],
    )
    assert (
        len(eta_centers) == images.shape[eta_axis]
        and len(phi_centers) == images.shape[phi_axis]
    )
    return_array = []
    for image in images:
        indices = np.where(image)
        eta = eta_centers[indices[eta_axis - 1]]
        phi = phi_centers[indices[phi_axis - 1]]
        pt = image[indices]
        return_array.append(
            np.swapaxes(
                np.array(
                    [
                        eta,
                        phi,
                        pt,
                    ]
                ),
                0,
                1,
            )
        )
    return np.array(return_array)


def tower_padding(
    tower,
    pad_axis=0,
    pad_size=4,
):
    if pad_axis == 0:
        new_shape = (
            tower.shape[0] + 2 * pad_size,
            tower.shape[1],
        )
        return_array = np.zeros(new_shape)
        return_array[
            pad_size:-pad_size,
            :,
        ] = tower
        for i in range(pad_size):
            return_array[i] = tower[tower.shape[0] - pad_size + i]
            return_array[i - pad_size] = tower[i]
        return return_array
    else:
        new_shape = (
            tower.shape[0],
            tower.shape[1] + 2 * pad_size,
        )
        return_array = np.zeros(new_shape)
        return_array[
            :,
            pad_size:-pad_size,
        ] = tower
        for i in range(pad_size):
            return_array[:, i] = tower[
                :,
                tower.shape[1] - pad_size + i,
            ]
            return_array[
                :,
                i - pad_size,
            ] = tower[:, i]
        return return_array


def tower_bin(tower, format="tower", **kwargs):
    bin_size = kwargs.get(
        "bin_size",
        (0.17, 0.17),
    )
    if format == "image":
        tower = np.array(
            [
                tower[2],
                tower[0],
                tower[1],
            ]
        )
        tower = np.swapaxes(tower, 0, 1)
    if kwargs.get(
        "return_seperate",
        False,
    ):
        center_range = kwargs.get(
            "center_range",
            (-1.6, 1.6),
        )
        tower_left = tower[tower[:, 1] < center_range[0]]
        tower_center = tower[
            np.logical_and(
                tower[:, 1] >= center_range[0],
                tower[:, 1] <= center_range[1],
            )
        ]
        tower_right = tower[tower[:, 1] > center_range[1]]
        assert tower.shape[0] == (
            tower_left.shape[0] + tower_right.shape[0] + tower_center.shape[0]
        )
        left_bin = binner(
            np.array(
                [
                    tower_left[:, 1],
                    tower_left[:, 2],
                    tower_left[:, 0],
                ]
            ),
            x_interval=(
                -5,
                center_range[0],
            ),
            y_interval=(
                -np.pi,
                np.pi,
            ),
            bin_size=bin_size,
            swap=True,
        )
        center_bin = binner(
            np.array(
                [
                    tower_center[:, 1],
                    tower_center[:, 2],
                    tower_center[:, 0],
                ]
            ),
            x_interval=center_range,
            y_interval=(
                -np.pi,
                np.pi,
            ),
            bin_size=bin_size,
            swap=True,
        )
        right_bin = binner(
            np.array(
                [
                    tower_right[:, 1],
                    tower_right[:, 2],
                    tower_right[:, 0],
                ]
            ),
            x_interval=(
                center_range[1],
                5,
            ),
            y_interval=(
                -np.pi,
                np.pi,
            ),
            bin_size=bin_size,
            swap=True,
        )
        if "plot" in sys.argv:
            seperate_image_plot(
                left_bin,
                center_bin,
                right_bin,
                save_path="./plots",
            )
            sys.exit()
        return (
            left_bin,
            center_bin,
            right_bin,
        )
    else:
        return binner(
            np.array(
                [
                    tower[:, 1],
                    tower[:, 2],
                    tower[:, 0],
                ]
            ),
            x_interval=(
                -5,
                5,
            ),
            y_interval=(
                -np.pi,
                np.pi,
            ),
            bin_size=bin_size,
            swap=True,
        )


def binner(
    array,
    x_interval=(-1.6, 1.6),
    y_interval=(-1.6, 1.6),
    expand=False,
    swap=False,
    **kwargs
):
    if array.shape[-1] != 3 or swap:
        array = np.swapaxes(array, 0, 1)
    if "shape" in kwargs:
        shape = kwargs.get("shape")
        (
            x_bin_size,
            y_bin_size,
        ) = (x_interval[1] - x_interval[0]) / shape[
            0
        ], (y_interval[1] - y_interval[0]) / shape[1]
    else:
        assert "bin_size" in kwargs
        bin_size = kwargs.get("bin_size")
        x_bin_size = bin_size[0]
        y_bin_size = bin_size[1]
        x_shape = math.ceil((x_interval[1] - x_interval[0]) / x_bin_size)
        y_shape = math.ceil((y_interval[1] - y_interval[0]) / y_bin_size)
        shape = (
            x_shape,
            y_shape,
        )
    error, err_count = (
        False,
        0,
    )
    binned = np.zeros(
        shape,
        dtype="float64",
    )
    for item in array:
        i, j = int((item[0] - x_interval[0]) / x_bin_size), int(
            (item[1] - y_interval[0]) / y_bin_size
        )
        try:
            binned[i, j] += item[2]
        except IndexError:
            # print (item)
            err_count += 1
            error = True
            pass
    binned = np.transpose(binned)
    if err_count > len(array) / 2.0:
        print(
            "Error",
            array.shape,
            err_count,
        )  # ,array)
    # print (len(array),err_count,len(array)/2.)
    # sys.exit()
    if not expand:
        return binned
    return np.expand_dims(binned, -1)
