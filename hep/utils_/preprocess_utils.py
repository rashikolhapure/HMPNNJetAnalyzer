#!/home/vishal/anaconda3/envs/scikit_hep/bin/python
# Author: Ng Vishal
# Date: Aug 31,2019
import os
import sys
import math

import numpy as np

np.set_printoptions(precision=16)
from ROOT import (
    TVector3,
    TLorentzVector,
)

from .fatjet import FatJet
from . import root_utils as ru
from ...plot_utils import (
    seperate_image_plot,
    plot_tower_jets,
)
from ..config import tower_index


##################################### IMAGE PREPROCESSING#########################################
def translate(*args, **kwargs):
    """array of elements numpy array or float with coordinate in (X,Y), return X-x and Y-y"""
    for item in args:
        # print (item.shape)
        item[0] = item[0] - kwargs["x"]
        item[1] = item[1] - kwargs["y"]
    if len(args) == 1:
        return args[0]
    return args


def rotate(*args, **kwargs):
    """array of elements TVector3 and theta, imtem.RotateZ(-theta)"""
    x, y = kwargs["x"], kwargs["y"]
    theta = np.arccos(
        x / np.sqrt(x**2 + y**2)
    )
    if y < 0:
        theta = -theta
    for item in args:
        # print (item.shape)
        item[0], item[1] = item[
            0
        ] * np.cos(theta) + item[
            1
        ] * np.sin(
            theta
        ), -item[
            0
        ] * np.sin(
            theta
        ) + item[
            1
        ] * np.cos(
            theta
        )
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


##################################################################################################


#################################################FAT JET #########################################
def process_fatjets(
    fatjets,
    operation="all",
    subparts="subjets",
    **kwargs
):
    """Regularize tower/fatjet in (eta,phi) plane wih translation to subpart[0], rotate such the subpart[1] is at eta=0, and reflect such that subpart[2]
    is at the positive phi"""
    # print_events(events)
    x_interval = kwargs.get(
        "x_interval", (-1.6, 1.6)
    )
    y_interval = kwargs.get(
        "y_interval", (-1.6, 1.6)
    )
    shape = kwargs.get(
        "shape", (32, 32)
    )
    return_shape = tuple(
        [len(fatjets)] + list(shape)
    )
    return_array = np.zeros(
        return_shape, dtype="float64"
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
        fatjet, subjets = translate(
            fatjet,
            subjets,
            x=subjets[0][0],
            y=subjets[1][0],
        )
        try:
            fatjet, subjets = rotate(
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
                    fatjet, subjets
                )
        except IndexError:
            pass
        return_array[
            fatjet_index
        ] = binner(
            fatjet,
            shape=(32, 32),
            x_interval=x_interval,
            y_interval=y_interval,
        )
    return return_array


def regularize_fatjet(fatjet, r=1.2):
    """<fatjet> has constituents as TLorentzVector return array f TVector3 with (eta,phi,pt) axes,
    regulates phi such that all components lie inside fatjet radius R in the Euclidean (eta,phi) plane,
    reclusters the fatjet with CA algorithm with r=0.4 and returns them in the same (eta,phi,pt) format
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
        fatjet, format="image"
    )
    delta = np.pi - abs(phi)
    if delta < r:
        for item in (num_fat, subjets):
            d = r - delta
            if phi < 0:
                indices = item[1] > 0
                item[1, indices] = (
                    -2 * np.pi
                    + item[1, indices]
                )
            else:
                indices = item[1] < 0
                item[1, indices] = (
                    2 * np.pi
                    + item[1, indices]
                )
    return num_fat, subjets


def remove_jets(
    lorentz_tower,
    lorentz_jets,
    r=0.5,
    **kwargs
):
    if kwargs.get("verbose", False):
        print(
            "Removing jet constituents..."
        )
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
            removed_constituents.append(
                item
            )
    if kwargs.get(
        "central_only", False
    ) or kwargs.get(
        "seperate_center", False
    ):
        assert len(lorentz_jets) == 2
        region = []
        for jet in lorentz_jets:
            if jet.Eta() < 0:
                region.append(
                    jet.Eta() + r
                )
            else:
                region.append(
                    jet.Eta() - r
                )
        region.sort()
        assert (
            lorentz_jets[0].Eta()
            * lorentz_jets[1].Eta()
            < 0
        )
        # assert region[0]*region[1]<0
        for (
            item
        ) in removed_constituents:
            if (
                item.Eta() >= region[0]
                and item.Eta()
                <= region[1]
            ):
                return_array.append(
                    item
                )
            else:
                other_array.append(
                    item
                )
        assert len(
            removed_constituents
        ) == (
            len(return_array)
            + len(other_array)
        )
    else:
        return_array = (
            removed_constituents
        )
    return_array = np.array(
        return_array
    )
    if kwargs.get(
        "sorted_by_pt", False
    ):
        return_array = ru.Sort(
            return_array
        )
    if kwargs.get(
        "seperate_center", False
    ):
        if kwargs.get(
            "sorted_by_pt", False
        ):
            other_array = ru.Sort(
                np.array(other_array)
            )
            # ru.Print(return_array),ru.Print(other_array)
        return return_array, np.array(
            other_array
        )
    else:
        return return_array


############################################################################################################
# --------------------------------------------BINNING UTILS-------------------------------------------------#
############################################################################################################


def image_to_var(
    images,
    eta_axis=2,
    phi_axis=1,
    eta_range=(-5, 5),
    phi_range=(-np.pi, np.pi),
):
    if images.shape[-1] == 1:
        images = np.squeeze(images)
    eta_interval = (
        abs(
            (
                eta_range[1]
                - eta_range[0]
            )
        )
        / images.shape[eta_axis]
    )
    phi_interval = (
        abs(
            (
                phi_range[1]
                - phi_range[0]
            )
        )
        / images.shape[phi_axis]
    )
    eta_centers = np.linspace(
        eta_range[0]
        + eta_interval / 2,
        eta_range[1]
        - eta_interval / 2,
        images.shape[eta_axis],
    )
    phi_centers = np.linspace(
        phi_range[0]
        + phi_interval / 2,
        phi_range[1]
        - phi_interval / 2,
        images.shape[phi_axis],
    )
    assert (
        len(eta_centers)
        == images.shape[eta_axis]
        and len(phi_centers)
        == images.shape[phi_axis]
    )
    return_array = []
    for image in images:
        indices = np.where(image)
        eta = eta_centers[
            indices[eta_axis - 1]
        ]
        phi = phi_centers[
            indices[phi_axis - 1]
        ]
        pt = image[indices]
        return_array.append(
            np.swapaxes(
                np.array(
                    [eta, phi, pt]
                ),
                0,
                1,
            )
        )
    return np.array(return_array)


def tower_padding(
    tower, pad_axis=0, pad_size=4
):
    if pad_axis == 0:
        new_shape = (
            tower.shape[0]
            + 2 * pad_size,
            tower.shape[1],
        )
        return_array = np.zeros(
            new_shape
        )
        return_array[
            pad_size:-pad_size, :
        ] = tower
        for i in range(pad_size):
            return_array[i] = tower[
                tower.shape[0]
                - pad_size
                + i
            ]
            return_array[
                i - pad_size
            ] = tower[i]
        return return_array
    else:
        new_shape = (
            tower.shape[0],
            tower.shape[1]
            + 2 * pad_size,
        )
        return_array = np.zeros(
            new_shape
        )
        return_array[
            :, pad_size:-pad_size
        ] = tower
        for i in range(pad_size):
            return_array[:, i] = tower[
                :,
                tower.shape[1]
                - pad_size
                + i,
            ]
            return_array[
                :, i - pad_size
            ] = tower[:, i]
        return return_array


def tower_bin(
    tower, format="tower", **kwargs
):
    bin_size = kwargs.get(
        "bin_size", (0.17, 0.17)
    )
    if format == "image":
        tower = np.array(
            [
                tower[2],
                tower[0],
                tower[1],
            ]
        )
        tower = np.swapaxes(
            tower, 0, 1
        )
    if kwargs.get(
        "return_seperate", False
    ):
        tower_left = tower[
            tower[:, 1] < -1.6
        ]
        tower_center = tower[
            np.logical_and(
                tower[:, 1] >= -1.6,
                tower[:, 1] <= 1.6,
            )
        ]
        tower_right = tower[
            tower[:, 1] > 1.6
        ]
        assert tower.shape[0] == (
            tower_left.shape[0]
            + tower_right.shape[0]
            + tower_center.shape[0]
        )
        left_bin = binner(
            np.array(
                [
                    tower_left[:, 1],
                    tower_left[:, 2],
                    tower_left[:, 0],
                ]
            ),
            x_interval=(-5, -1.6),
            y_interval=(-np.pi, np.pi),
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
            x_interval=(-1.6, 1.6),
            y_interval=(-np.pi, np.pi),
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
            x_interval=(1.6, 5),
            y_interval=(-np.pi, np.pi),
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
            x_interval=(-5, 5),
            y_interval=(-np.pi, np.pi),
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
        array = np.swapaxes(
            array, 0, 1
        )
    if "shape" in kwargs:
        shape = kwargs.get("shape")
        x_bin_size, y_bin_size = (
            x_interval[1]
            - x_interval[0]
        ) / shape[0], (
            y_interval[1]
            - y_interval[0]
        ) / shape[
            1
        ]
    else:
        assert "bin_size" in kwargs
        bin_size = kwargs.get(
            "bin_size"
        )
        x_bin_size = bin_size[0]
        y_bin_size = bin_size[1]
        x_shape = math.ceil(
            (
                x_interval[1]
                - x_interval[0]
            )
            / x_bin_size
        )
        y_shape = math.ceil(
            (
                y_interval[1]
                - y_interval[0]
            )
            / y_bin_size
        )
        shape = (x_shape, y_shape)
    error, err_count = False, 0
    binned = np.zeros(
        shape, dtype="float64"
    )
    for item in array:
        i, j = int(
            (item[0] - x_interval[0])
            / x_bin_size
        ), int(
            (item[1] - y_interval[0])
            / y_bin_size
        )
        try:
            binned[i, j] += item[2]
        except IndexError:
            # print (item)
            err_count += 1
            error = True
            pass
    binned = np.transpose(binned)
    if err_count > len(array) / 2:
        print(
            "Error",
            array.shape,
            err_count,
            array,
        )
    if not expand:
        return binned
    return np.expand_dims(binned, -1)
