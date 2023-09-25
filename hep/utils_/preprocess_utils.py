from typing import (
    Dict,
    List,
    Tuple,
    Union,
)
from ..config import (
    tower_index,
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
import os
import sys
import math

import numpy as np

np.set_printoptions(precision=16)


"""IMAGE PREPROCESSING"""


def translate(
    *args: Union[np.ndarray, float],
    **kwargs: float,
) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    """
    Translate the coordinates of an array of elements.

    Parameters:
    ----------
    *args : numpy.ndarray or float
        An array of elements with coordinates in (X, Y) or a float.
    **kwargs : dict
        Keyword arguments specifying the translation amounts, 'x' and 'y'.

    Returns:
    -------
    numpy.ndarray or tuple
        If a single array is provided, it returns the translated array.
        If multiple arrays are provided, it returns a tuple of translated
        arrays.

    Notes:
    -----
    This function translates the coordinates of an array or multiple
    arrays by subtracting the specified 'x' and 'y' translation amounts.
    The translation is applied element-wise to each array in *args.
    """
    for item in args:
        # print (item.shape)
        item[0] = item[0] - kwargs["x"]
        item[1] = item[1] - kwargs["y"]
    if len(args) == 1:
        return args[0]
    return args


def rotate(
    *args: np.ndarray,
    **kwargs: float
) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
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


def reflect(
    *args: np.ndarray
) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    """Reflect along x axis"""
    for item in args:
        # print(item.shape)
        item[1] = -item[1]
    if len(args) == 1:
        return args[0]
    return args


# FAT JET
def process_fatjets(
    fatjets: List[np.ndarray],
    operation: str = "all",
    subparts: str = "subjets",
    **kwargs: Union[float, Tuple[float, float]]
) -> np.ndarray:
    """Regularize tower/fatjet in (eta,phi) plane wih translation to
    subpart[0], rotate such the subpart[1] is at eta=0, and reflect
    such that subpart[2] is at the positive phi

    Regularize and process a list of fatjets.

    Parameters:
    ----------
    fatjets : list
        A list of fatjets to be processed.
    operation : str, optional
        The type of operation to perform (default is "all").
    subparts : str, optional
        The subparts to consider (default is "subjets").
    **kwargs : dict
        Additional keyword arguments for customization.

    Returns:
    -------
    numpy.ndarray
        An array containing the processed fatjets.

    Notes:
    -----
    This function regularizes and processes a list of fatjets. It translates
    each
    fatjet's subparts to a specified (x, y) point, rotates the fatjet, and
    reflects it if necessary. The processed fatjets are binned into a grid
    defined by 'shape' and 'x_interval'/'y_interval' parameters.
    """
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


def regularize_fatjet(
    fatjet: np.ndarray,
    r: float = 1.2
) -> Tuple[np.ndarray, np.ndarray]:
    """<fatjet> has constituents as TLorentzVector return array f TVector3 with
    (eta,phi,pt) axes, regulates phi such that all components lie inside fatjet
    radius R in the Euclidean (eta,phi) plane, reclusters the fatjet with CA
    algorithm with r=0.4 and returns them in the same (eta,phi,pt) format

    Regularize a fatjet in the (eta, phi, pt) plane.

    Parameters:
    ----------
    fatjet : TLorentzVector
        The fatjet with constituents as TLorentzVectors.
    r : float, optional
        The regularization radius (default is 1.2).

    Returns:
    -------
    numpy.ndarray, numpy.ndarray
        Two arrays of (eta, phi, pt) components representing the regularized
        fatjet and its subjets.

    Notes:
    -----
    This function regularizes a fatjet in the (eta, phi, pt) plane by ensuring
    that all components lie inside the fatjet radius R in the Euclidean
    (eta, phi) plane. It also reclusters the fatjet with the CA algorithm
    using a specified radius (r=0.4) and returns both the fatjet and its
    subjets in the same (eta, phi, pt) format.
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


def remove_jets(
    lorentz_tower: List[np.ndarray],
    lorentz_jets: List[np.ndarray],
    r: float = 0.5,
    **kwargs: Dict[str, Union[bool, float]]
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Remove jet constituents from a Lorentz tower.

    Parameters:
    ----------
    lorentz_tower : list
        A list of Lorentz vectors representing the Lorentz tower.
    lorentz_jets : list
        A list of Lorentz vectors representing the Lorentz jets.
    r : float, optional
        The maximum distance (DeltaR) to consider for constituent removal
        (default is 0.5).
    **kwargs : dict
        Additional keyword arguments for customization.

    Returns:
    -------
    numpy.ndarray or tuple
        Depending on the keyword arguments, it returns the removed
        constituents, or it returns a tuple containing the removed
        constituents and other constituents not removed.

    Notes:
    -----
    This function removes constituents from a Lorentz tower that are within a
    specified distance 'r' of any of the Lorentz jets. The removed
    constituents are returned as a numpy array. Additional keyword argument
    can control the behavior of the function, such as sorting by PT or
    separating central and other constituents.
    """
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
    images: np.ndarray,
    eta_axis: int = 2,
    phi_axis: int = 1,
    eta_range: Tuple[float, float] = (-5, 5),
    phi_range: Tuple[float, float] = (
        -np.pi,
        np.pi
    )
) -> np.ndarray:
    """
    Convert images to (eta, phi, pt) variables.

    Parameters:
    ----------
    images : numpy.ndarray
        A multi-dimensional array containing the input images.
    eta_axis : int, optional
        The axis representing eta in the input images (default is 2).
    phi_axis : int, optional
        The axis representing phi in the input images (default is 1).
    eta_range : tuple, optional
        The range of eta values (default is (-5, 5)).
    phi_range : tuple, optional
        The range of phi values (default is (-pi, pi)).

    Returns:
    -------
    numpy.ndarray
        An array of (eta, phi, pt) variables corresponding to the input images.

    Notes:
    -----
    This function converts input images to variables in the (eta, phi, pt)
    format. The input images are expected to have axes representing
    (batch, eta, phi, channel), and this function swaps the axes
    to obtain the variables (eta, phi, pt) for each image in the batch.
    """
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
    tower: np.ndarray,
    pad_axis: int = 0,
    pad_size: int = 4
) -> np.ndarray:
    """
    Apply padding to a tower along a specified axis.

    Parameters:
    ----------
    tower : numpy.ndarray
        The input tower to which padding is applied.
    pad_axis : int, optional
        The axis along which padding is applied
        (0 for rows, 1 for columns, default is 0).
    pad_size : int, optional
        The size of padding to apply (default is 4).

    Returns:
    -------
    numpy.ndarray
        The tower with padding applied along the specified axis.

    Notes:
    -----
    This function applies padding to a tower along a specified axis.
    Padding is added symmetrically to both sides of the specified axis,
    and the resulting tower is returned.
    """
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


def tower_bin(
    tower: np.ndarray,
    format: str = "tower",
    **kwargs
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Bin tower data into a grid.

    Parameters:
    ----------
    tower : numpy.ndarray
        The input tower data.
    format : str, optional
        The format of the input tower data (default is "tower").
    **kwargs : dict
        Additional keyword arguments for customization.

    Returns:
    -------
    numpy.ndarray or tuple
        Depending on the keyword arguments, it returns the binned tower data
        or a tuple containing separate binned data for different regions.

    Notes:
    -----
    This function bins tower data into a grid format. It supports
    customizations such as bin size, format conversion, and returning
    separate binned data for different regions.
    """
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
        tower_left = tower[tower[:, 1] < -1.6]
        tower_center = tower[
            np.logical_and(
                tower[:, 1] >= -1.6,
                tower[:, 1] <= 1.6,
            )
        ]
        tower_right = tower[tower[:, 1] > 1.6]
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
                -1.6,
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
            x_interval=(
                -1.6,
                1.6,
            ),
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
                1.6,
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
    array: np.ndarray,
    x_interval: Tuple[float, float] = (-1.6, 1.6),
    y_interval: Tuple[float, float] = (-1.6, 1.6),
    expand: bool = False,
    swap: bool = False,
    **kwargs
) -> Union[np.ndarray, np.ndarray]:
    """
    Bin an array of data into a grid.

    Parameters:
    ----------
    array : numpy.ndarray
        The input array of data.
    x_interval : tuple, optional
        The x-axis interval for binning (default is (-1.6, 1.6)).
    y_interval : tuple, optional
        The y-axis interval for binning (default is (-1.6, 1.6)).
    expand : bool, optional
        Whether to expand the result with an extra dimension
        (default is False).
    swap : bool, optional
        Whether to swap the axes of the input array
        (default is False).
    **kwargs : dict
        Additional keyword arguments for customization.

    Returns:
    -------
    numpy.ndarray
        The binned data in a grid format.

    Notes:
    -----
    This function takes an array of data and bins it into a grid format based
    on the specified intervals or bin sizes. It supports options for expanding
    the result with an extra dimension and swapping the axes
    of the input array.
    """
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
