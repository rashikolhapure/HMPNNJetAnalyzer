import sys


import numpy as np
import matplotlib as mpl

import matplotlib.pyplot as plt
from matplotlib.patches import (
    Circle,
    Rectangle,
)

from .plotter import Plotter

colors = {
    0: "navy",
    1: "royalblue",
    2: "cyan",
    3: "lime",
    4: "gold",
}


def bar_plot(data=None, labels=None, per_label=True, label_rotate=0, **kwargs):
    """
    Create a horizontal bar plot with optional customization.

    Args:
        data (list): List of data values to be plotted.
        labels (list): List of labels for the data values.
        per_label (bool, optional): If True, display labels for each data point. If False, only display labels for data points with non-zero values. Default is True.
        label_rotate (int, optional): Rotation angle for y-axis labels. Default is 0.
        **kwargs: Additional keyword arguments for customizing the plot.

    Keyword Args:
        name (str, optional): Name for saving the plot. Default is "unamed".
        save_path (str, optional): Directory path for saving the plot. Default is "./plots".
        add_text (list, optional): List of additional text to be added next to the data values.
        title (str, optional): Title for the plot.
        color (list, optional): List of colors for the bars.
        color_to_label (dict, optional): Dictionary mapping colors to labels for legend.
        xlim (tuple, optional): Tuple specifying the x-axis limits. Default is (0.5, 1.0).

    Returns:
        None
    """
    # plt.style.use('seaborn')
    save_name = kwargs.pop("name", "unamed")
    save_path = kwargs.pop(
        "save_path",
        "./plots",
    )
    # this is for plotting purpose
    p = Plotter(
        x_name="AUC",
        y_name="",
        range={
            "x": (0.5, 1),
            "y": (
                -0.5,
                len(data),
            ),
        },
        size=(16.02, 10),
    )

    # p.fig.tight_layout()
    index = np.arange(len(labels))
    if "add_text" in kwargs:
        add_text = kwargs.pop("add_text")
    else:
        add_text = None
    count = 0
    new_labels = []
    new_indices = []
    for x, y, z in zip(data, index, labels):
        print(x, y, z)
        if x > 0:
            if add_text is not None:
                p.axes.text(
                    x + 0.02,
                    y - 0.2,
                    str(
                        round(
                            x,
                            3,
                        )
                    )
                    + " "
                    + add_text[count],
                    fontsize=15,
                )
            else:
                p.axes.text(
                    x + 0.02,
                    y - 0.2,
                    str(
                        round(
                            x,
                            4,
                        )
                    ),
                    fontsize=40,
                )
        else:
            # p.axes.text(0.5,y-0.4,z,fontsize=18)
            new_labels.append(z), new_indices.append(count - 1)
            labels[count] = ""
        count += 1
    if per_label:
        (
            new_labels,
            new_indices,
        ) = (
            labels,
            index,
        )
    if kwargs.get("title", None) is not None:
        title = kwargs.pop("title")
    else:
        title = None
    if "color" in kwargs:
        colors = kwargs.get("color")
        try:
            (
                color_to_label,
                set_legends,
            ) = (
                kwargs.pop("color_to_label"),
                True,
            )
        except KeyError:
            (
                color_to_label,
                set_legends,
            ) = ({}, False)
        for item in np.unique(colors):
            print(
                item,
                np.unique(colors),
            )
            if item not in color_to_label:
                continue
            extra = Rectangle(
                (0, 0),
                0,
                0,
                color=item,
                fill=True,
                edgecolor="none",
                linewidth=0,
                label=color_to_label[item],
            )
            p.axes.add_patch(extra)
    p.axes.set_xlim(
        kwargs.pop(
            "xlim",
            (0.5, 1.0),
        )
    )
    p.axes.barh(index, data, 1.0, **kwargs)

    p.axes.set_yticks(new_indices)
    p.axes.set_yticklabels(new_labels)
    plt.xticks(
        np.linspace(0.5, 1.0, 6),
        fontsize=28,
    )
    p.axes.tick_params(
        axis="y",
        which="major",
        labelrotation=label_rotate,
        length=3,
        labelsize=100,
    )
    if set_legends:
        p.axes.legend(
            loc="best",
            prop={"size": 20},
        )
    print(
        new_labels,
        new_indices,
    )
    # sys.exit()
    if title is not None:
        y_lim = p.axes.get_ylim()
        x_lim = p.axes.get_xlim()
        p.axes.text(
            x_lim[0] + 0.05 * (x_lim[1] - x_lim[0]),
            y_lim[1] - 0.005 * (y_lim[1] - y_lim[0]),
            title,
            fontsize=35,
            fontdict={
                "weight": "bold",
                "c": "k",
            },
            bbox=dict(
                boxstyle="square",
                facecolor="w",
                alpha=1,
            ),
        )
    p.save_fig(
        save_name,
        save_path=save_path,
        extension="pdf",
    )
    return


def plot_tower_jets(
    array,
    lorentz_jet,
    name=None,
    lorentz_met=None,
    path="./plots",
    dpi=100,
    plotter=None,
    scatter_kwargs={},
):
    """
    Plot tower jets and optionally MET on a polar plot.

    Args:
        array (numpy.ndarray): Tower data to be plotted.
        lorentz_jet (list): List of Lorentz vectors representing jets.
        name (str, optional): Name for saving the plot. Default is None.
        lorentz_met (LorentzVector, optional): Lorentz vector representing MET (missing transverse energy). Default is None.
        path (str, optional): Directory path for saving the plot. Default is "./plots".
        dpi (int, optional): Dots per inch for the saved plot. Default is 100.
        plotter (Plotter, optional): An instance of Plotter to use for plotting. If None, a new Plotter instance will be created. Default is None.
        scatter_kwargs (dict, optional): Additional keyword arguments for customizing the tower scatter plot.

    Returns:
        None or Plotter: If plotter is None, returns None. If plotter is provided, returns the Plotter instance.
    """
    if plotter is None:
        p = Plotter()
    else:
        p = plotter
    if lorentz_met:
        p.axes.scatter(
            [0.0],
            [lorentz_met.Phi()],
            marker="^",
            c="r",
            s=40,
            label="$MET$",
        )
    for i in range(len(lorentz_jet)):
        colors[i + 4] = "yellow"
    for i, item in enumerate(lorentz_jet):
        p.axes.add_patch(
            Circle(
                (
                    item.Eta(),
                    item.Phi(),
                ),
                radius=0.5,
                fill=False,
                color=colors[i],
                label="$j_"
                + str(i)
                + "\\;p_T=$"
                + str(
                    round(
                        item.Pt(),
                        1,
                    )
                )
                + " GeV",
            )
        )
    # p.axes.add_patch(Circle((lorentz_jet[1].Eta(),lorentz_jet[1].Phi()),
    # radius=0.5,fill=False,color="Blue",label="$j_2\;p_T=$"+str(round(lorentz_jet[1].Pt(),1))+"
    # GeV"))
    p.tower_scatter(array, **scatter_kwargs)
    if plotter is None:
        p.axes.legend(loc="best")
        p.save_fig(
            name,
            save_path=path,
            dpi=dpi,
        )
    else:
        return p


def subplot_compare_pileup(
    tower_no_pileup,
    tower_pileup,
    no_pileup_jets,
    pileup_jets,
    name,
    path="./plots",
    index="",
):
    """
    Create a comparison subplot of tower data with and without pileup.

    Args:
        tower_no_pileup (numpy.ndarray): Tower data without pileup.
        tower_pileup (numpy.ndarray): Tower data with pileup.
        no_pileup_jets (list): List of jets without pileup.
        pileup_jets (list): List of jets with pileup.
        name (str): Name for saving the plot.
        path (str, optional): Directory path for saving the plot. Default is "./plots".
        index (str, optional): Index or identifier for the plot. Default is an empty string.

    Returns:
        None
    """
    p = Plotter(projection="subplots")
    for i in range(len(pileup_jets)):
        colors[i + 4] = "yellow"
    fig, axes = plt.subplots(
        ncols=2,
        figsize=(20, 10),
    )
    p.fig = fig
    names = {
        0: "no_pileup",
        1: "pileup",
    }
    jets = {
        0: no_pileup_jets,
        1: pileup_jets,
    }
    for (
        j,
        array,
    ) in enumerate(
        [
            tower_no_pileup,
            tower_pileup,
        ]
    ):
        p.axes = axes[j]
        p.axes.set_title(names[j] + " index " + index)
        for (
            i,
            item,
        ) in enumerate(jets[j]):
            p.axes.add_patch(
                Circle(
                    (
                        item.Eta(),
                        item.Phi(),
                    ),
                    radius=0.5,
                    fill=False,
                    color=colors[i],
                    label="$j_"
                    + str(i)
                    + "\\;p_T=$"
                    + str(
                        round(
                            item.Pt(),
                            1,
                        )
                    )
                    + " GeV",
                )
            )
        p.tower_scatter(array)
        if j == 0:
            p.axes.legend(loc="best")
    p.save_fig(
        name,
        save_path=path,
        dpi=300,
    )
    return


def compare_pileup_image(
    no_pileup,
    pileup,
    name,
    names={
        0: "no_pileup",
        1: "pileup",
    },
    path="./plots",
    index="",
):
    """
    Create a comparison subplot of pileup images with and without pileup.

    Args:
        no_pileup (numpy.ndarray): Pileup image without pileup.
        pileup (numpy.ndarray): Pileup image with pileup.
        name (str): Name for saving the plot.
        names (dict, optional): Dictionary mapping indices to names for labeling. Default is {0: "no_pileup", 1: "pileup"}.
        path (str, optional): Directory path for saving the plot. Default is "./plots".
        index (str, optional): Index or identifier for the plot. Default is an empty string.

    Returns:
        None
    """
    p = Plotter(projection="subplots")
    fig, axes = plt.subplots(
        ncols=2,
        figsize=(20, 10),
    )
    p.fig = fig
    jets = {
        0: no_pileup,
        1: pileup,
    }
    for (
        j,
        array,
    ) in enumerate([no_pileup, pileup]):
        p.axes = axes[j]
        p.axes.set_title(names[j] + " index " + index)
        p.Image(array)
    p.save_fig(
        name,
        save_path=path,
        dpi=300,
    )
    return


def seperate_image_plot(
    left_bin, center_bin, right_bin, save_path=None, **kwargs
):
    """
    Create a subplot of three images.

    Args:
        left_bin (numpy.ndarray): Left image.
        center_bin (numpy.ndarray): Center image.
        right_bin (numpy.ndarray): Right image.
        save_path (str, optional): Directory path for saving the plot. Default is None.
        **kwargs: Additional keyword arguments.

    Keyword Args:
        name (str, optional): Name for saving the plot. Default is "separated_image".

    Returns:
        None
    """
    P = Plotter(
        projection="subplots",
    )
    (
        P.fig,
        axes,
    ) = plt.subplots(
        ncols=3,
        figsize=(15, 10),
    )
    P.axes = axes[0]
    P.Image(left_bin)
    P.axes = axes[1]
    P.Image(center_bin)
    P.axes = axes[2]
    P.Image(right_bin)
    if save_path is not None:
        P.save_fig(
            kwargs.get(
                "name",
                "seperated_image",
            )
        )
    else:
        P.Show()
    return
