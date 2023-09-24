import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import (
    Circle,
    FancyArrowPatch,
)
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import (
    Axes3D,
)
from mpl_toolkits.axes_grid1 import (
    make_axes_locatable,
)
from mpl_toolkits.mplot3d import (
    proj3d,
)

try:
    from hep_ml.hep.utils.fatjet import (
        FatJet,
        Print,
    )
    import hep_ml.hep.utils.root_utils as ru
    from ROOT import (
        TVector3,
        TLorentzVector,
    )
except ModuleNotFoundError:
    pass
from hep_ml.hep.config import (
    FinalStates,
)
from hep_ml.hep.config import (
    track_index,
)
import numpy as np
import os
import sys

pwd = os.getcwd()


class Arrow3D(FancyArrowPatch):
    """
    3D arrow patch.

    This class extends the FancyArrowPatch to create 3D arrow patches.

    Args:
        xs (tuple): Tuple of starting x-coordinates for the arrow tail.
        ys (tuple): Tuple of starting y-coordinates for the arrow tail.
        zs (tuple): Tuple of starting z-coordinates for the arrow tail.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Attributes:
        _verts3d (tuple): Tuple of (xs, ys, zs) representing the 3D coordinates
        of the arrow.

    Methods:
        draw(renderer): Draws the 3D arrow patch on the given renderer.

    Usage:
        arrow = Arrow3D((x_start, x_end), (y_start, y_end), (z_start, z_end),
                color='red')
    """

    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = (xs, ys, zs)

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)


class Plotter:
    """
    A class for creating custom plots.

    This class provides a convenient way to create custom plots with various
    options for labels, ranges, figure size, projection, and more.

    Args:
        x_name (str, optional): Label for the x-axis. Defaults to None.
        y_name (str, optional): Label for the y-axis.
            Defaults to None.
        range (dict, optional): Range for x and y axes.
            Defaults to None.
        size (tuple, optional): Figure size (width, height).
            Defaults to (10, 10).
        projection (str, optional): Plot projection
            ('3d', 'image', 'subplots'). Defaults to None.
        title (str, optional): Figure title. Defaults to None.
        set_range (bool, optional): Set axis range based on input range.
            Defaults to False.

    Attributes:
        image_range (dict): Default range x and y axes in image projection.
        fig (Figure): Matplotlib figure object.
        axes (Axes): Matplotlib axes object.
        marker (str): Marker style for scatter plots.
        cmap (str): Colormap for scatter plots.
        markersize (int): Marker size for scatter plots.

    Methods:
        No public methods.

    Usage:
        p = Plotter(x_name="X-axis", y_name="Y-axis", size=(8, 6))
    """

    def __init__(
        self,
        x_name=None,
        y_name=None,
        range=None,
        size=(10, 10),
        projection=None,
        title=None,
        set_range=False,
    ):
        """
        Initialize a custom plotting object.

        Args:
            x_name (str, optional): Label for the x-axis.
                Defaults to None.
            y_name (str, optional): Label for the y-axis.
                Defaults to None.
            range (dict, optional): Range for x and y axes.
                Defaults to None.
            size (tuple, optional): Figure size (width, height).
                Defaults to (10, 10).
            projection (str, optional): Plot projection
            (None, '3d', 'image', 'subplots'). Defaults to None.
            title (str, optional): Figure title. Defaults to None.
            set_range (bool, optional): Set axis range based on input range.
                Defaults to False.
        """
        self.image_range = {
            "x": (-5, 5),
            "y": (-5, 5),
        }
        if range != "image" and not range:
            range = self.image_range
        self.title = title
        self.projection = projection
        if projection != "subplots":
            fig, axes = (
                plt.figure(figsize=size),
                plt.axes(),
            )
            if self.title is not None:
                fig.suptitle(title)
            if projection == "3d":
                axes = fig.add_subplot(
                    111,
                    projection="3d",
                )
            elif projection == "image":
                pass
            else:
                fig.add_subplot(axes)
            if projection is None and range is not None and set_range:
                axes.set_xlim(range["x"]), axes.set_ylim(range["y"])
            if x_name is not None:
                axes.set_xlabel(
                    x_name,
                    fontsize=40,
                )
            if y_name is not None:
                axes.set_ylabel(
                    y_name,
                    fontsize=40,
                )
            (
                self.fig,
                self.axes,
            ) = (
                fig,
                axes,
            )
        else:
            (
                self.fig,
                self.axes,
            ) = (
                None,
                None,
            )
        (
            self.marker,
            self.cmap,
            self.markersize,
        ) = (
            "s",
            "Blues",
            10,
        )

    def SetRange(self, range=None):
        """
        Set the axis range based on the given range dictionary.

        Args:
            range (dict, optional): Range for x and y axes. Defaults to None.

        Returns:
            None
        """
        if range is None:
            range = self.image_range
        self.axes.set_xlim(range["x"]), self.axes.set_ylim(range["y"])
        return

    def Show(self):
        """
        Display the plot.

        Returns:
            None
        """
        plt.show()
        return

    def track_scatter(self, track, **kwargs):
        """
        Create a scatter plot for track data.

        Args:
            track (array-like): Track data to be plotted.
            **kwargs: Additional keyword arguments for scatter plot
                customization.

        Returns:
            None
        """
        if "s" not in kwargs:
            kwargs["s"] = 8
        eta = track[
            :,
            track_index.Eta,
        ]
        phi = track[
            :,
            track_index.Phi,
        ]
        pt = track[:, track_index.PT]
        self.scatter_plot([eta, phi, pt], **kwargs)
        return

    def tower_scatter(self, tower, val="ET", **kwargs):
        """
        Create a scatter plot for tower data.

        Args:
            tower (numpy.ndarray): Tower data containing information such as
                Eta, Phi, and value.
            val (str, optional): Value to be plotted (e.g., "ET"). Defaults to
                "ET".
            **kwargs: Additional keyword arguments for scatter plot
            customization.

        Returns:
            None
        """
        # if "edgecolors" not in kwargs:kwargs["edgecolors"]="b"
        if "s" not in kwargs:
            kwargs["s"] = 8
        tower_map = FinalStates.index_map
        eta = tower[
            :,
            tower_map["Eta"],
        ]
        phi = tower[
            :,
            tower_map["Phi"],
        ]
        Z = tower[:, tower_map[val]]
        self.scatter_plot([eta, phi, Z], **kwargs)
        return

    def Tower(
        self,
        lepton,
        fatjet,
        run_name=None,
        show=False,
        r_fat=1.2,
        r_lep=1.0,
        circle=False,
        **kwargs
    ):
        """Plot lepton, fatjet[0], and fatjet[1] in the (eta, phi) plane,
        colormapped logarithmically with pt.
        numpy.ndarray of TlorentzVectors lepton,fatjet[0],fatjet[1] are
        plotted in the (eta,phi) plane, colormapped logarithmically with pt

        Args:
            lepton (numpy.ndarray or TLorentzVector): Lepton data for plotting.
            fatjet (tuple of numpy.ndarray or TLorentzVector): Tuple
                containing fatjet[0] and fatjet[1] data for plotting.
            run_name (str, optional): Name of the run. Defaults to None.
            show (bool, optional): Display the plot. Defaults to False.
            r_fat (float, optional): Radius for fatjets. Defaults to 1.2.
            r_lep (float, optional): Radius for the lepton. Defaults to 1.0.
            circle (bool, optional): If True, plot markers as circles.
                Defaults to False.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        if isinstance(lepton[0], TLorentzVector):
            (
                lep,
                fat1,
                fat2,
            ) = (
                ru.GetNumpy(np.sum(lepton)),
                ru.GetNumpy(np.sum(fatjet[0])),
                ru.GetNumpy(np.sum(fatjet[1])),
            )
            (
                lep_np,
                fat1_np,
                fat2_np,
            ) = (
                ru.GetNumpy(lepton),
                ru.GetNumpy(fatjet[0]),
                ru.GetNumpy(fatjet[1]),
            )
            # print (lep_np.shape,fat1_np.shape,fat2_np.shape)
        else:
            (
                lep,
                fat1,
                fat2,
            ) = (
                kwargs["lep_vec"],
                kwargs["fat_vec"][0],
                kwargs["fat_vec"][1],
            )
            (
                lep_np,
                fat1_np,
                fat2_np,
            ) = (
                lepton,
                fatjet[0],
                fatjet[1],
            )
        # print (lep_np)
        if circle:
            self.axes.scatter(
                lep[0],
                lep[1],
                c="red",
                s=60,
                marker="o",
            )
            self.axes.scatter(
                fat1[0],
                fat1[1],
                c="blue",
                s=60,
                marker="o",
                cmap="Blues",
                norm=colors.LogNorm(
                    vmin=np.min(fat1_np[2]),
                    vmax=np.max(fat1_np[2]),
                ),
            )
            self.axes.scatter(
                fat2[0],
                fat2[1],
                c="green",
                s=60,
                marker="o",
                cmap="YlGn",
                norm=colors.LogNorm(
                    vmin=np.min(fat1_np[2]),
                    vmax=np.max(fat1_np[2]),
                ),
            )
        else:
            # print ("here")
            self.axes.scatter(
                lep_np[0],
                lep_np[1],
                c="red",
                s=self.markersize,
                marker=self.marker,
            )
            self.axes.scatter(
                fat1_np[0],
                fat1_np[1],
                c=fat1_np[2],
                s=self.markersize,
                marker=self.marker,
                cmap="Blues",
                norm=colors.LogNorm(
                    vmin=np.min(fat1_np[2]),
                    vmax=np.max(fat1_np[2]),
                ),
            )
            self.axes.scatter(
                fat2_np[0],
                fat2_np[1],
                c=fat2_np[2],
                s=self.markersize,
                marker=self.marker,
                cmap="YlGn",
                norm=colors.LogNorm(
                    vmin=np.min(fat1_np[2]),
                    vmax=np.max(fat1_np[2]),
                ),
            )
        self.axes.add_patch(
            Circle(
                (
                    lep[0],
                    lep[1],
                ),
                radius=r_lep,
                fill=False,
                color="red",
                label="Lepton",
            )
        )
        self.axes.add_patch(
            Circle(
                (
                    fat1[0],
                    fat1[1],
                ),
                radius=r_fat,
                fill=False,
                color="Blue",
                label="Fat1",
            )
        )
        self.axes.add_patch(
            Circle(
                (
                    fat2[0],
                    fat2[1],
                ),
                radius=r_fat,
                color="green",
                label="Fat2",
                fill=False,
            )
        )
        self.axes.legend(loc="best")
        # sys.exit()
        # if show == True:
        if show:
            # plt.colorbar()
            plt.show()
        return

    def Tower3D(
        self,
        lepton,
        fatjet,
        run_name=None,
        show=False,
    ):
        """
        Plot a 3D representation of lepton, fatjet[0], and fatjet[1].

        Args:
            lepton (numpy.ndarray): Lepton data for plotting.
            fatjet (tuple of numpy.ndarray): Tuple containing fatjet[0] and
                fatjet[1] data for plotting.
            run_name (str, optional): Name of the run. Defaults to None.
            show (bool, optional): Display the plot. Defaults to False.

        Returns:
            None
        """
        (
            lepton,
            fatjet[0],
            fatjet[1],
        ) = (
            ru.GetNumpy(lepton),
            ru.GetNumpy(fatjet[0]),
            ru.GetNumpy(fatjet[1]),
        )
        # print (lepton.shape,fatjet[0].shape,fatjet[1].shape)
        # sys.exit()
        tot = np.concatenate(
            (
                fatjet[0],
                fatjet[1],
                lepton,
            ),
            axis=1,
        )
        extremum = np.swapaxes(
            np.array(
                [
                    np.min(
                        tot,
                        axis=1,
                    )
                    - 0.2,
                    np.max(
                        tot,
                        axis=1,
                    )
                    + 0.2,
                ]
            ),
            0,
            1,
        )
        self.axes.set_xscale("symlog"), self.axes.set_yscale(
            "symlog"
        ), self.axes.set_zscale("symlog")
        self.axes.set_xlabel("$P_x$"), self.axes.set_ylabel(
            "$P_y$"
        ), self.axes.set_zlabel("$P_z$")
        self.axes.set_xlim(extremum[0]), self.axes.set_ylim(
            extremum[1]
        ), self.axes.set_zlim(extremum[2])
        count = 0
        for item in np.swapaxes(fatjet[0], 0, 1):
            if count == 0:
                self.axes.add_artist(
                    Arrow3D(
                        (
                            0,
                            item[0],
                        ),
                        (
                            0,
                            item[1],
                        ),
                        (
                            0,
                            item[2],
                        ),
                        mutation_scale=20,
                        lw=1,
                        arrowstyle="-|>",
                        color="b",
                        label="Fat1",
                    )
                )
            else:
                self.axes.add_artist(
                    Arrow3D(
                        (
                            0,
                            item[0],
                        ),
                        (
                            0,
                            item[1],
                        ),
                        (
                            0,
                            item[2],
                        ),
                        mutation_scale=20,
                        lw=1,
                        arrowstyle="-|>",
                        color="b",
                    )
                )
        for item in np.swapaxes(fatjet[1], 0, 1):
            if count == 0:
                self.axes.add_artist(
                    Arrow3D(
                        (
                            0,
                            item[0],
                        ),
                        (
                            0,
                            item[1],
                        ),
                        (
                            0,
                            item[2],
                        ),
                        mutation_scale=20,
                        lw=1,
                        arrowstyle="-|>",
                        color="g",
                        label="Fat2",
                    )
                )
            else:
                self.axes.add_artist(
                    Arrow3D(
                        (
                            0,
                            item[0],
                        ),
                        (
                            0,
                            item[1],
                        ),
                        (
                            0,
                            item[2],
                        ),
                        mutation_scale=20,
                        lw=1,
                        arrowstyle="-|>",
                        color="g",
                    )
                )
        for item in np.swapaxes(lepton, 0, 1):
            if count == 0:
                self.axes.add_artist(
                    Arrow3D(
                        (
                            0,
                            item[0],
                        ),
                        (
                            0,
                            item[1],
                        ),
                        (
                            0,
                            item[2],
                        ),
                        mutation_scale=20,
                        lw=1,
                        arrowstyle="-|>",
                        color="r",
                        label="Lepton",
                    )
                )
            else:
                self.axes.add_artist(
                    Arrow3D(
                        (
                            0,
                            item[0],
                        ),
                        (
                            0,
                            item[1],
                        ),
                        (
                            0,
                            item[2],
                        ),
                        mutation_scale=20,
                        lw=1,
                        arrowstyle="-|>",
                        color="r",
                    )
                )
        self.axes.legend(loc="best")
        if show:
            plt.show()
        return

    def Subjet(
        self,
        fatjet,
        label=None,
        show=False,
        r=0.4,
        no_subjets=3,
    ):
        self.FatJet(
            fatjet,
            show=False,
        )
        subjets = FatJet().Recluster(
            fatjet,
            r=r,
            subjets=no_subjets,
        )
        plotting_arrays = FatJet().GetConstituents(
            subjets,
            format="image",
        )
        cmap, color = [
            "Blues",
            "YlGn",
            "RdPu",
        ], ["b", "g", "r"]
        count = 1
        for (
            array,
            coordinate,
        ) in zip(
            plotting_arrays,
            subjets,
        ):
            """
            Plot subjets of a fatjet.

            Args:
                fatjet (numpy.ndarray): Fatjet data.
                label (str, optional): Label for the plot. Defaults to None.
                show (bool, optional): Display the plot. Defaults to False.
                r (float, optional): Radius for subjets. Defaults to 0.4.
                no_subjets (int, optional): Number of subjets to plot.
                    Defaults to 3.

            Returns:
                None
            """
            # self.axes.scatter(array[0],array[1],c=array[2],cmap=cmap[count-1],s=self.markersize,marker=self.marker)
            self.axes.add_patch(
                Circle(
                    (
                        coordinate.eta,
                        coordinate.phi,
                    ),
                    radius=r,
                    fill=False,
                    color=color[count - 1],
                    label="SubJet" + str(count),
                )
            )
            count += 1
        self.axes.legend(loc="best")
        if show:
            plt.show()
        return

    def FatJet(
        self,
        fatjet,
        label=None,
        show=False,
        cmap="Blues",
        s=50,
    ):
        """
        Plot a fat jet.

        Args:
            fatjet (numpy.ndarray): Fatjet data.
            label (str, optional): Label for the plot. Defaults to None.
            show (bool, optional): Display the plot. Defaults to False.
            cmap (str, optional): Colormap for the jet. Defaults to "Blues".
            s (int, optional): Marker size. Defaults to 50.

        Returns:
            None
        """
        # ru.Print(np.sum(fatjet))
        center = [
            np.sum(fatjet).Eta(),
            np.sum(fatjet).Phi(),
        ]
        if len(fatjet.shape) != 2:
            fatjet = ru.GetNumpy(fatjet)
        extremum = np.swapaxes(
            np.array(
                [
                    np.min(
                        fatjet,
                        axis=1,
                    )
                    - 0.2,
                    np.max(
                        fatjet,
                        axis=1,
                    )
                    + 0.2,
                ]
            ),
            0,
            1,
        )
        self.axes.set_xlim(extremum[0]), self.axes.set_ylim(extremum[1])
        self.axes.scatter(
            fatjet[0],
            fatjet[1],
            marker="s",
            s=s,
            c=fatjet[2],
            cmap=cmap,
            norm=colors.LogNorm(
                vmin=np.min(fatjet[2]),
                vmax=np.max(fatjet[2]),
            ),
            label=label,
        )
        self.axes.scatter(
            center[0],
            center[1],
            marker="^",
            s=s,
            c="r",
            label="center",
        )
        if show:
            plt.show()
        return

    def Fat3D(
        self,
        fatjet,
        label=None,
        show=False,
        arrow_color="b",
        axis="on",
        summed=None,
    ):
        """
        Plot a 3D representation of a fat jet.

        Args:
            fatjet (numpy.ndarray): Fatjet data.
            label (str, optional): Label for the plot. Defaults to None.
            show (bool, optional): Display the plot. Defaults to False.
            arrow_color (str, optional): Color of the arrows representing the
                fat jet constituents. Defaults to "b".
            axis (str, optional): Display the axis. Defaults to "on".
            summed (TLorentzVector, optional): Sum of the fatjet constituents.
                Defaults to None.

        Returns:
            None
        """
        Z = np.array(
            [item.P() for item in fatjet],
            dtype="float64",
        )
        if len(fatjet.shape) != 2:
            fatjet = ru.GetNumpy(
                fatjet,
                format="lorentz",
            )
        print(fatjet.shape)
        if summed is None:
            extremum = np.swapaxes(
                np.array(
                    [
                        np.min(
                            fatjet,
                            axis=1,
                        )
                        - 0.2,
                        np.max(
                            fatjet,
                            axis=1,
                        )
                        + 0.2,
                    ]
                ),
                0,
                1,
            )
        else:
            extremum = np.swapaxes(
                np.array(
                    [
                        np.min(
                            np.concatenate(
                                (
                                    fatjet,
                                    [
                                        [summed.Px()],
                                        [summed.Py()],
                                        [summed.Pz()],
                                        [summed.E()],
                                    ],
                                ),
                                axis=1,
                            ),
                            axis=1,
                        )
                        - 0.2,
                        np.max(
                            np.concatenate(
                                (
                                    fatjet,
                                    [
                                        [summed.Px()],
                                        [summed.Py()],
                                        [summed.Pz()],
                                        [summed.E()],
                                    ],
                                ),
                                axis=1,
                            ),
                            axis=1,
                        )
                        + 0.2,
                    ]
                ),
                0,
                1,
            )
        # print (extremum)
        self.axes.set_xscale("symlog"), self.axes.set_yscale(
            "symlog"
        ), self.axes.set_zscale("symlog")
        self.axes.set_xlabel("$P_x$"), self.axes.set_ylabel(
            "$P_y$"
        ), self.axes.set_zlabel("$P_z$")
        self.axes.set_xlim(extremum[0]), self.axes.set_ylim(
            extremum[1]
        ), self.axes.set_zlim(extremum[2])
        if axis == "off":
            self.axes.axis("off")
        # self.axes.scatter(fatjet[0],fatjet[1],fatjet[2],c=Z,cmap="Blues",
        # norm=colors.LogNorm(vmin=np.min(Z), vmax=np.max(Z)))
        count = 0
        for item in np.swapaxes(fatjet, 0, 1):
            if count == 0:
                self.axes.add_artist(
                    Arrow3D(
                        (
                            0,
                            item[0],
                        ),
                        (
                            0,
                            item[1],
                        ),
                        (
                            0,
                            item[2],
                        ),
                        mutation_scale=20,
                        lw=1,
                        arrowstyle="-|>",
                        color=arrow_color,
                        label=label,
                    )
                )
            else:
                self.axes.add_artist(
                    Arrow3D(
                        (
                            0,
                            item[0],
                        ),
                        (
                            0,
                            item[1],
                        ),
                        (
                            0,
                            item[2],
                        ),
                        mutation_scale=20,
                        lw=1,
                        arrowstyle="-|>",
                        color=arrow_color,
                    )
                )
        if summed is not None:
            self.axes.add_artist(
                Arrow3D(
                    (
                        0,
                        summed.X(),
                    ),
                    (
                        0,
                        summed.Y(),
                    ),
                    (
                        0,
                        summed.Z(),
                    ),
                    mutation_scale=20,
                    lw=1,
                    arrowstyle="-|>",
                    color="k",
                    label="Summed",
                )
            )
        if show:
            self.axes.legend(loc="best")
            plt.show()
        return

    def plot_axes(self, axes):
        """
        Plot axes on the given axes.

        Args:
            axes (matplotlib.axes.Axes): The axes to plot the axes on.

        Returns:
            None
        """
        x_bound = axes.get_xbound()
        axes.plot(
            np.linspace(
                x_bound[0],
                x_bound[1],
                100,
            ),
            np.zeros(100),
            "--k",
            alpha=0.5,
        )
        y_bound = axes.get_ybound()
        axes.plot(
            np.zeros(100),
            np.linspace(
                y_bound[0],
                y_bound[1],
                100,
            ),
            "--k",
            alpha=0.5,
        )

    def save_fig(
        self,
        title,
        extension="png",
        dpi=100,
        legend_axes=[0],
        save_path="plots",
        set_legends=False,
        **kwargs
    ):
        """
        Save the current figure to a file.

        Args:
            title (str): The title of the saved figure.
            extension (str, optional): The file extension for the saved fig.
                Defaults to "png".
            dpi (int, optional): The resolution (dots per inch) for the saved
                figure. Defaults to 100.
            legend_axes (list, optional): Indices of axes to include legends
                for. Defaults to [0].
            save_path (str, optional): The directory path where the fig will
                be saved. Defaults to "plots".
            set_legends (bool, optional): Whether to set legends on the plot
                Defaults to False.
            **kwargs: Additional keyword arguments for customization.

        Returns:
            None
        """
        pwd = os.getcwd()
        plot_axes = kwargs.get(
            "plot_axes",
            False,
        )
        if plot_axes:
            if self.projection == "subplots":
                [self.plot_axes(item) for item in self.axes]
            else:
                self.plot_axes(self.axes)
        if save_path == "plots":
            dirs = os.listdir(os.getcwd())
            if "plots" not in dirs:
                os.mkdir("plots")
            os.chdir(pwd + "/plots")
        else:
            os.chdir(save_path)
        if self.projection == "subplots":
            for (
                i,
                item,
            ) in enumerate(self.axes):
                if i in legend_axes and set_legends:
                    item.legend(
                        loc=kwargs.get(
                            "legend_loc",
                            "best",
                        ),
                        prop={
                            "size": kwargs.get(
                                "legend_size",
                                25,
                            )
                        },
                        title=kwargs.get("legend_title"),
                        title_fontsize=kwargs.get(
                            "legend_title_fontsize",
                            30,
                        ),
                    )
                item.tick_params(
                    axis="both",
                    which="major",
                    labelsize=kwargs.get(
                        "tick_label_size",
                        28,
                    ),
                )
        else:
            self.axes.tick_params(
                axis="both",
                which="major",
                labelsize=kwargs.get(
                    "tick_label_size",
                    28,
                ),
            )
            if set_legends:
                self.axes.legend(
                    loc=kwargs.get(
                        "legend_loc",
                        "best",
                    ),
                    prop={
                        "size": kwargs.get(
                            "legend_size",
                            25,
                        )
                    },
                    title=kwargs.get("legend_title"),
                    title_fontsize=kwargs.get(
                        "legend_title_fontsize",
                        30,
                    ),
                )
        self.fig.savefig(
            title + "." + extension,
            format=extension,
            dpi=dpi,
            bbox_inches=kwargs.get(
                "bbox_inches",
                "tight",
            ),
            pad_inches=kwargs.get("pad", 0.4),
        )
        # plt.show(block=False)
        plt.close()
        print(
            "Output plot saved at ",
            os.getcwd(),
            title + "." + extension,
        )
        os.chdir(pwd)
        return

    def Image(
        self,
        array,
        title=None,
        cmap="viridis_r",
        show=False,
        extension="eps",
        set_colorbar=True,
        log_scale=False,
        **kwargs
    ):
        """
        Display an image representation of a 2D array.

        Args:
            array (numpy.ndarray): The 2D array to display as an image.
            title (str, optional): The title of the image plot. Defaults to
                None.
            cmap (str, optional): The colormap for the image. Defaults to
                "viridis_r".
            show (bool, optional): Whether to display the image plot. Defaults
                False.
            extension (str, optional): The file extension for saving the img.
                Defaults to "eps".
            set_colorbar (bool, optional): Whether to add a colorbar to plot.
                Defaults to True.
            log_scale (bool, optional): Whether to use a logarithmic scale
                for color mapping. Defaults to False.
            **kwargs: Additional keyword arguments for customization.

        Returns:
            matplotlib.image.AxesImage: The image object.
        """
        # plt.title(title)
        if array.shape[-1] == 1:
            array = np.squeeze(array)
        if log_scale:
            # array=array+np.min(array[ array != 0])
            print(
                np.min(array[array != 0]),
                np.max(array),
            )
            # if self.projection=="subplots":
            # im=self.axes.matshow(array,cmap=cmap,origin="lower",
            # norm=colors.LogNorm(vmin=kwargs.get("vmin",1),
            # vmax=kwargs.get("vmax",200)) )
            im = self.axes.matshow(
                array,
                cmap=cmap,
                origin="lower",
                norm=colors.LogNorm(
                    vmin=kwargs.get(
                        "vmin",
                        np.min(array[array != 0]),
                    ),
                    vmax=kwargs.get(
                        "vmax",
                        np.max(array),
                    ),
                ),
            )
        else:
            im = self.axes.matshow(
                array,
                cmap=cmap,
                origin="lower",
            )  # ,norm=colors.LogNorm(vmin=np.min(array), vmax=np.max(array)))
        if self.projection != "subplots" and set_colorbar:
            self.set_colorbar(im)
        if show:
            plt.show()
        return im

    def set_colorbar(
        self, im, cax=None, axes=None, ylabel=None, ylabelsize=30, clim=None, **kwargs
    ):
        """
        Add a colorbar to the current plot.

        Args:
            im (matplotlib.image.AxesImage): The image object for which to add
                the colorbar.
            cax (matplotlib.axes.Axes, optional): The axes to use for the
                colorbar. Defaults to None.
            axes (matplotlib.axes.Axes, optional): The axes associated with
                colorbar. Defaults to None.
            ylabel (str, optional): The label for the colorbar. Defaults None.
            ylabelsize (int, optional): The font size for the colorbar label.
                Defaults to 30.
            clim (tuple, optional): The data value limits for the colorbar.
                Defaults to None.
            **kwargs: Additional keyword arguments for customization.

        Returns:
            matplotlib.colorbar.Colorbar: The colorbar object.
        """
        if cax is None:
            if axes is None:
                divider = make_axes_locatable(self.axes)
                cax = divider.append_axes(
                    "right",
                    size="5%",
                    pad=kwargs.get(
                        "pad",
                        0.05,
                    ),
                    norm=kwargs.get("norm"),
                )
                cbar = self.fig.colorbar(
                    im,
                    cax=cax,
                    orientation="vertical",
                )  # format='%.0e'
            else:
                cbar = self.fig.colorbar(
                    im,
                    ax=axes,
                    pad=kwargs.get(
                        "pad",
                        0.05,
                    ),
                )  # format='%.0e'
        else:
            cbar = self.fig.colorbar(
                im,
                cax=cax,
                format="%.0e",
                norm=kwargs.get("norm"),
            )
        cbar.ax.tick_params(
            axis="y",
            which="major",
            labelsize=kwargs.get(
                "labelsize",
                30,
            ),
        )
        if clim is not None:
            # cbar.vmin,cbar.vmax=clim[0],clim[1]
            # print (cbar.vmin,cbar.vmax)
            # sys.exit()
            cbar.set_clim(*clim)
        if ylabel is not None:
            cbar.ax.set_ylabel(
                ylabel,
                labelpad=0.01,
                size=ylabelsize,
            )
        return cbar

    def scatter_plot(
        self,
        array,
        title=None,
        cmap="viridis_r",
        label=None,
        marker="s",
        show=False,
        s=30,
        c=None,
        log_scale=False,
        set_colorbar=False,
        **kwargs
    ):  # plot jet image with cmap
        """
        Create a scatter plot of data points.

        Args:
            array (numpy.ndarray): The data array containing X, Y, and Z
                coordinates.
            title (str, optional): The title of the scatter plot. Defaults
                None.
            cmap (str, optional): The colormap to use for coloring points.
                Defaults to "viridis_r".
            label (str, optional): The label for the data points. Defaults
                None.
            marker (str, optional): The marker style for data points.
                Defaults to "s".
            show (bool, optional): Whether to display the plot.
                Defaults to False.
            s (int, optional): The size of markers for data points. Defaults to
                30.
            c (array-like, optional): The array for color mapping data points.
                Defaults to None.
            log_scale (bool, optional): Whether to apply a logarithmic scale
                to the color mapping. Defaults to False.
            set_colorbar (bool, optional): Whether to add a colorbar to the
                plot. Defaults to False.
            **kwargs: Additional keyword arguments for customization.

        Returns:
            matplotlib.collections.PathCollection: The scatter plot object.
        """
        if "axes" in kwargs:
            axes = kwargs.pop("axes")
        else:
            axes = None

        if isinstance(array[0], TLorentzVector):
            array = np.array([[item.Eta(), item.Phi(), item.Pt()] for item in array])
            array = np.swapaxes(array, 0, 1)
        elif isinstance(array[0], TVector3):
            array = np.array([[item.X(), item.Y(), item.Z()] for item in array])

        # if array.shape[0]!=3 or: array=np.swapaxes(array,0,1)
        X, Y, Z = (
            array[0],
            array[1],
            array[2],
        )
        if log_scale:
            if np.min(Z) < 0:
                assert abs(np.min(Z)) < 10e-12
                Z = Z + np.min(Z)
            Z = Z + np.min(Z[Z != 0]) * 0.000000001
        if cmap is None:
            color = c
        else:
            color = Z
        if log_scale:
            im = self.axes.scatter(
                X,
                Y,
                marker=marker,
                s=s,
                c=color,
                cmap=cmap,
                norm=colors.LogNorm(
                    vmin=np.min(Z),
                    vmax=np.max(Z),
                ),
                label=label,
                **kwargs
            )
        else:
            im = self.axes.scatter(
                X, Y, marker=marker, s=s, c=color, cmap=cmap, label=label, **kwargs
            )
        if set_colorbar:
            self.set_colorbar(im, axes=axes)
        if show:
            plt.show()
        return im
