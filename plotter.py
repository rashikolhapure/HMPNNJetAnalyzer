import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import proj3d

try:
    from hep_ml.hep.utils.fatjet import FatJet, Print
    import hep_ml.hep.utils.root_utils as ru
    from ROOT import TVector3, TLorentzVector
except ModuleNotFoundError:
    pass
from hep_ml.hep.config import FinalStates
from hep_ml.hep.config import track_index
import numpy as np
import os
import sys

pwd = os.getcwd()


class Arrow3D(FancyArrowPatch):
    """3d arrow patch"""

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


class Plotter:
    """class to illustrate different steps of preprocessing whole event or fatjet"""

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
        self.image_range = {"x": (-5, 5), "y": (-5, 5)}
        if range != "image" and not range:
            range = self.image_range
        self.title = title
        self.projection = projection
        if projection != "subplots":
            fig, axes = plt.figure(figsize=size), plt.axes()
            if self.title is not None:
                fig.suptitle(title)
            if projection == "3d":
                axes = fig.add_subplot(111, projection="3d")
            elif projection == "image":
                pass
            else:
                fig.add_subplot(axes)
            if projection == None and range != None and set_range:
                axes.set_xlim(range["x"]), axes.set_ylim(range["y"])
            if x_name is not None:
                axes.set_xlabel(x_name, fontsize=40)
            if y_name is not None:
                axes.set_ylabel(y_name, fontsize=40)
            self.fig, self.axes = fig, axes
        else:
            self.fig, self.axes = None, None
        self.marker, self.cmap, self.markersize = "s", "Blues", 10

    def SetRange(self, range=None):
        if range == None:
            range = self.image_range
        self.axes.set_xlim(range["x"]), self.axes.set_ylim(range["y"])
        return

    def Show(self):
        plt.show()
        return

    def track_scatter(self, track, **kwargs):
        if "s" not in kwargs:
            kwargs["s"] = 8
        eta = track[:, track_index.Eta]
        phi = track[:, track_index.Phi]
        pt = track[:, track_index.PT]
        self.scatter_plot([eta, phi, pt], **kwargs)
        return

    def tower_scatter(self, tower, val="ET", **kwargs):
        # if "edgecolors" not in kwargs:kwargs["edgecolors"]="b"
        if "s" not in kwargs:
            kwargs["s"] = 8
        tower_map = FinalStates.index_map
        eta = tower[:, tower_map["Eta"]]
        phi = tower[:, tower_map["Phi"]]
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
        """numpy.ndarray of TlorentzVectors lepton,fatjet[0],fatjet[1] are plotted in the (eta,phi) plane, colormapped logarithmically with pt"""
        if type(lepton[0]) == TLorentzVector:
            lep, fat1, fat2 = (
                ru.GetNumpy(np.sum(lepton)),
                ru.GetNumpy(np.sum(fatjet[0])),
                ru.GetNumpy(np.sum(fatjet[1])),
            )
            lep_np, fat1_np, fat2_np = (
                ru.GetNumpy(lepton),
                ru.GetNumpy(fatjet[0]),
                ru.GetNumpy(fatjet[1]),
            )
            # print (lep_np.shape,fat1_np.shape,fat2_np.shape)
        else:
            lep, fat1, fat2 = (
                kwargs["lep_vec"],
                kwargs["fat_vec"][0],
                kwargs["fat_vec"][1],
            )
            lep_np, fat1_np, fat2_np = lepton, fatjet[0], fatjet[1]
        # print (lep_np)
        if circle:
            self.axes.scatter(lep[0], lep[1], c="red", s=60, marker="o")
            self.axes.scatter(
                fat1[0],
                fat1[1],
                c="blue",
                s=60,
                marker="o",
                cmap="Blues",
                norm=colors.LogNorm(
                    vmin=np.min(fat1_np[2]), vmax=np.max(fat1_np[2])
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
                    vmin=np.min(fat1_np[2]), vmax=np.max(fat1_np[2])
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
                    vmin=np.min(fat1_np[2]), vmax=np.max(fat1_np[2])
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
                    vmin=np.min(fat1_np[2]), vmax=np.max(fat1_np[2])
                ),
            )
        self.axes.add_patch(
            Circle(
                (lep[0], lep[1]),
                radius=r_lep,
                fill=False,
                color="red",
                label="Lepton",
            )
        )
        self.axes.add_patch(
            Circle(
                (fat1[0], fat1[1]),
                radius=r_fat,
                fill=False,
                color="Blue",
                label="Fat1",
            )
        )
        self.axes.add_patch(
            Circle(
                (fat2[0], fat2[1]),
                radius=r_fat,
                color="green",
                label="Fat2",
                fill=False,
            )
        )
        self.axes.legend(loc="best")
        # sys.exit()
        if show == True:
            # plt.colorbar()
            plt.show()
        return

    def Tower3D(self, lepton, fatjet, run_name=None, show=False):
        lepton, fatjet[0], fatjet[1] = (
            ru.GetNumpy(lepton),
            ru.GetNumpy(fatjet[0]),
            ru.GetNumpy(fatjet[1]),
        )
        # print (lepton.shape,fatjet[0].shape,fatjet[1].shape)
        # sys.exit()
        tot = np.concatenate((fatjet[0], fatjet[1], lepton), axis=1)
        extremum = np.swapaxes(
            np.array([np.min(tot, axis=1) - 0.2, np.max(tot, axis=1) + 0.2]),
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
                        (0, item[0]),
                        (0, item[1]),
                        (0, item[2]),
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
                        (0, item[0]),
                        (0, item[1]),
                        (0, item[2]),
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
                        (0, item[0]),
                        (0, item[1]),
                        (0, item[2]),
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
                        (0, item[0]),
                        (0, item[1]),
                        (0, item[2]),
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
                        (0, item[0]),
                        (0, item[1]),
                        (0, item[2]),
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
                        (0, item[0]),
                        (0, item[1]),
                        (0, item[2]),
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

    def Subjet(self, fatjet, label=None, show=False, r=0.4, no_subjets=3):
        self.FatJet(fatjet, show=False)
        subjets = FatJet().Recluster(fatjet, r=r, subjets=no_subjets)
        plotting_arrays = FatJet().GetConstituents(subjets, format="image")
        cmap, color = ["Blues", "YlGn", "RdPu"], ["b", "g", "r"]
        count = 1
        for array, coordinate in zip(plotting_arrays, subjets):
            # self.axes.scatter(array[0],array[1],c=array[2],cmap=cmap[count-1],s=self.markersize,marker=self.marker)
            self.axes.add_patch(
                Circle(
                    (coordinate.eta, coordinate.phi),
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

    def FatJet(self, fatjet, label=None, show=False, cmap="Blues", s=50):
        # ru.Print(np.sum(fatjet))
        center = [np.sum(fatjet).Eta(), np.sum(fatjet).Phi()]
        if len(fatjet.shape) != 2:
            fatjet = ru.GetNumpy(fatjet)
        extremum = np.swapaxes(
            np.array(
                [np.min(fatjet, axis=1) - 0.2, np.max(fatjet, axis=1) + 0.2]
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
                vmin=np.min(fatjet[2]), vmax=np.max(fatjet[2])
            ),
            label=label,
        )
        self.axes.scatter(
            center[0], center[1], marker="^", s=s, c="r", label="center"
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
        Z = np.array([item.P() for item in fatjet], dtype="float64")
        if len(fatjet.shape) != 2:
            fatjet = ru.GetNumpy(fatjet, format="lorentz")
        print(fatjet.shape)
        if summed == None:
            extremum = np.swapaxes(
                np.array(
                    [
                        np.min(fatjet, axis=1) - 0.2,
                        np.max(fatjet, axis=1) + 0.2,
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
        # self.axes.scatter(fatjet[0],fatjet[1],fatjet[2],c=Z,cmap="Blues",norm=colors.LogNorm(vmin=np.min(Z), vmax=np.max(Z)))
        count = 0
        for item in np.swapaxes(fatjet, 0, 1):
            if count == 0:
                self.axes.add_artist(
                    Arrow3D(
                        (0, item[0]),
                        (0, item[1]),
                        (0, item[2]),
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
                        (0, item[0]),
                        (0, item[1]),
                        (0, item[2]),
                        mutation_scale=20,
                        lw=1,
                        arrowstyle="-|>",
                        color=arrow_color,
                    )
                )
        if summed != None:
            self.axes.add_artist(
                Arrow3D(
                    (0, summed.X()),
                    (0, summed.Y()),
                    (0, summed.Z()),
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
        x_bound = axes.get_xbound()
        axes.plot(
            np.linspace(x_bound[0], x_bound[1], 100),
            np.zeros(100),
            "--k",
            alpha=0.5,
        )
        y_bound = axes.get_ybound()
        axes.plot(
            np.zeros(100),
            np.linspace(y_bound[0], y_bound[1], 100),
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
        pwd = os.getcwd()
        plot_axes = kwargs.get("plot_axes", False)
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
            for i, item in enumerate(self.axes):
                if i in legend_axes and set_legends:
                    item.legend(
                        loc=kwargs.get("legend_loc", "best"),
                        prop={"size": kwargs.get("legend_size", 25)},
                        title=kwargs.get("legend_title"),
                        title_fontsize=kwargs.get("legend_title_fontsize", 30),
                    )
                item.tick_params(
                    axis="both",
                    which="major",
                    labelsize=kwargs.get("tick_label_size", 28),
                )
        else:
            self.axes.tick_params(
                axis="both",
                which="major",
                labelsize=kwargs.get("tick_label_size", 28),
            )
            if set_legends:
                self.axes.legend(
                    loc=kwargs.get("legend_loc", "best"),
                    prop={"size": kwargs.get("legend_size", 25)},
                    title=kwargs.get("legend_title"),
                    title_fontsize=kwargs.get("legend_title_fontsize", 30),
                )
        self.fig.savefig(
            title + "." + extension,
            format=extension,
            dpi=dpi,
            bbox_inches=kwargs.get("bbox_inches", "tight"),
            pad_inches=kwargs.get("pad", 0.4),
        )
        # plt.show(block=False)
        plt.close()
        print("Output plot saved at ", os.getcwd(), title + "." + extension)
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
        # plt.title(title)
        if array.shape[-1] == 1:
            array = np.squeeze(array)
        if log_scale:
            # array=array+np.min(array[ array != 0])
            print(np.min(array[array != 0]), np.max(array))
            # if self.projection=="subplots":
            # im=self.axes.matshow(array,cmap=cmap,origin="lower",norm=colors.LogNorm(vmin=kwargs.get("vmin",1), vmax=kwargs.get("vmax",200)) )
            im = self.axes.matshow(
                array,
                cmap=cmap,
                origin="lower",
                norm=colors.LogNorm(
                    vmin=kwargs.get("vmin", np.min(array[array != 0])),
                    vmax=kwargs.get("vmax", np.max(array)),
                ),
            )
        else:
            im = self.axes.matshow(
                array, cmap=cmap, origin="lower"
            )  # ,norm=colors.LogNorm(vmin=np.min(array), vmax=np.max(array)))
        if self.projection != "subplots" and set_colorbar:
            self.set_colorbar(im)
        if show:
            plt.show()
        return im

    def set_colorbar(
        self,
        im,
        cax=None,
        axes=None,
        ylabel=None,
        ylabelsize=30,
        clim=None,
        **kwargs
    ):
        if cax is None:
            if axes is None:
                divider = make_axes_locatable(self.axes)
                cax = divider.append_axes(
                    "right",
                    size="5%",
                    pad=kwargs.get("pad", 0.05),
                    norm=kwargs.get("norm"),
                )
                cbar = self.fig.colorbar(
                    im, cax=cax, orientation="vertical"
                )  # format='%.0e'
            else:
                cbar = self.fig.colorbar(
                    im, ax=axes, pad=kwargs.get("pad", 0.05)
                )  # format='%.0e'
        else:
            cbar = self.fig.colorbar(
                im, cax=cax, format="%.0e", norm=kwargs.get("norm")
            )
        cbar.ax.tick_params(
            axis="y", which="major", labelsize=kwargs.get("labelsize", 30)
        )
        if clim is not None:
            # cbar.vmin,cbar.vmax=clim[0],clim[1]
            # print (cbar.vmin,cbar.vmax)
            # sys.exit()
            cbar.set_clim(*clim)
        if ylabel is not None:
            cbar.ax.set_ylabel(ylabel, labelpad=0.01, size=ylabelsize)
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
        if "axes" in kwargs:
            axes = kwargs.pop("axes")
        else:
            axes = None
        """
        if type(array[0])==TLorentzVector:
            array=np.array([[item.Eta(),item.Phi(),item.Pt()] for item in array])
            array=np.swapaxes(array,0,1)
        elif type(array[0]) ==TVector3:
            array=np.array([[item.X(),item.Y(),item.Z()] for item in array])
        """
        # if array.shape[0]!=3 or: array=np.swapaxes(array,0,1)
        X, Y, Z = array[0], array[1], array[2]
        if log_scale:
            if np.min(Z) < 0:
                assert abs(np.min(Z)) < 10e-12
                Z = Z + np.min(Z)
            Z = Z + np.min(Z[Z != 0]) * 0.000000001
        if cmap == None:
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
                norm=colors.LogNorm(vmin=np.min(Z), vmax=np.max(Z)),
                label=label,
                **kwargs
            )
        else:
            im = self.axes.scatter(
                X,
                Y,
                marker=marker,
                s=s,
                c=color,
                cmap=cmap,
                label=label,
                **kwargs
            )
        if set_colorbar:
            self.set_colorbar(im, axes=axes)
        if show:
            plt.show()
        return im
