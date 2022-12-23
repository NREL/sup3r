# -*- coding: utf-8 -*-
"""Utilities module for plotting data
"""

import imageio
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


def pretty_labels(xlabel, ylabel, fontsize=14, title=None):
    """Make pretty labels for plots

    Parameters
    ----------
    xlabel : str
        label for x abscissa
    ylabel : str
        label for y abscissa
    fontsize : int, optional
        size of the plot font, by default 14
    title : str, optional
        plot title, by default None

    """
    plt.xlabel(
        xlabel,
        fontsize=fontsize,
        fontweight="bold",
        fontname="Times New Roman",
    )
    plt.ylabel(
        ylabel,
        fontsize=fontsize,
        fontweight="bold",
        fontname="Times New Roman",
    )
    if title is not None:
        plt.title(
            title,
            fontsize=fontsize,
            fontweight="bold",
            fontname="Times New Roman",
        )
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontname("Times New Roman")
        tick.label1.set_fontweight("bold")
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontname("Times New Roman")
        tick.label1.set_fontweight("bold")
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color("black")
    plt.grid(color="k", linestyle="-", linewidth=0.5)
    plt.tight_layout()


def ax_pretty_labels(ax, xlabel, ylabel, fontsize=14, title=None):
    """Make pretty labels for ax plots

    Parameters
    ----------
    ax : axis handle
        handle for axis that contains the plot
    xlabel : str
        label for x abscissa
    ylabel : str
        label for y abscissa
    fontsize : int, optional
        size of the plot font, by default 14
    title : str, optional
        plot title, by default None

    """
    ax.set_xlabel(
        xlabel,
        fontsize=fontsize,
        fontweight="bold",
        fontname="Times New Roman",
    )
    ax.set_ylabel(
        ylabel,
        fontsize=fontsize,
        fontweight="bold",
        fontname="Times New Roman",
    )
    if title is not None:
        ax.set_title(
            title,
            fontsize=fontsize,
            fontweight="bold",
            fontname="Times New Roman",
        )
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontname("Times New Roman")
        tick.label1.set_fontweight("bold")
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontname("Times New Roman")
        tick.label1.set_fontweight("bold")
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color("black")
    ax.grid(color="k", linestyle="-", linewidth=0.5)
    plt.tight_layout()


def plot_legend(fontsize=16):
    """Make pretty legend

    Parameters
    ----------
    fontsize : int, optional
        size of the plot font, by default 16

    """
    plt.legend()
    leg = plt.legend(
        prop={
            "family": "Times New Roman",
            "size": fontsize - 3,
            "weight": "bold",
        }
    )
    leg.get_frame().set_linewidth(2.0)
    leg.get_frame().set_edgecolor("k")


def ax_plot_legend(ax, fontsize=16):
    """Make pretty legend for ax plots,

    Parameters
    ----------
    ax : axis handle
        handle for axis that contains the plot
    fontsize : int, optional
        size of the plot font, by default 16

    """
    ax.legend()
    leg = ax.legend(
        prop={
            "family": "Times New Roman",
            "size": fontsize - 3,
            "weight": "bold",
        }
    )
    leg.get_frame().set_linewidth(2.0)
    leg.get_frame().set_edgecolor("k")


def make_movie(ntime, movieDir, movieName, fps=24):
    """Make movie from png

    Parameters
    ----------
    ntime : int
        number of snapshots
    movieDir : str
        path to folder containing images to compile into a movie
    movieName : str
        path to movie to generate
    fps : int, optional
        number of frame per second for the movie, by default 24

    """
    # initiate an empty  list of "plotted" images
    myimages = []
    # loops through available pngs
    for i in range(ntime):
        # Read in picture
        fname = movieDir + "/im_" + str(i) + ".png"
        myimages.append(imageio.imread(fname))
    imageio.mimsave(movieName, myimages, fps=fps)


def plot_single_contour(
    data,
    xbound,
    ybound,
    CBLabel='',
    title='',
    xAxisName=None,
    yAxisName=None,
    vmin=None,
    vmax=None,
    suptitle=None
):
    """Plot single contour

    Parameters
    ----------
    data : numpy array
        data to plot, must be 2D
    xbound : list
        min and max bounds of x axis
    ybound : list
        min and max bounds of y axis
    CBLabel : str, optional
        label of color bar, by default empty string
    title : str, optional
        contour title, by default empty string
    xAxisName : str, optional
        x axis label, by default None
    yAxisName : str, optional
        y axis label, by default None
    vmin : float, optional
        min val of the contour, by default None
    vmax : float, optional
        max val of the contour, by default None
    suptitle : str, optional
        global title of the subplots
    """
    fig, axs = plt.subplots(1, 1, figsize=(3, 4))
    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)
    im = axs.imshow(
        data.T,
        cmap=cm.jet,
        interpolation="bicubic",
        vmin=vmin,
        vmax=vmax,
        extent=[xbound[0], xbound[1], ybound[1], ybound[0]],
        origin='lower',
        aspect="auto",
    )
    divider = make_axes_locatable(axs)
    cax = divider.append_axes("right", size="10%", pad=0.2)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(CBLabel)
    ax = cbar.ax
    text = ax.yaxis.label
    font = matplotlib.font_manager.FontProperties(
        family="times new roman", weight="bold", size=14
    )
    text.set_font_properties(font)
    ax_pretty_labels(
        axs,
        xAxisName,
        yAxisName,
        12,
        title,
    )
    ax.set_xticks([])  # values
    ax.set_xticklabels([])  # labels
    for lab in cbar.ax.yaxis.get_ticklabels():
        lab.set_weight("bold")
        lab.set_family("serif")
        lab.set_fontsize(12)
    if suptitle is not None:
        plt.suptitle(suptitle, fontsize=12, fontweight='bold')
        plt.subplots_adjust(top=0.85)
    return fig


def _pick_first_or_none(listArg):
    """Utilitie to select either none or first value

    Parameters
    ----------
    listArg : list
        list that is either None of no

    Returns
    -------
    firstEntry : type of list entry or None
        Either the first entry of the list or None

    """
    firstEntry = None
    if listArg is not None:
        firstEntry = listArg[0]
    return firstEntry


def plot_multi_contour(
    listData,
    xbound,
    ybound,
    listCBLabel,
    listTitle,
    listXAxisName=None,
    listYAxisName=None,
    vminList=None,
    vmaxList=None,
    suptitle=None
):
    """Plot multiple contours as subplots

    Parameters
    ----------
    listData : list
        list of 2D numpy arrays containing data to plot
    xbound : list
        min and max bounds of x axis
    ybound : list
        min and max bounds of y axis
    listCBLabel : list
        list of individual labels of color bar
    listTitle : list
        list of individual contour titles
    listXAxisName : list, optional
        list of individual x axis label, by default None
    listYAxisName : list, optional
        list of individual y axis label, by default None
    vminList : list, optional
        list of individual min val of contour, by default None
    vmaxList : list, optional
        list of individual max val of contour, by default None
    suptitle : str, optional
        global title of the subplots
    """
    fig, axs = plt.subplots(1, len(listData), figsize=(len(listData) * 3, 4))
    if len(listData) == 1:
        plot_single_contour(listData[0], xbound, ybound,
                            listCBLabel[0], listTitle[0],
                            _pick_first_or_none(listXAxisName),
                            _pick_first_or_none(listYAxisName),
                            _pick_first_or_none(vminList),
                            _pick_first_or_none(vmaxList),
                            suptitle)
    else:
        for i_dat, data in enumerate(listData):
            if vminList is None:
                vmin = np.nanmin(data)
            else:
                vmin = vminList[i_dat]
            if vmaxList is None:
                vmax = np.nanmax(data)
            else:
                vmax = vmaxList[i_dat]
            im = axs[i_dat].imshow(
                data.T,
                cmap=cm.jet,
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
                extent=[xbound[0], xbound[1], ybound[1], ybound[0]],
                origin="lower",
                aspect="auto",
            )
            divider = make_axes_locatable(axs[i_dat])
            cax = divider.append_axes("right", size="10%", pad=0.2)
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_label(listCBLabel[i_dat])
            ax = cbar.ax
            text = ax.yaxis.label
            font = matplotlib.font_manager.FontProperties(
                family="times new roman", weight="bold", size=14
            )
            text.set_font_properties(font)
            if i_dat > 0:
                listYAxisName[i_dat] = ""
            ax_pretty_labels(
                axs[i_dat],
                listXAxisName[i_dat],
                listYAxisName[i_dat],
                12,
                listTitle[i_dat],
            )
            axs[i_dat].set_xticks([])  # values
            axs[i_dat].set_xticklabels([])  # labels
            if i_dat != 0:
                axs[i_dat].set_yticks([])  # values
                axs[i_dat].set_yticklabels([])  # labels
            for lab in cbar.ax.yaxis.get_ticklabels():
                lab.set_weight("bold")
                lab.set_family("serif")
                lab.set_fontsize(12)
        if suptitle is not None:
            plt.suptitle(suptitle, fontsize=12, fontweight='bold')
            plt.subplots_adjust(top=0.85)

    return fig
