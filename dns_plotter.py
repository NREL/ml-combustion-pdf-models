#!/usr/bin/env python3

# ========================================================================
#
# Imports
#
# ========================================================================
import numpy as np
import yt
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.ticker import SymmetricalLogLocator
from matplotlib.backends.backend_pdf import PdfPages


# ========================================================================
#
# Function definitions
#
# ========================================================================
def plot_dns(fdir):

    # Load the data
    ds = yt.load(fdir, unit_system="mks")

    # Setup
    field = "SRC_PV"
    L = (ds.domain_right_edge - ds.domain_left_edge).d
    width = L[0]
    res = 512
    # all dices
    # zlocs = np.array([0.0225, 0.0375] + [0.0525 + i * 0.0125 for i in range(0, 9)])
    # dices that matter
    zlocs = np.array([0.0525 + i * 0.0125 for i in range(0, 9)])
    zlocs = zlocs[::2]

    fname = "src_pv.pdf"
    with PdfPages(fname) as pdf:
        plt.close("all")
        plt.rc("text", usetex=True)
        linthresh = 1e-3

        # Get a slice in x
        slc = yt.SlicePlot(ds, "x", fields=[field])
        frb = slc.data_source.to_frb(width, res)
        x_slc = np.array(frb[field])

        fig0 = plt.figure(0)
        ax0 = fig0.add_subplot(111)
        im = ax0.imshow(
            x_slc,
            origin="lower",
            extent=[
                ds.domain_left_edge.d[0],
                ds.domain_right_edge.d[0],
                ds.domain_left_edge.d[2],
                ds.domain_right_edge.d[2],
            ],
            aspect="equal",
            cmap="inferno",
            norm=colors.SymLogNorm(
                linthresh=linthresh, linscale=0.5, vmin=x_slc.min(), vmax=x_slc.max()
            ),
        )
        cbar = plt.colorbar(
            im, ax=ax0, ticks=SymmetricalLogLocator(linthresh=linthresh, base=10)
        )
        cbar.ax.set_title(r"$\dot{\omega}$")

        for zloc in zlocs:
            ax0.plot(
                [ds.domain_left_edge.d[0], ds.domain_right_edge.d[0]],
                [zloc, zloc],
                color="w",
                lw=1,
                ls="--",
            )

        ax0.set_xlabel(r"$y~[\mathrm{m}]$", fontsize=22, fontweight="bold")
        ax0.set_ylabel(r"$z~[\mathrm{m}]$", fontsize=22, fontweight="bold")
        plt.setp(ax0.get_xmajorticklabels(), fontsize=18)
        plt.setp(ax0.get_ymajorticklabels(), fontsize=18)
        fig0.subplots_adjust(bottom=0.15)
        fig0.subplots_adjust(left=0.17)
        pdf.savefig(dpi=300)

        # Get slices in z
        for k, zloc in enumerate(zlocs):
            slc = yt.SlicePlot(ds, "z", fields=[field], center=[0, 0, zloc])
            frb = slc.data_source.to_frb(width, res)
            z_slc = np.array(frb[field])

            fig0 = plt.figure(k + 1)
            ax0 = fig0.add_subplot(111)
            im = ax0.imshow(
                z_slc,
                origin="lower",
                extent=[
                    ds.domain_left_edge.d[0],
                    ds.domain_right_edge.d[0],
                    ds.domain_left_edge.d[1],
                    ds.domain_right_edge.d[1],
                ],
                aspect="equal",
                cmap="inferno",
                norm=colors.SymLogNorm(
                    linthresh=linthresh,
                    linscale=0.5,
                    vmin=x_slc.min(),
                    vmax=x_slc.max(),
                ),
            )
            cbar = plt.colorbar(
                im, ax=ax0, ticks=SymmetricalLogLocator(linthresh=linthresh, base=10)
            )
            cbar.ax.set_title(r"$\dot{\omega}$")

            ax0.set_xlabel(r"$x~[\mathrm{m}]$", fontsize=22, fontweight="bold")
            ax0.set_ylabel(r"$y~[\mathrm{m}]$", fontsize=22, fontweight="bold")
            plt.setp(ax0.get_xmajorticklabels(), fontsize=18)
            plt.setp(ax0.get_ymajorticklabels(), fontsize=18)
            fig0.subplots_adjust(bottom=0.15)
            fig0.subplots_adjust(left=0.17)
            pdf.savefig(dpi=300)


# ========================================================================
def plot_dns_with_dices(fdir):

    # Load the data
    ds = yt.load(fdir, unit_system="mks")

    # Setup
    field = "SRC_PV"
    L = (ds.domain_right_edge - ds.domain_left_edge).d
    width = L[0]
    res = 512
    zlocs = np.array([0.0525 + i * 0.0125 for i in range(0, 9)])
    xlo = -0.07
    dw = 0.14
    dh = 0.00625

    fname = "src_pv_with_dices.pdf"
    with PdfPages(fname) as pdf:
        plt.close("all")
        plt.rc("text", usetex=True)
        linthresh = 1e-3

        # Get a slice in x
        slc = yt.SlicePlot(ds, "x", fields=[field])
        frb = slc.data_source.to_frb(width, res)
        x_slc = np.array(frb[field])

        fig0 = plt.figure(0)
        ax0 = fig0.add_subplot(111)
        im = ax0.imshow(
            x_slc,
            origin="lower",
            extent=[
                ds.domain_left_edge.d[0],
                ds.domain_right_edge.d[0],
                ds.domain_left_edge.d[2],
                ds.domain_right_edge.d[2],
            ],
            aspect="equal",
            cmap="inferno",
            norm=colors.SymLogNorm(
                linthresh=linthresh, linscale=0.5, vmin=x_slc.min(), vmax=x_slc.max()
            ),
        )

        ax0_divider = make_axes_locatable(ax0)
        cax0 = ax0_divider.append_axes("top", size="7%", pad="2%")
        cb0 = plt.colorbar(
            im,
            cax=cax0,
            ticks=SymmetricalLogLocator(linthresh=linthresh, base=10),
            orientation="horizontal",
        )
        cax0.xaxis.set_ticks_position("top")
        cax0.tick_params(axis="both", which="major", labelsize=6)
        for label in cax0.xaxis.get_ticklabels()[::2]:
            label.set_visible(False)
        cax0.set_title(r"$\dot{\omega}$")

        for i, zloc in enumerate(zlocs):
            zc = zloc - 0.5 * dh
            xhi = xlo + dw
            rect = patches.Rectangle(
                (xlo, zc),
                dw,
                dh,
                linewidth=1,
                edgecolor=(1, 1, 1, 1),
                facecolor=(1, 1, 1, 0.3),
            )
            ax0.add_patch(rect)

            ax0.annotate(
                r"$\mathcal{{V}}_{0:d}$".format(i + 1),
                (xhi + dh, zc),
                color="w",
                fontsize=10,
                ha="center",
                va="center",
            )

        ax0.set_xlabel(r"$y~[\mathrm{m}]$", fontsize=22, fontweight="bold")
        ax0.set_ylabel(r"$z~[\mathrm{m}]$", fontsize=22, fontweight="bold")
        plt.setp(ax0.get_xmajorticklabels(), fontsize=18)
        plt.setp(ax0.get_ymajorticklabels(), fontsize=18)
        fig0.subplots_adjust(bottom=0.15)
        fig0.subplots_adjust(left=0.17)

        pdf.savefig(dpi=300)
