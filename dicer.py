#!/usr/bin/env python3
"""Get some dices and filter them
"""

# ========================================================================
#
# Imports
#
# ========================================================================
import os
import argparse
import numpy as np
import yt
from yt.units import m, s
import time
from datetime import timedelta


# ========================================================================
#
# Function definitions
#
# ========================================================================
def load_pelec_data(fdir):
    """
    Load PeleC data using yt
    """
    print(fdir)
    # Load data
    ds = yt.load(fdir, unit_system="mks")

    return ds


# ========================================================================
def _scalar_dissipation_rate(field, data):
    dxmin = data.index.get_smallest_dx()
    return (
        data["Diff_Z"]
        * (m ** 2 / s)
        * (
            yt.YTArray(np.gradient(data["Z"], dxmin, axis=0) ** 2, "1.0/m**2")
            + yt.YTArray(np.gradient(data["Z"], dxmin, axis=1) ** 2, "1.0/m**2")
            + yt.YTArray(np.gradient(data["Z"], dxmin, axis=2) ** 2, "1.0/m**2")
        )
    )


# ========================================================================
def _progress_dissipation_rate(field, data):
    dxmin = data.index.get_smallest_dx()
    return (
        data["Diff_C"]
        * (m ** 2 / s)
        * (
            yt.YTArray(np.gradient(data["C"], dxmin, axis=0) ** 2, "1.0/m**2")
            + yt.YTArray(np.gradient(data["C"], dxmin, axis=1) ** 2, "1.0/m**2")
            + yt.YTArray(np.gradient(data["C"], dxmin, axis=2) ** 2, "1.0/m**2")
        )
    )


# ========================================================================
#
# Main
#
# ========================================================================
if __name__ == "__main__":

    # Timer
    start = time.time()

    # Parse arguments
    parser = argparse.ArgumentParser(description="A simple dicing tool")
    parser.add_argument(
        "-f",
        "--folder",
        dest="folder",
        help="Folder containing plot files",
        type=str,
        default=".",
    )
    parser.add_argument(
        "-o", "--output", help="Output folder", type=str, default="data"
    )
    parser.add_argument(
        "-z",
        "--zcenters",
        help="Z-centers of dices",
        default=[
            0.0225,
            0.0375,
            0.0525,
            0.065,
            0.0775,
            0.09,
            0.1025,
            0.115,
            0.1275,
            0.14,
            0.1525,
        ],
        type=float,
        nargs="+",
    )
    parser.add_argument(
        "-ht", "--height", help="Height of each dice", default=0.00625, type=float
    )
    parser.add_argument(
        "--extent",
        help="Extent in x and y direction of dice",
        default=[-0.07, 0.07],
        type=float,
        nargs="+",
    )
    args = parser.parse_args()

    # Setup
    fields_load = ["Rho", "Z", "C", "Diff_C", "Diff_Z", "SDR", "PDR", "SRC_PV", "Temp"]
    fields_save = ["Rho", "Z", "C", "SDR", "PDR", "SRC_PV", "Temp"]
    fdir = os.path.abspath(args.folder)
    odir = os.path.abspath(args.output)
    if not os.path.exists(odir):
        os.makedirs(odir)

    # Load the data
    ds = load_pelec_data(fdir)
    max_level = ds.index.max_level
    ref = int(np.product(ds.ref_factors[0:max_level]))
    L = (ds.domain_right_edge - ds.domain_left_edge).d
    N = ds.domain_dimensions * ref
    dxmin = ds.index.get_smallest_dx()
    extents = np.array(
        [
            ds.domain_left_edge.d[0],
            ds.domain_right_edge.d[1],
            ds.domain_left_edge.d[0],
            ds.domain_right_edge.d[1],
        ]
    )
    ds.add_field(
        ("boxlib", "SDR"),
        sampling_type="cell",
        function=_scalar_dissipation_rate,
        units="1.0/s",
    )
    ds.add_field(
        ("boxlib", "PDR"),
        sampling_type="cell",
        function=_progress_dissipation_rate,
        units="1.0/s",
    )
    print(ds.field_list)

    # Take the dices
    for i, zcenter in enumerate(args.zcenters):

        # Take a dice
        print("Taking a dice centered at z = {0:f}".format(zcenter))
        low = np.array([args.extent[0], args.extent[0], zcenter - 0.5 * args.height])
        high = np.array([args.extent[1], args.extent[1], zcenter + 0.5 * args.height])

        dims = (high - low) / dxmin

        dice = ds.covering_grid(
            max_level, left_edge=low, dims=dims.astype(int), fields=fields_load
        )

        print("  dice size: ", dice[fields_load[0]].d.shape)

        # Save the dices
        fname = os.path.join(odir, "dice_{0:04d}".format(i))
        fields_to_save = dict(
            zip(fields_load, [dice[field].d for field in fields_load])
        )
        np.savez_compressed(
            fname, fdir=fdir, z=zcenter, dx=dxmin, low=low, high=high, **fields_to_save
        )

    # output timer
    end = time.time() - start
    print(
        "Elapsed time "
        + str(timedelta(seconds=end))
        + " (or {0:f} seconds)".format(end)
    )
