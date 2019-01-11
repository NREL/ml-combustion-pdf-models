#!/usr/bin/env python3
"""
Generate PDFs from DNS data
"""

# ========================================================================
#
# Imports
#
# ========================================================================
import os
import io
import itertools
import numpy as np
import pandas as pd
from scipy import stats

import utilities


# ========================================================================
#
# Function definitions
#
# ========================================================================
def load_raw_pdf_data(fname):
    """
    Load the data and get a data frame (save it for later)
    """

    # Read bins
    Zbins = np.array([])
    Cbins = np.array([])
    with open(fname, "r") as f:
        next(f)
        for k, line in enumerate(f):
            line = line.split()
            if len(line) == 3:
                Zbin, Cbin, _ = line
                Zbins = np.append(Zbins, np.float(Zbin))
                Cbins = np.append(Cbins, np.float(Cbin))
            else:
                break
    bins = pd.DataFrame({"Zbins": Zbins, "Cbins": Cbins})

    # Read the PDF labels and values
    s = io.StringIO()
    with open(fname, "r") as f:
        label = 0
        for k, line in enumerate(f):
            line = line.split()
            if len(line) == 4:
                Z, Zvar, C, Cvar = line
                label += 1
                print("Processing PDF {0:d}".format(label))
                s.write(
                    "\n"
                    + str(
                        [
                            label,
                            np.float(C),
                            np.float(Cvar),
                            np.float(Z),
                            np.float(Zvar),
                        ]
                    )[1:-1]
                )
                continue
            if len(line) == 3:
                _, _, pdf = line
                s.write("," + str(pdf))

    # Convert to dataframe
    s.seek(0)
    names = ["C", "Cvar", "Z", "Zvar"] + [
        "Y{0:04d}".format(i) for i in range(len(Zbins))
    ]
    df = pd.read_csv(s, index_col=0, names=names)

    # Save these to a file
    df.to_pickle("pdfs.gz")
    bins.to_pickle("bins.gz")
    return df, bins


# ========================================================================
def gen_pdf_from_dice(fname):
    """
    Generate PDFs from a dice of data

    :param fname: dice file name
    :type fname: str
    :return: PDFs
    :rtype: dataframe
    """

    # Load dice file
    dat = np.load(fname)
    lo = dat["low"]
    dx = dat["dx"]

    # Variables
    rho = dat["Rho"]
    Z = np.clip(dat["Z"], 0.0, 1.0)
    C = np.clip(dat["C"], 0.0, None)
    SRC_PV = dat["SRC_PV"]
    rhoZ = rho * Z
    rhoC = rho * C
    rhoSRC_PV = rho * SRC_PV

    # PDF bins
    nc = 32
    nz = 64
    cbin_edges = np.linspace(0, 0.21, nc + 1)
    zbin_edges = np.linspace(0, 1, nz + 1)
    Zbins, Cbins = np.meshgrid(
        utilities.edges_to_midpoint(zbin_edges), utilities.edges_to_midpoint(cbin_edges)
    )
    bins = pd.DataFrame({"Zbins": np.ravel(Zbins), "Cbins": np.ravel(Cbins)})
    bins.to_pickle("bins.gz")

    # Loop on all blocks of width^3 separated by stride
    width = 32
    stride = 8
    N = rho.shape
    ranges = [
        range(0, N[0] - width, stride),
        range(0, N[1] - width, stride),
        range(0, N[2] - width, stride),
    ]

    # PDFs storage
    npdfs = np.prod([len(x) for x in ranges])
    pdfs = np.zeros((npdfs, 8 + nz * nc))
    src_pv_means = np.zeros((npdfs, nz * nc))

    # Loop on all the blocks
    for cnt, (i, j, k) in enumerate(itertools.product(ranges[0], ranges[1], ranges[2])):

        # Get center of block
        bc = [
            lo[0] + (i + width // 2) * dx,
            lo[1] + (j + width // 2) * dx,
            lo[2] + (k + width // 2) * dx,
        ]

        # Favre averages
        block = np.s_[i : i + width, j : j + width, k : k + width]
        rho_ = np.sum(rho[block])
        C_ = np.sum(rhoC[block]) / rho_
        Cvar_ = np.sum(rho[block] * (C[block] - C_) ** 2) / rho_
        Z_ = np.sum(rhoZ[block]) / rho_
        Zvar_ = np.sum(rho[block] * (Z[block] - Z_) ** 2) / rho_
        SRC_PV_ = np.sum(rhoSRC_PV[block]) / rho_

        # Compute density-weighted PDF
        pdf, _, _, _ = stats.binned_statistic_2d(
            np.ravel(Z[block]),
            np.ravel(C[block]),
            np.ravel(rho[block]),
            statistic="sum",
            bins=[zbin_edges, cbin_edges],
        )
        pdf /= rho_

        # Compute SRC_PV conditional means
        means, _, _, _ = stats.binned_statistic_2d(
            np.ravel(Z[block]),
            np.ravel(C[block]),
            np.ravel(SRC_PV[block]),
            statistic="mean",
            bins=[zbin_edges, cbin_edges],
        )
        means[np.isnan(means)] = 0.0

        # Store
        pdfs[cnt, :3] = bc
        pdfs[cnt, 3:8] = [C_, Cvar_, SRC_PV_, Z_, Zvar_]
        pdfs[cnt, 8:] = pdf.flatten(order="F")
        src_pv_means[cnt, :] = means.flatten(order="F")

    # Create dataframes and save for later
    pdfs = pd.DataFrame(
        pdfs,
        columns=["xc", "yc", "zc", "C", "Cvar", "SRC_PV", "Z", "Zvar"]
        + ["Y{0:04d}".format(i) for i in range(nz * nc)],
    )
    pdfs.to_pickle(os.path.splitext(fname)[0] + "_pdfs.gz")

    src_pv_means = pd.DataFrame(
        src_pv_means, columns=["Y{0:04d}".format(i) for i in range(nz * nc)]
    )
    src_pv_means.to_pickle(os.path.splitext(fname)[0] + "_src_pv_means.gz")

    return pdfs, bins, src_pv_means
