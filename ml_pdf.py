#!/usr/bin/env python3
"""
Machine learning for PDF shapes
"""

# ========================================================================
#
# Imports
#
# ========================================================================
import os
import time
import datetime
import numpy as np
import pickle
import pandas as pd
from scipy import stats
from scipy import signal
from scipy import ndimage
from scipy.stats.kde import gaussian_kde
from scipy.spatial import distance
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from imblearn.over_sampling import RandomOverSampler
import torch
from torch import nn, optim
from torch.autograd import Variable
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

import utilities as utilities
import lrp


# ===============================================================================
#
# Some defaults variables
#
# ===============================================================================
cmap = [
    "#EE2E2F",
    "#008C48",
    "#185AA9",
    "#F47D23",
    "#662C91",
    "#A21D21",
    "#B43894",
    "#010202",
]
dashseq = [
    (None, None),
    [10, 5],
    [10, 4, 3, 4],
    [3, 3],
    [10, 4, 3, 4, 3, 4],
    [3, 3],
    [3, 3],
    [3, 3],
    [3, 3],
]
markertype = ["s", "d", "o", "p", "h"]


# ========================================================================
#
# Function definitions
#
# ========================================================================
def load_dice(fname):
    """
    Load the data and get a training ready data frame
    """
    dat = np.load(fname)

    df = pd.DataFrame(
        {
            "Z": dat["Z"].flatten(),
            "Z4": dat["Z4"].flatten(),
            "Z8": dat["Z8"].flatten(),
            "Z16": dat["Z16"].flatten(),
            "Z32": dat["Z32"].flatten(),
            "C": dat["C"].flatten(),
            "C4": dat["C4"].flatten(),
            "C8": dat["C8"].flatten(),
            "C16": dat["C16"].flatten(),
            "C32": dat["C32"].flatten(),
            "SRC_PV": dat["SRC_PV"].flatten(),
            "rhoSRC_PV": (dat["Rho"] * dat["SRC_PV"]).flatten(),
            "SRC_PV4": dat["SRC_PV4"].flatten(),
            "SRC_PV8": dat["SRC_PV8"].flatten(),
            "SRC_PV16": dat["SRC_PV16"].flatten(),
            "SRC_PV32": dat["SRC_PV32"].flatten(),
            "Zvar4": dat["Zvar4"].flatten(),
            "Zvar8": dat["Zvar8"].flatten(),
            "Zvar16": dat["Zvar16"].flatten(),
            "Zvar32": dat["Zvar32"].flatten(),
            "Cvar4": dat["Cvar4"].flatten(),
            "Cvar8": dat["Cvar8"].flatten(),
            "Cvar16": dat["Cvar16"].flatten(),
            "Cvar32": dat["Cvar32"].flatten(),
        }
    )

    # Clip variables
    df.Z = np.clip(df.Z, 0.0, 1.0)
    df.Z4 = np.clip(df.Z4, 0.0, 1.0)
    df.Z8 = np.clip(df.Z8, 0.0, 1.0)
    df.Z16 = np.clip(df.Z16, 0.0, 1.0)
    df.Z32 = np.clip(df.Z32, 0.0, 1.0)

    df.C = np.clip(df.C, 0.0, None)
    df.C4 = np.clip(df.C4, 0.0, None)
    df.C8 = np.clip(df.C8, 0.0, None)
    df.C16 = np.clip(df.C16, 0.0, None)
    df.C32 = np.clip(df.C32, 0.0, None)

    return dat, df


# ========================================================================
def gen_training(df, oname="training"):
    """
    Generate scaled training, dev, test arrays
    """

    x_vars = get_xnames()
    y_vars = get_ynames(df)
    return utilities.gen_training(df, x_vars, y_vars, oname)


# ========================================================================
def get_xnames():
    return ["C", "Cvar", "Z", "Zvar"]


# ========================================================================
def get_ynames(df):
    return [col for col in df if col.startswith("Y")]


# ========================================================================
def closest_point(point, points):
    """Find index of closest point"""
    closest_index = distance.cdist([point], np.asarray(points)).argmin()
    if isinstance(points, pd.DataFrame):
        return points.iloc[closest_index, :]
    else:
        return points[closest_index, :]


# ========================================================================
def wide_to_narrow(X, Y, bins):
    """
    Convert data from predicting a Y(Zbin,Cbin) as a vector to
    individual predictions of Y(Zbin,Cbin) given a Zbin and Cbin label
    in the input data.
    """
    varname = "variable"
    valname = "Y"
    x_vars = get_xnames()

    dev = pd.concat([X, Y], axis=1)
    left = pd.melt(
        dev.reset_index(),
        id_vars=x_vars + ["index"],
        value_vars=Y.columns,
        var_name=varname,
        value_name=valname,
    )
    right = pd.concat([bins, pd.DataFrame(Y.columns, columns=[varname])], axis=1)
    narrow = pd.merge(left, right, on=[varname]).set_index(["index", varname])

    narrow = narrow.reindex(X.index, level="index")

    return narrow.drop(columns=[valname]), narrow[valname]


# ========================================================================
def narrow_to_wide(Xn, Yn, idx=None):
    """
    Reverse of wide_to_narrow
    """

    varname = "variable"
    valname = "Y"
    x_vars = get_xnames()
    bin_names = ["Zbins", "Cbins"]

    narrow = pd.concat([Xn, Yn], axis=1).drop(columns=bin_names)
    wide = narrow.reset_index().pivot(
        index="index", columns="variable", values=x_vars + [valname]
    )

    # Get X
    X = wide[x_vars].stack().xs("Y0000", level=varname)
    X.index.name = None

    # Get Y
    Y = wide[valname]
    Y.columns.name = None
    Y.index.name = None

    # Sort according to original wide
    if idx is not None:
        X = X.reindex(idx)
        Y = Y.reindex(idx)

    return X, Y


# ========================================================================
def fix_imbalance(df, n_clusters=10):
    """
    Fix an imbalanced data set by over sampling minority classes
    """

    x_vars = get_xnames()
    classes = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(df[x_vars])

    ros = RandomOverSampler(random_state=0)
    X_resampled, classes_resampled = ros.fit_sample(df, classes)
    return pd.DataFrame(X_resampled, columns=df.columns)


# ========================================================================
def deprecated_gen_conditional_means_dice(df, zbin_edges, cbin_edges, oname):
    """
    Generate the conditional means for a dataframe
    """
    means, _, _, nbins = stats.binned_statistic_2d(
        df.Z, df.C, df.rhoSRC_PV, statistic="mean", bins=[zbin_edges, cbin_edges]
    )

    # Plot
    ma_means = np.ma.array(means, mask=np.isnan(means))
    cm = matplotlib.cm.viridis
    cm.set_bad("white", 1.0)
    plt.figure(0)
    plt.clf()
    im = plt.imshow(
        ma_means.T,
        extent=[
            np.min(zbin_edges),
            np.max(zbin_edges),
            np.min(cbin_edges),
            np.max(cbin_edges),
        ],
        origin="lower",
        aspect="auto",
        cmap=cm,
    )
    plt.colorbar(im)
    plt.xlabel("Mixture Fraction")
    plt.ylabel("Progress Variable")
    plt.title("Conditional means")
    plt.tight_layout()
    plt.savefig(oname + ".png", format="png", dpi=300, bbox_inches="tight")

    # Fix nans
    means[np.isnan(means)] = 0.0

    # Save for later
    np.savez_compressed(
        oname + ".npz",
        means=means,
        zbin_edges=zbin_edges,
        cbin_edges=cbin_edges,
        nbins=nbins,
    )

    return means


# ========================================================================
def jensen_shannon_divergence(p, q):
    """
    This will be part of scipy as some point.
    See https://github.com/scipy/scipy/pull/8295
    We use this implementation for now: https://stackoverflow.com/questions/15880133/jensen-shannon-divergence

    :param p: PDF (normalized to 1)
    :type p: array
    :param q: PDF (normalized to 1)
    :type q: array
    """
    eps = 1e-13
    M = np.clip(0.5 * (p + q), eps, None)
    return 0.5 * (stats.entropy(p, M) + stats.entropy(q, M))


# ========================================================================
def jensen_shannon_distance(p, q):
    """
    Jensen-Shannon distance
    :param p: PDF (normalized to 1)
    :type p: array
    :param q: PDF (normalized to 1)
    :type q: array
    """
    return np.sqrt(jensen_shannon_divergence(p, q))


# ========================================================================
def calculate_jsd(y, yp):
    """
    Calculate the JSD metric on each PDF prediction
    """
    y = np.asarray(y, dtype=np.float64)
    yp = np.asarray(yp, dtype=np.float64)
    return np.array(
        [jensen_shannon_divergence(y[i, :], yp[i, :]) for i in range(y.shape[0])]
    )


# ========================================================================
def pdf_distances(base, datadir="data"):
    """
    Compute the minimum JSD between all PDFs in base dice with all the
    other dice PDFs.

    Only do this with PDFs in the dev set as this is an expensive operation.

    :param base: baseline dice
    :type base: str
    :param datadir: data directory
    :type datadir: str
    :return: minimum distances
    :rtype: dataframe
    """

    others = [
        "dice_0002",
        "dice_0003",
        "dice_0004",
        "dice_0005",
        "dice_0006",
        "dice_0007",
        "dice_0008",
        "dice_0009",
        "dice_0010",
    ]
    try:
        others.remove(base)
    except ValueError:
        pass

    # Read the baseline PDFs
    Ydev_base = pd.read_pickle(os.path.join(datadir, f"{base}_ydev.gz"))

    # Compute all the distances and keep the minimum for each baseline sample
    distances = {}
    for k, other in enumerate(others):

        # Get pairwise distance matrix
        Ydev_other = pd.read_pickle(os.path.join(datadir, f"{other}_ydev.gz"))
        d = distance.cdist(Ydev_base, Ydev_other, jensen_shannon_divergence)

        # Find the minimum distance from other to base
        idx = d.argmin(axis=0)

        distances[other] = pd.DataFrame(index=Ydev_other.index)
        distances[other]["r"] = d[idx, np.arange(0, Ydev_other.shape[0])]
        distances[other]["idx"] = Ydev_base.index[idx]

    # Save
    with open(os.path.join(datadir, f"{base}_pdf_distances.pkl"), "wb") as f:
        pickle.dump(distances, f, pickle.HIGHEST_PROTOCOL)

    return distances


# ========================================================================
def clip_normalize(y):
    """
    Clip and normalize (along axis=1)
    """
    y = np.clip(y, 0, 1)
    return y / np.sum(y, axis=1, keepdims=True)


# ========================================================================
def rmse_metric(true, predicted):
    return np.sqrt(mean_squared_error(true, predicted))


# ========================================================================
def error_metrics(true, predicted, verbose=False):
    """
    Compute some error metrics
    """
    rmse = rmse_metric(true, predicted)
    mae = mean_absolute_error(true, predicted)
    r2 = r2_score(true, predicted)

    if verbose:
        print(f"RMSE: {rmse:.3f}")
        print(f"MAE: {mae:.3f}")
        print(f"R2: {r2:.3f}")

    return rmse, mae, r2


# ========================================================================
def src_pv_normalization(
    dices=[
        "dice_0002",
        "dice_0003",
        "dice_0004",
        "dice_0005",
        "dice_0006",
        "dice_0007",
        "dice_0008",
        "dice_0009",
        "dice_0010",
    ],
    datadir="data",
):
    """Compute the normalization constant"""
    src_pv_sum = 0.0
    count = 0
    for dice in dices:
        pdf = pd.read_pickle(os.path.join(datadir, f"{dice}_pdfs.gz"))
        src_pv_sum += np.sum(pdf.SRC_PV ** 2)
        count += pdf.SRC_PV.shape[0]

    return np.sqrt(src_pv_sum / count)


# ========================================================================
def convolution_means(pdf, means):
    """
    Perform the PDF convolution given means

    means can be one for each PDF or means_dice.flatten(order='F')

    :param pdf: predictions from model (model.predict(X))
    :type pdf: array
    :param means: conditional means
    :type means: array
    """

    return np.sum(means * pdf, axis=1)


# ========================================================================
def create_logdir(model_name):
    """Create a log directory for a model"""
    time = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
    logdir = os.path.abspath(os.path.join("runs", f"{time}_{model_name}"))
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    return logdir


# ========================================================================
def lr_training(Xtrain, Xdev, Ytrain, Ydev):
    """
    Train using a Linear Regression
    """
    model_name = "LR"
    logdir = create_logdir(model_name)

    # Training
    LR = LinearRegression().fit(Xtrain, Ytrain)
    joblib.dump(LR, os.path.join(logdir, model_name + ".pkl"))
    mtrain = clip_normalize(LR.predict(Xtrain))
    mdev = clip_normalize(LR.predict(Xdev))

    # Summarize training
    summarize_training(
        Ytrain, mtrain, Ydev, mdev, os.path.join(logdir, model_name + ".log")
    )

    return mtrain, mdev, LR


# ========================================================================
def br_training(Xtrain, Xdev, Ytrain, Ydev):
    """
    Train using a Bayesian ridge regression
    """
    model_name = "BR"
    logdir = create_logdir(model_name)

    # Training
    BR = MultiOutputRegressor(BayesianRidge()).fit(Xtrain, Ytrain)
    joblib.dump(BR, os.path.join(logdir, model_name + ".pkl"))
    mtrain = BR.predict(Xtrain)
    mdev = BR.predict(Xdev)

    # Summarize training
    summarize_training(
        Ytrain, mtrain, Ydev, mdev, os.path.join(logdir, model_name + ".log")
    )

    return mtrain, mdev, BR


# ========================================================================
def pr_training(Xtrain, Xdev, Ytrain, Ydev, order=6):
    """
    Train using a polynomial regression
    """
    model_name = f"PR{order}"
    logdir = create_logdir(model_name)

    # Training
    PR = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=order)),
            ("linear", LinearRegression(fit_intercept=False)),
        ]
    )
    PR = PR.fit(Xtrain, Ytrain)
    joblib.dump(PR, os.path.join(logdir, model_name + ".pkl"))
    mtrain = clip_normalize(PR.predict(Xtrain))
    mdev = clip_normalize(PR.predict(Xdev))

    # Summarize training
    summarize_training(
        Ytrain, mtrain, Ydev, mdev, os.path.join(logdir, model_name + ".log")
    )

    return mtrain, mdev, PR


# ========================================================================
def svr_training(Xtrain, Xdev, Ytrain, Ydev):
    """
    Train using a support vector regression
    """
    model_name = "SVR"
    logdir = create_logdir(model_name)

    # Training
    svr = MultiOutputRegressor(SVR(kernel="rbf", epsilon=1e-3))
    grid_param_svr = {
        "estimator__C": [1e0, 1e1, 1e2, 1e3],
        "estimator__gamma": np.logspace(-2, 2, 5),
    }
    SR = GridSearchCV(estimator=svr, param_grid=grid_param_svr, cv=5, n_jobs=-1).fit(
        Xtrain, Ytrain
    )
    print("Best estimator and parameter set found on training set:")
    print(SR.best_estimator_)
    print(SR.best_params_)

    joblib.dump(SR, os.path.join(logdir, model_name + ".pkl"))
    mtrain = SR.predict(Xtrain)
    mdev = SR.predict(Xdev)

    # Summarize training
    summarize_training(
        Ytrain, mtrain, Ydev, mdev, os.path.join(logdir, model_name + ".log")
    )

    return mtrain, mdev, SR


# ========================================================================
def gp_training(Xtrain, Xdev, Ytrain, Ydev):
    """
    Train using a gaussian process regression
    """
    model_name = "GP"
    logdir = create_logdir(model_name)

    # Training
    kernel = 6.2 ** 2 * Matern(
        length_scale=[1, 1, 1, 1], length_scale_bounds=(1e-1, 1e4), nu=1.5
    ) + WhiteKernel(noise_level=2, noise_level_bounds=(1e-1, 3e0))
    GP = GaussianProcessRegressor(
        kernel=kernel, alpha=0, n_restarts_optimizer=3, normalize_y=True
    ).fit(Xtrain, Ytrain)
    print("Trained GP kernel:", GP.kernel_)
    joblib.dump(GP, os.path.join(logdir, model_name + ".pkl"))
    mtrain = GP.predict(Xtrain)
    mdev = GP.predict(Xdev)

    # Summarize training
    summarize_training(
        Ytrain, mtrain, Ydev, mdev, os.path.join(logdir, model_name + ".log")
    )

    return mtrain, mdev, GP


# ========================================================================
def count_rf_parameters(model):
    return np.sum([t.tree_.node_count for t in model.estimators_])


# ========================================================================
def rf_training(Xtrain, Xdev, Ytrain, Ydev, nestim=100, max_depth=30):
    """
    Train using a Random Forest Regression
    """

    # Setup
    model_name = "RF"
    logdir = create_logdir(model_name)
    np.random.seed(985_721)

    # Training
    start = time.time()
    RF = RandomForestRegressor(n_estimators=nestim, max_depth=max_depth, n_jobs=1).fit(
        Xtrain, Ytrain
    )
    end = time.time() - start

    joblib.dump(RF, os.path.join(logdir, model_name + ".pkl"))
    print("Trained RandomForest")
    print("  Feature importance", RF.feature_importances_)
    mtrain = RF.predict(Xtrain)
    mdev = RF.predict(Xdev)

    # Summarize training
    summarize_training(
        Ytrain,
        mtrain,
        Ydev,
        mdev,
        fname=os.path.join(logdir, model_name + ".log"),
        timing=end,
        dofs=count_rf_parameters(RF),
    )

    return mtrain, mdev, RF


# ========================================================================
def betaPDF(mean, var, centers, eps=1e-6):
    """
    Calculate beta PDF

    :param mean: mean
    :type mean: float
    :param var: variance
    :type var: float
    :param centers: bin centers
    :type centers: array
    :param eps: smallness threshold
    :type eps: float
    :return: pdf
    :rtype: array
    """

    pdf = np.zeros(centers.shape)

    if var < eps:

        if mean > np.max(centers):
            pdf[-1] = 1.0
            return pdf

        else:
            idx = np.argmax(centers > mean)
            if (idx == 0) or (idx == len(pdf) - 1):
                pdf[idx] = 1.0
                return pdf
            else:
                pdf[idx - 1] = (centers[idx] - mean) / (centers[idx] - centers[idx - 1])
                pdf[idx] = (mean - centers[idx - 1]) / (centers[idx] - centers[idx - 1])
                return pdf

    elif var > mean * (1.0 - mean):
        pdf[0] = 1.0 - mean
        pdf[-1] = mean
        return pdf

    else:
        a = mean * (mean * (1.0 - mean) / var - 1.0)
        b = a / mean - a
        ni = 1024
        x = np.linspace(0, 1, ni)
        pdf = np.interp(centers, x, stats.beta.pdf(x, a, b))
        pdf /= np.sum(pdf)

    return pdf


# ========================================================================
class AnalyticalPDFModel:
    """Generic analytical PDF model"""

    def __init__(self, zbin_edges, cbin_edges):
        """
        :param zbin_edges: bin edges for Z
        :type bins: array
        :param cbin_edges: bin edges for C
        :type bins: array
        """
        self.zbin_edges = zbin_edges
        self.cbin_edges = cbin_edges
        self.eps = 1e-13
        self.cscale = cbin_edges[-1]
        self.nc = len(cbin_edges) - 1
        self.nz = len(zbin_edges) - 1
        self.cbin_centers = utilities.edges_to_midpoint(cbin_edges)
        self.zbin_centers = utilities.edges_to_midpoint(zbin_edges)
        self.cbin_widths = np.diff(cbin_edges)
        self.zbin_widths = np.diff(zbin_edges)
        self.seed = 9_023_457


# ========================================================================
class DD(AnalyticalPDFModel):
    """
    delta(Z) - delta(C) PDF
    """

    def __init__(self, zbin_edges, cbin_edges):
        super().__init__(zbin_edges, cbin_edges)

    def predict(self, X):
        """
        :param X: conditional variables
        :type X: dataframe
        :return: PDFs
        :rtype: array
        """
        # Get indexes for the bins
        self.zbin_edges[-1] += self.eps
        self.cbin_edges[-1] += self.eps
        idx_z = np.digitize(X.Z, self.zbin_edges)
        idx_c = np.digitize(X.C, self.cbin_edges)

        # Generate delta PDFs
        return np.array(
            [
                signal.unit_impulse(
                    (self.nz, self.nc), (idx_z[i] - 1, idx_c[i] - 1)
                ).flatten(order="F")
                for i in range(X.shape[0])
            ]
        )


# ========================================================================
class BD(AnalyticalPDFModel):
    """
    beta(Z) - delta(C) PDF
    """

    def __init__(self, zbin_edges, cbin_edges):
        super().__init__(zbin_edges, cbin_edges)

    def predict(self, X):
        """
        :param X: conditional variables
        :type X: dataframe
        :return: PDFs
        :rtype: array
        """
        self.cbin_edges[-1] += self.eps
        idx_c = np.digitize(X.C, self.cbin_edges)

        # Generate beta-delta PDFs
        npdfs = X.shape[0]
        pdfs = np.zeros((X.shape[0], self.nz * self.nc))
        np.random.seed(self.seed)
        for i in range(npdfs):
            c_pdf = signal.unit_impulse(self.nc, idx_c[i] - 1)
            z_pdf = betaPDF(X.Z.iloc[i], X.Zvar.iloc[i], self.zbin_centers)
            pdfs[i, :] = np.outer(z_pdf, c_pdf).flatten(order="F")

        return pdfs


# ========================================================================
class BB(AnalyticalPDFModel):
    """
    beta(Z) - beta(C) PDF
    """

    def __init__(self, zbin_edges, cbin_edges):
        super().__init__(zbin_edges, cbin_edges)

    def predict(self, X):
        """
        :param X: conditional variables
        :type X: dataframe
        :return: PDFs
        :rtype: array
        """

        # Generate beta-delta PDFs
        npdfs = X.shape[0]
        pdfs = np.zeros((X.shape[0], self.nz * self.nc))
        np.random.seed(self.seed)
        for i in range(npdfs):
            c_pdf = betaPDF(
                X.C.iloc[i] / self.cscale,
                X.Cvar.iloc[i] / (self.cscale ** 2),
                self.cbin_centers / self.cscale,
            )
            z_pdf = betaPDF(X.Z.iloc[i], X.Zvar.iloc[i], self.zbin_centers)
            pdfs[i, :] = np.outer(z_pdf, c_pdf).flatten(order="F")
        return pdfs


# ========================================================================
# Torch Variable handler
class VariableHandler:
    def __init__(self, device=torch.device("cpu"), dtype=torch.float):
        self.device = device
        self.dtype = dtype

    def tovar(self, input):
        return Variable(torch.as_tensor(input, dtype=self.dtype, device=self.device))


# ========================================================================
# Network Architecture from infoGAN (https://arxiv.org/abs/1606.03657)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.input_height = 32
        self.input_width = 64
        self.input_dim = 1 + 4
        self.output_dim = 1

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_height // 4) * (self.input_width // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )

    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = self.conv(x)
        x = x.view(-1, 128 * (self.input_height // 4) * (self.input_width // 4))
        x = self.fc(x)
        return x

    def load(self, fname):
        """Load pickle file containing model"""
        self.load_state_dict(
            torch.load(fname, map_location=lambda storage, loc: storage)
        )
        self.eval()


# ========================================================================
class SoftmaxImage(nn.Module):
    """Apply Softmax on an image.

    Softmax2d applies on second dimension (i.e. channels), which is
    not what I want. This applies along the H and W dimensions, where
    (N, C, H, W) is the size of the input.

    """

    def __init__(self, channels, height, width):
        super(SoftmaxImage, self).__init__()
        self.channels = channels
        self.height = height
        self.width = width
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        x = x.view(-1, self.channels, self.height * self.width)
        x = self.softmax(x)
        x = x.view(-1, self.channels, self.height, self.width)
        return x


# ========================================================================
# Network Architecture from infoGAN (https://arxiv.org/abs/1606.03657)
class Generator(nn.Module):
    def __init__(self, noise_size, vh=None):
        super(Generator, self).__init__()
        self.input_height = 32
        self.input_width = 64
        self.noise_size = noise_size
        self.input_dim = noise_size + 4
        self.output_dim = 1
        if vh is None:
            self.vh = VariableHandler()
        else:
            self.vh = vh

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_height // 4) * (self.input_width // 4)),
            nn.BatchNorm1d(128 * (self.input_height // 4) * (self.input_width // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            SoftmaxImage(1, self.input_height, self.input_width),
        )

    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = self.fc(x)
        x = x.view(-1, 128, (self.input_height // 4), (self.input_width // 4))
        x = self.deconv(x)

        return x

    def inference(self, x):
        noise = self.vh.tovar(torch.rand(x.shape[0], self.noise_size))
        return self.forward(noise, x)

    def predict(self, X, batch_size=64, nestim=1):
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        meval = np.zeros((n, self.input_height * self.input_width))
        for batch, i in enumerate(range(0, n, batch_size)):
            slc = np.s_[i : i + batch_size, :]
            xsub = self.vh.tovar(X[slc])
            meval[slc] = np.mean(
                [
                    self.inference(xsub).cpu().data.numpy().reshape(xsub.shape[0], -1)
                    for j in range(nestim)
                ]
            )
        return meval

    def load(self, fname):
        """Load pickle file containing model"""
        self.load_state_dict(
            torch.load(fname, map_location=lambda storage, loc: storage)
        )
        self.eval()


# ========================================================================
def cgan_training(Xtrain, Xdev, Ytrain, Ydev, use_gpu=False):
    """
    Train using a conditional GAN
    """

    if use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    dtype = torch.double
    vh = VariableHandler(device=device, dtype=dtype)

    # Make sure inputs are numpy arrays
    Xtrain = np.asarray(Xtrain, dtype=np.float64)
    Ytrain = np.asarray(Ytrain, dtype=np.float64)
    Xdev = np.asarray(Xdev, dtype=np.float64)
    Ydev = np.asarray(Ydev, dtype=np.float64)

    # Sizes
    batch_size = 64
    input_height = 32
    input_width = 64
    nsample_lbls = 16
    nsample_noise = 10
    noise_size = 100
    nlabels = Xtrain.shape[1]
    torch.manual_seed(5_465_462)

    # Construct the G and D models
    D = Discriminator().to(device=device, dtype=dtype)
    G = Generator(noise_size, vh).to(device=device, dtype=dtype)

    # The number of times entire dataset is trained
    nepochs = 500

    # Learning rate
    lr_D = 1e-3
    lr_G = 1e-3
    decay_rate = 0.98

    # Loss and optimizers
    criterion = nn.BCELoss().to(device=device)
    D_optimizer = optim.SGD(D.parameters(), lr=lr_D, momentum=0.5, nesterov=True)
    G_optimizer = optim.SGD(G.parameters(), lr=lr_G, momentum=0.5, nesterov=True)
    D_scheduler = optim.lr_scheduler.StepLR(D_optimizer, step_size=1, gamma=decay_rate)
    G_scheduler = optim.lr_scheduler.StepLR(G_optimizer, step_size=1, gamma=decay_rate)

    # Tensorboard writer
    writer = SummaryWriter()
    logdir = writer.file_writer.get_logdir()
    model_name = "CGAN"

    # Validation images, labels and noise
    xdev_sub = vh.tovar(Xdev[:nsample_lbls, :])
    ydev_sub = vh.tovar(Ydev[:nsample_lbls, :])
    valimgs = ydev_sub.view(nsample_lbls, -1, input_height, input_width)
    vallbl = xdev_sub.expand(input_height, input_width, nsample_lbls, nlabels).permute(
        2, 3, 0, 1
    )
    grid = vutils.make_grid(valimgs, nrow=nsample_lbls, normalize=True, scale_each=True)
    writer.add_image("True PDF", grid, 0)
    fixed_noise = vh.tovar(
        torch.rand(nsample_noise, noise_size)
        .to(device=device)
        .repeat(1, nsample_lbls)
        .reshape(-1, noise_size)
    )
    fixed_labels = xdev_sub.repeat(nsample_noise, 1)

    # Graphs in Tensorboard
    xdummy = vh.tovar(torch.rand(1, 1, input_height, input_width))
    ldummy = vh.tovar(torch.rand(1, nlabels, input_height, input_width))
    writer.add_graph(D, (xdummy, ldummy), verbose=False)
    writer.add_graph(G, (fixed_noise, fixed_labels), verbose=False)

    # Train the model
    nbatches = Xtrain.shape[0] // batch_size
    D.train()
    for epoch in range(nepochs):
        G.train()
        permutation = torch.randperm(Xtrain.shape[0])
        for batch, i in enumerate(range(0, Xtrain.shape[0], batch_size)):

            # Global step
            step = epoch * nbatches + batch

            # Take a batch
            indices = permutation[i : i + batch_size]
            batch_x = vh.tovar(Xtrain[indices, :])
            batch_y = vh.tovar(Ytrain[indices, :])

            # Reshape these for the D network
            actual_batch_size = batch_x.shape[0]
            labels = batch_x.expand(
                input_height, input_width, actual_batch_size, nlabels
            ).permute(2, 3, 0, 1)
            imgs = batch_y.view(actual_batch_size, -1, input_height, input_width)
            noise = vh.tovar(torch.rand((actual_batch_size, noise_size)))

            # Real and fake labels
            real_label = vh.tovar(torch.ones(actual_batch_size, 1))
            fake_label = vh.tovar(torch.zeros(actual_batch_size, 1))

            # update the D network
            D_optimizer.zero_grad()

            D_real = D(imgs, labels)
            D_real_loss = criterion(D_real, real_label)

            G_ = G(noise, batch_x)
            D_fake = D(G_, labels)
            D_fake_loss = criterion(D_fake, fake_label)

            D_loss = D_real_loss + D_fake_loss
            writer.add_scalar("D_real_loss", D_real_loss.item(), step)
            writer.add_scalar("D_fake_loss", D_fake_loss.item(), step)
            writer.add_scalar("D_loss", D_loss.item(), step)

            D_loss.backward()
            D_optimizer.step()

            # update G network
            G_optimizer.zero_grad()

            G_ = G(noise, batch_x)
            D_fake = D(G_, labels)
            G_loss = criterion(D_fake, real_label)
            writer.add_scalar("G_loss", G_loss.item(), step)

            G_loss.backward()
            G_optimizer.step()

            if batch % 10 == 0:

                print(
                    "Epoch [{0:d}/{1:d}], Batch [{2:d}/{3:d}], D_loss: {4:.4e}, G_loss: {5:.4e}".format(
                        epoch + 1,
                        nepochs,
                        batch + 1,
                        nbatches,
                        D_loss.item(),
                        G_loss.item(),
                    )
                )

        # Adaptive time step
        G_scheduler.step()
        D_scheduler.step()
        for param_group in D_optimizer.param_groups:
            print("Current learning rate for discriminator:", param_group["lr"])
        for param_group in G_optimizer.param_groups:
            print("                      for generator:", param_group["lr"])

        # Visualize results in Tensorboard
        G.eval()
        samples = G(fixed_noise, fixed_labels)
        grid = vutils.make_grid(
            samples, nrow=nsample_lbls, normalize=True, scale_each=True
        )
        writer.add_image("Generator", grid, step)

        # Save the models
        torch.save(G.state_dict(), os.path.join(logdir, model_name + "_G.pkl"))
        torch.save(D.state_dict(), os.path.join(logdir, model_name + "_D.pkl"))

    writer.close()

    # Stuff we need to do to get plots...
    G.eval()
    mtrain = G.predict(Xtrain)
    mdev = G.predict(Xdev)

    # Summarize training
    summarize_training(
        Ytrain, mtrain, Ydev, mdev, os.path.join(logdir, model_name + ".log")
    )

    return mtrain, mdev, G


# ========================================================================
# Conditional variational autoencoder
# CVAE paper: Learning Structured Output Representation using Deep Conditional Generative Models
# https://papers.nips.cc/paper/5775-learning-structured-output-representation-using-deep-conditional-generative-models
# code adapted from https://github.com/timbmg/VAE-CVAE-MNIST/blob/master/models.py
class CVAE(nn.Module):
    def __init__(
        self, encoder_layer_sizes, latent_size, decoder_layer_sizes, nlabels=0, vh=None
    ):
        super(CVAE, self).__init__()
        self.latent_size = latent_size
        self.decoder_layer_sizes = decoder_layer_sizes
        if vh is None:
            self.vh = VariableHandler()
        else:
            self.vh = vh

        self.encoder = Encoder(encoder_layer_sizes, latent_size, nlabels).to(
            device=vh.device, dtype=vh.dtype
        )
        self.decoder = Decoder(decoder_layer_sizes, latent_size, nlabels).to(
            device=vh.device, dtype=vh.dtype
        )

    def forward(self, x, c):

        batch_size = x.size(0)

        means, log_var = self.encoder(x, c)

        std = torch.exp(0.5 * log_var)
        eps = self.vh.tovar(torch.randn([batch_size, self.latent_size]))
        z = eps * std + means

        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z

    def inference(self, c):
        z = self.vh.tovar(torch.randn(c.shape[0], self.latent_size))
        recon_x = self.decoder(z, c)
        return recon_x

    def predict(self, X, batch_size=64, nestim=1):
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        meval = np.zeros((n, self.decoder_layer_sizes[-1]))
        for batch, i in enumerate(range(0, n, batch_size)):
            slc = np.s_[i : i + batch_size, :]
            c = self.vh.tovar(X[slc])
            meval[slc] = np.mean(
                [self.inference(c).cpu().data.numpy() for j in range(nestim)], axis=0
            )
        return meval

    def load(self, fname):
        """Load pickle file containing model"""
        self.load_state_dict(
            torch.load(fname, map_location=lambda storage, loc: storage)
        )
        self.eval()


class Encoder(nn.Module):
    def __init__(self, layer_sizes, latent_size, nlabels):

        super(Encoder, self).__init__()

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(name="L%i" % (i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A%i" % (i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c):

        x = torch.cat((x, c), dim=-1)
        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):
    def __init__(self, layer_sizes, latent_size, nlabels):

        super(Decoder, self).__init__()

        self.MLP = nn.Sequential()

        input_size = latent_size + nlabels

        for i, (in_size, out_size) in enumerate(
            zip([input_size] + layer_sizes[:-1], layer_sizes)
        ):
            self.MLP.add_module(name="L%i" % (i), module=nn.Linear(in_size, out_size))
            if i + 1 < len(layer_sizes):
                self.MLP.add_module(name="A%i" % (i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="softmax", module=nn.Softmax(dim=1))

    def forward(self, z, c):
        z = torch.cat((z, c), dim=-1)
        x = self.MLP(z)
        return x


def loss_fn(recon_x, x, mean, log_var):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction="sum")

    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return BCE + KLD


def cvae_training(Xtrain, Xdev, Ytrain, Ydev, use_gpu=False):
    """
    Train using a conditional VAE
    """

    if use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    vh = VariableHandler(device=device, dtype=torch.double)

    # Make sure inputs are numpy arrays
    Xtrain = np.asarray(Xtrain, dtype=np.float64)
    Ytrain = np.asarray(Ytrain, dtype=np.float64)
    Xdev = np.asarray(Xdev, dtype=np.float64)
    Ydev = np.asarray(Ydev, dtype=np.float64)

    # Sizes
    nlabels = Xtrain.shape[1]
    input_size = Ytrain.shape[1]
    batch_size = 64
    encoder_layer_sizes = [input_size + nlabels, 512, 256]
    latent_size = 10
    decoder_layer_sizes = [256, 512, input_size]
    torch.manual_seed(5_465_462)

    # The number of times entire dataset is trained
    nepochs = 500

    # Learning rate
    lr = 1e-3

    # CVAE model
    cvae = CVAE(
        encoder_layer_sizes=encoder_layer_sizes,
        latent_size=latent_size,
        decoder_layer_sizes=decoder_layer_sizes,
        nlabels=nlabels,
        vh=vh,
    ).to(device=device)

    # Optimizer
    optimizer = optim.Adam(cvae.parameters(), lr=lr)

    # Tensorboard writer
    writer = SummaryWriter()
    logdir = writer.file_writer.get_logdir()
    model_name = "CVAE"

    # Graphs in Tensorboard
    xdummy = vh.tovar(torch.rand(1, input_size))
    ldummy = vh.tovar(torch.rand(1, nlabels))
    writer.add_graph(cvae, (xdummy, ldummy), verbose=False)

    # Train the model
    nbatches = Xtrain.shape[0] // batch_size
    start = time.time()
    for epoch in range(nepochs):
        cvae.train()
        permutation = torch.randperm(Xtrain.shape[0])
        for batch, i in enumerate(range(0, Xtrain.shape[0], batch_size)):

            # Global step
            step = epoch * nbatches + batch

            # Take a batch
            indices = permutation[i : i + batch_size]
            batch_c = vh.tovar(Xtrain[indices, :])
            batch_x = vh.tovar(Ytrain[indices, :])

            # Forward model
            recon_x, mean, log_var, z = cvae(batch_x, batch_c)

            # Loss
            loss = loss_fn(recon_x, batch_x, mean, log_var)
            writer.add_scalar("loss", loss, step)
            if batch % 10 == 0:

                print(
                    "Epoch [{0:d}/{1:d}], Batch [{2:d}/{3:d}], Loss: {4:.4e}".format(
                        epoch + 1, nepochs, batch + 1, nbatches, loss
                    )
                )

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Save the models
        torch.save(cvae.state_dict(), os.path.join(logdir, model_name + ".pkl"))

    end = time.time() - start
    writer.close()

    cvae.eval()
    mtrain = cvae.predict(Xtrain)
    mdev = cvae.predict(Xdev)

    # Summarize training
    summarize_training(
        Ytrain,
        mtrain,
        Ydev,
        mdev,
        fname=os.path.join(logdir, model_name + ".log"),
        timing=end,
        dofs=count_parameters(cvae),
    )

    return mtrain, mdev, cvae


# ========================================================================
# Fully connected NN
class Net(nn.Module):
    def __init__(self, input_size, layer_sizes, vh=None):
        super(Net, self).__init__()

        if vh is None:
            self.vh = VariableHandler()
        else:
            self.vh = vh

        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.MLP = nn.Sequential()
        for i, (in_size, out_size) in enumerate(
            zip([input_size] + layer_sizes[:-1], layer_sizes)
        ):
            if i + 1 < len(layer_sizes):
                self.MLP.add_module(
                    name="L%i" % (i), module=nn.Linear(in_size, out_size)
                )
                self.MLP.add_module(name="A%i" % (i), module=nn.LeakyReLU())
                self.MLP.add_module(name="B%i" % (i), module=nn.BatchNorm1d(out_size))
            else:
                self.MLP.add_module(
                    name="L%i" % (i), module=nn.Linear(in_size, out_size)
                )
                self.MLP.add_module(name="softmax", module=nn.Softmax(dim=1))

    def forward(self, x):
        return self.MLP(x)

    def predict(self, X, batch_size=64):
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        meval = np.zeros((n, self.layer_sizes[-1]))
        for batch, i in enumerate(range(0, n, batch_size)):
            slc = np.s_[i : i + batch_size, :]
            meval[slc] = self.forward(self.vh.tovar(X[slc])).cpu().data.numpy()
        return meval

    def load(self, fname):
        """Load pickle file containing model"""
        self.load_state_dict(
            torch.load(fname, map_location=lambda storage, loc: storage)
        )
        self.eval()


# ========================================================================
# Clip to [0,1] and normalize (because we are predicting a PDF)
class ClampNorm(nn.Module):
    def __init__(self):
        super(ClampNorm, self).__init__()

    def forward(self, x):
        out = x.clamp(0.0, 1.0)
        return out / out.sum(1, keepdim=True)


# ========================================================================
# Linear regression
class LinearRegNet(nn.Module):
    def __init__(self, D_in, D_out):
        super(LinearRegNet, self).__init__()
        self.fc1 = nn.Linear(D_in, D_out)

    def forward(self, x):
        out = self.fc1(x)
        return out


# ========================================================================
class RelErrorLoss(nn.Module):
    def __init__(self):
        super(RelErrorLoss, self).__init__()
        self.eps = 1e-6

    def forward(self, input, target):
        return torch.mean(torch.abs(target - input) / (target + self.eps))


# ========================================================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ========================================================================
def dnn_training(Xtrain, Xdev, Ytrain, Ydev, use_gpu=False):
    """
    Train using a deep neural network
    """

    if use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    dtype = torch.double
    vh = VariableHandler(device=device, dtype=dtype)

    # Make sure inputs are numpy arrays
    Xtrain = np.asarray(Xtrain, dtype=np.float64)
    Ytrain = np.asarray(Ytrain, dtype=np.float64)
    Xdev = np.asarray(Xdev, dtype=np.float64)
    Ydev = np.asarray(Ydev, dtype=np.float64)

    # N is batch size; D_in is input dimension; D_out is output dimension
    batch_size = 64
    input_size = Xtrain.shape[1]
    layer_sizes = [256, 512, Ytrain.shape[1]]
    torch.manual_seed(5_465_462)

    # Construct the NN model
    model = Net(input_size, layer_sizes, vh).to(device=device, dtype=dtype)

    # The number of times entire dataset is trained
    nepochs = 500

    # Learning rate
    learning_rate = 1e-4

    # Loss and optimizer
    # criterion = nn.BCELoss().to(device=device)
    criterion = nn.MSELoss().to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Tensorboard output
    writer = SummaryWriter()
    logdir = writer.file_writer.get_logdir()
    model_name = "DNN"
    xdummy = vh.tovar(torch.randn(1, Xtrain.shape[1]))
    writer.add_graph(model, (xdummy,), verbose=True)

    # Train the model
    nbatches = Xtrain.shape[0] // batch_size
    start = time.time()
    for epoch in range(nepochs):

        model.train()
        permutation = torch.randperm(Xtrain.shape[0])

        for batch, i in enumerate(range(0, Xtrain.shape[0], batch_size)):

            # Global step
            step = epoch * nbatches + batch

            # Take a batch
            indices = permutation[i : i + batch_size]
            batch_x = vh.tovar(Xtrain[indices, :])
            batch_y = vh.tovar(Ytrain[indices, :])

            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(batch_x)

            # Compute and log information
            loss = criterion(y_pred, batch_y)
            writer.add_scalar("loss", loss.item(), step)
            if batch % 10 == 0:

                print(
                    "Epoch [{0:d}/{1:d}], Batch [{2:d}/{3:d}], Loss: {4:.4e}".format(
                        epoch + 1, nepochs, batch + 1, nbatches, loss.item()
                    )
                )

                # # Logging to tensorboardX
                # writer.add_text("Text", "text logged at step:" + str(step), step)
                # for name, param in model.named_parameters():
                #     writer.add_histogram(name, param.clone().cpu().data.numpy(), step)

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation loss and adaptive time step
        model.eval()
        val_loss = criterion(model(vh.tovar(Xdev)), vh.tovar(Ydev))
        writer.add_scalar("val_loss", val_loss.item(), step)
        print(
            "Epoch [{0:d}/{1:d}], Validation loss: {2:.4e}".format(
                epoch + 1, nepochs, val_loss.item()
            )
        )
        for param_group in optimizer.param_groups:
            print("Current learning rate", param_group["lr"])

        # Save the models
        torch.save(model.state_dict(), os.path.join(logdir, model_name + ".pkl"))

    end = time.time() - start
    writer.close()

    model.eval()
    mtrain = model.predict(Xtrain)
    mdev = model.predict(Xdev)

    # Summarize training
    summarize_training(
        Ytrain,
        mtrain,
        Ydev,
        mdev,
        fname=os.path.join(logdir, model_name + ".log"),
        timing=end,
        dofs=count_parameters(model),
    )

    return mtrain, mdev, model


# ========================================================================
def predict_all_dice(model, model_scaler, datadir="data", half=False):
    """
    Predict on data from all dices
    """
    lst = []
    dices = [
        "dice_0002",
        "dice_0003",
        "dice_0004",
        "dice_0005",
        "dice_0006",
        "dice_0007",
        "dice_0008",
        "dice_0009",
        "dice_0010",
    ]

    # Normalization constant (computed on all the data)
    src_pv_norm = src_pv_normalization()

    for dice in dices:
        print(f"Predicting model on {dice}")

        # Load data
        pdf = pd.read_pickle(os.path.join(datadir, f"{dice}_pdfs.gz"))
        means = pd.read_pickle(os.path.join(datadir, f"{dice}_src_pv_means.gz"))
        Xdev = pd.read_pickle(os.path.join(datadir, f"{dice}_xdev.gz"))
        Ydev = pd.read_pickle(os.path.join(datadir, f"{dice}_ydev.gz"))
        dat = np.load(os.path.join(datadir, f"{dice}.npz"))
        z = dat["z"]

        # Switch scaler
        scaler = joblib.load(os.path.join(datadir, f"{dice}_scaler.pkl"))
        Xdev = utilities.switch_scaler(Xdev, scaler, model_scaler)

        if half:
            idx = pdf.xc > 0
            Xdev = Xdev.loc[idx.loc[Xdev.index]]
            Ydev = Ydev.loc[idx.loc[Ydev.index]]

        # Prediction
        mdev = model.predict(Xdev)
        jsd = calculate_jsd(Ydev, mdev)
        jsd90 = np.percentile(jsd, [90])

        # Perform convolution
        conv = convolution_means(mdev, means.loc[Ydev.index])
        rmse, mae, r2 = error_metrics(pdf.SRC_PV.loc[Ydev.index], conv)

        lst.append(
            {
                "z": z,
                "rmse": rmse / src_pv_norm,
                "mae": mae,
                "r2": r2,
                "jsd90": jsd90[0],
            }
        )

    return pd.DataFrame(lst)


# ========================================================================
def predict_full_dices(
    model,
    model_scaler,
    dices=[
        "dice_0002",
        "dice_0003",
        "dice_0004",
        "dice_0005",
        "dice_0006",
        "dice_0007",
        "dice_0008",
        "dice_0009",
        "dice_0010",
    ],
    datadir="data",
    half=False,
):
    """
    Predict on all data from all dices
    """
    lst = []

    for dice in dices:
        print(f"Predicting model on {dice}")

        # Load data
        pdf = pd.read_pickle(os.path.join(datadir, f"{dice}_pdfs.gz"))
        means = pd.read_pickle(os.path.join(datadir, f"{dice}_src_pv_means.gz"))
        X = pd.DataFrame(
            model_scaler.transform(pdf[get_xnames()]),
            index=pdf.index,
            columns=get_xnames(),
        )

        # Prediction
        mpred = model.predict(X)

        # Perform convolution and save data
        df = pd.DataFrame(
            {
                "xc": pdf.xc,
                "yc": pdf.yc,
                "zc": pdf.zc,
                "exact": pdf.SRC_PV,
                "model": convolution_means(mpred, means),
            },
            index=pdf.index,
        )
        df["dice"] = dice
        lst.append(df)

    return pd.concat(lst)


# ========================================================================
def lrp_all_dice(DNN, model_scaler, datadir="data"):
    """
    Calculate DNN LRP on data from all dices
    """
    lst = []
    dices = [
        "dice_0002",
        # "dice_0003",
        "dice_0004",
        # "dice_0005",
        "dice_0006",
        # "dice_0007",
        "dice_0008",
        # "dice_0009",
        "dice_0010",
    ]
    fname = "lrp_hist.pdf"
    with PdfPages(fname) as pdf:
        plt.close("all")
        plt.rc("text", usetex=True)

        for d, dice in enumerate(dices):
            print(f"DNN LRP on {dice}")

            # Load data
            Xdev = pd.read_pickle(os.path.join(datadir, f"{dice}_xdev.gz"))
            dat = np.load(os.path.join(datadir, f"{dice}.npz"))
            z = dat["z"]

            # Switch scaler
            scaler = joblib.load(os.path.join(datadir, f"{dice}_scaler.pkl"))
            Xdev = utilities.switch_scaler(Xdev, scaler, model_scaler)

            # LRP
            lrp_values = lrp.eval_lrp(Xdev, DNN)

            # Compute relevances (TODO: figure out which one to use)
            relevances = np.mean(lrp_values, axis=0)
            # relevances = np.mean(lrp_values, axis=0) / np.sum(np.mean(lrp_values, axis=0))
            # relevances = np.mean(lrp_values / np.sum(lrp_values, axis=1)[:, None], axis=0)

            # Store
            lst.append(
                {
                    "z": z,
                    Xdev.columns[0]: relevances[0],
                    Xdev.columns[1]: relevances[1],
                    Xdev.columns[2]: relevances[2],
                    Xdev.columns[3]: relevances[3],
                }
            )

            # Histogram plot of LRP values
            nbins = 100
            bins = np.linspace(-0.2, 1.0, 100)
            for k in range(lrp_values.shape[1]):
                hist, bins = np.histogram(lrp_values[:, k], bins=nbins, density=True)
                centers = utilities.edges_to_midpoint(bins)

                plt.figure(k)
                plt.plot(centers, hist, color=cmap[d % len(cmap)])

        for k, xname in enumerate(get_xnames()):
            plt.figure(k)
            plt.title(xname)
            plt.tight_layout()
            pdf.savefig(dpi=300, bbox_inches="tight")

    return pd.DataFrame(lst)


# ========================================================================
def shuffled_input_loss(model, X, Y):
    """Get the loss from shuffling each input column

    This provides an estimate of the feature importance of a model by
    evaluating the change in the loss metric when input columns are
    shuffled one at a time.

    Note that this is not a "perfect" way of computing feature
    importance. There are many ways to compute feature importance and
    there are many failure modes
    (ftp://ftp.sas.com/pub/neural/importance.html). One major issue is
    collinearity of input variables.

    :param model: model
    :type model: model
    :param X: data for prediction
    :type X: dataframe
    :param Y: true values
    :type Y: dataframe
    :return: loss normalized by unshuffled loss
    :rtype: dataframe
    """

    def loss(true, predicted, metric="rmse"):
        if metric == "jsd":
            jsd = calculate_jsd(true, predicted)
            return np.percentile(jsd, [90])[0]
        elif metric == "rmse":
            return rmse_metric(true, predicted)

    dic = {}
    metric = "rmse"
    np.random.seed(985_721)
    dic["original"] = loss(Y, model.predict(X), metric=metric)
    for col in X:

        # Shuffle a single column and compute the loss
        df = X.copy()
        np.random.shuffle(df[col].values)
        dic[col] = loss(Y, model.predict(df), metric=metric)

    return pd.DataFrame([dic])


# ========================================================================
def prediction_times(models, X, Y):
    """
    Get predictions times for different models

    :param models: dictionary of models (models must contain predict function)
    :type models: dict
    :param X: data for prediction
    :type X: array
    :param Y: true values
    :type Y: array
    :return: prediction times
    :rtype: dataframe
    """

    lst = []
    N = 10
    for key, model in models.items():

        # Estimate the prediction time
        end = []
        for k in range(N):
            start = time.time()
            mpredict = model.predict(X)
            end.append(time.time() - start)

        # Calculate the prediction error
        jsd = calculate_jsd(Y, mpredict)
        jsd90 = np.percentile(jsd, [90])

        lst.append({"model": key, "time": np.mean(end) / X.shape[0], "error": jsd90[0]})

    return pd.DataFrame(lst)


# ========================================================================
def summarize_training(
    ytrain, mtrain, ydev, mdev, fname="summary.log", timing=0.0, dofs=0
):
    """
    Summarize training

    :param label: method label
    :type label: string
    :param ytrain: true training values
    :type ytrain: array
    :param mtrain: predicted training values
    :type mtrain: array
    :param ydev: true dev values
    :type ydev: array
    :param mdev: predicted dev values
    :type mdev: array
    :param fname: log filename
    :type fname: str
    :param timing: training time
    :type timing: float
    :param dofs: number of degrees of freedom in model
    :type dofs: int
    """

    jsd_train = calculate_jsd(ytrain, mtrain)
    jsd_dev = calculate_jsd(ydev, mdev)
    std_error_train = np.std(np.ravel(ytrain - mtrain) ** 2)
    std_error_dev = np.std(np.ravel(ydev - mdev) ** 2)

    percentiles = [85, 90, 95]
    percentiles_train = np.percentile(jsd_train, percentiles)
    percentiles_dev = np.percentile(jsd_dev, percentiles)

    msg = (
        f"""Training data errors\n"""
        f"""  MAE: {mean_absolute_error(ytrain, mtrain):e}\n"""
        f"""  MSE: {mean_squared_error(ytrain, mtrain):e}\n"""
        f"""  std SE: {std_error_train:e}\n"""
        f"""  R^2: {r2_score(ytrain, mtrain):.2f}\n"""
        f"""  JSD 85 percentile: {percentiles_train[0]:5f}\n"""
        f"""  JSD 90 percentile: {percentiles_train[1]:5f}\n"""
        f"""  JSD 95 percentile: {percentiles_train[2]:5f}\n"""
        f"""\n"""
        f"""Dev data errors\n"""
        f"""  MAE: {mean_absolute_error(ydev, mdev):e}\n"""
        f"""  MSE: {mean_squared_error(ydev, mdev):e}\n"""
        f"""  std SE: {std_error_dev:e}\n"""
        f"""  R^2: {r2_score(ydev, mdev):.2f}\n"""
        f"""  JSD 85 percentile: {percentiles_dev[0]:5f}\n"""
        f"""  JSD 90 percentile: {percentiles_dev[1]:5f}\n"""
        f"""  JSD 95 percentile: {percentiles_dev[2]:5f}\n"""
        f"""\n"""
        f"""Training time: {timing:5f} seconds\n"""
        f"""Model DoFs: {dofs:5d}\n"""
    )

    # Output and write to file
    print(msg)
    with open(fname, "w") as f:
        f.write(msg)


# ========================================================================
def plot_result(ytrain, mtrain, ydev, mdev, labels, bins, fname="summary.pdf"):
    """
    Plot results

    :param ytrain: true training values
    :type ytrain: array
    :param mtrain: predicted training values
    :type mtrain: array
    :param ydev: true dev values
    :type ydev: array
    :param mdev: predicted dev values
    :type mdev: array
    :param labels: PDF labels, i.e. pdf.loc[Xdev.index,Xdev.columns]
    :type labels: dataframe
    :param bins: bins for Z and C
    :type bins: dataframe
    :param fname: plot filename
    :type fname: str
    """

    ytrain = np.asarray(ytrain, dtype=np.float64)
    mtrain = np.asarray(mtrain, dtype=np.float64)
    ydev = np.asarray(ydev, dtype=np.float64)
    mdev = np.asarray(mdev, dtype=np.float64)

    with PdfPages(fname) as pdf:
        plt.close("all")

        # Plot some sample PDF predictions
        nc = len(np.unique(bins.Cbins))
        nz = len(np.unique(bins.Zbins))
        # C = np.reshape(bins.Cbins.values, (nc, nz))
        # Z = np.reshape(bins.Zbins.values, (nc, nz))
        zbin_edges = utilities.midpoint_to_edges(bins.Zbins)
        cbin_edges = utilities.midpoint_to_edges(bins.Cbins)
        extent = [
            zbin_edges.min(),
            zbin_edges.max(),
            zbin_edges.min(),
            cbin_edges.max(),
        ]
        n = 10

        # # Random sort
        # np.random.seed(42)
        # indices = np.random.randint(low=0, high=ydev.shape[0], size=n)

        # Sort by decreasing JSD
        jsd_dev = calculate_jsd(ydev, mdev)
        indices = np.argsort(-jsd_dev)[:n]

        plt.figure(0, figsize=(24, 32))
        plt.clf()
        for i, idx in enumerate(indices):

            ax = plt.subplot(n, 3, 1 + i * 3)
            im = plt.imshow(
                np.reshape(ydev[idx, :], (nc, nz)),
                origin="lower",
                extent=extent,
                aspect="auto",
            )
            plt.colorbar(im)
            plt.xlabel("Mixture Fraction")
            plt.ylabel("Progress Variable")
            label = (
                f"""index:{labels.iloc[idx].name}\n"""
                f"""c  ={labels.iloc[idx].C:.6f}\n"""
                f"""c''={labels.iloc[idx].Cvar:.6f}\n"""
                f"""Z  ={labels.iloc[idx].Z:.6f}\n"""
                f"""Z''={labels.iloc[idx].Zvar:.6f}"""
            )
            plt.title(f"True Joint PDF")
            style = dict(size=10, color="white", ha="left", va="top")
            ax.text(0.02, 0.2, label, **style)

            plt.subplot(n, 3, 2 + i * 3)
            im = plt.imshow(
                np.reshape(mdev[idx, :], (nc, nz)),
                origin="lower",
                extent=extent,
                aspect="auto",
            )
            plt.colorbar(im)
            plt.xlabel("Mixture Fraction")
            plt.ylabel("Progress Variable")
            plt.title("Predicted Joint PDF")

            plt.subplot(n, 3, 3 + i * 3)
            err_dev = mdev - ydev
            im = plt.imshow(
                np.reshape(err_dev[idx, :], (nc, nz)),
                origin="lower",
                extent=extent,
                aspect="auto",
            )
            plt.colorbar(im)
            plt.xlabel("Mixture Fraction")
            plt.ylabel("Progress Variable")
            plt.title(
                "Error in PDF (JSD = {0:f})".format(
                    jensen_shannon_divergence(ydev[idx, :], mdev[idx, :])
                )
            )

        plt.tight_layout()
        pdf.savefig(dpi=300, bbox_inches="tight")

        # PDF of the JSD
        jsd_train = calculate_jsd(ytrain, mtrain)
        jsd_dev = jsd_dev[np.isfinite(jsd_dev)]
        jsd_train = jsd_train[np.isfinite(jsd_train)]

        bins = np.linspace(0, 0.5, 100)
        hist_train, _ = np.histogram(jsd_train, bins=bins, density=True)
        hist_dev, _ = np.histogram(jsd_dev, bins=bins, density=True)
        cum_train = np.cumsum(hist_train) * np.diff(bins)
        cum_dev = np.cumsum(hist_dev) * np.diff(bins)
        centers = utilities.edges_to_midpoint(bins)

        plt.figure(1)
        plt.clf()
        plt.plot(centers, hist_train, label="Train")
        plt.plot(centers, hist_dev, label="Dev")
        plt.legend()
        plt.title(f"PDF of JSD")
        plt.xlabel("")
        plt.tight_layout()
        pdf.savefig(dpi=300, bbox_inches="tight")

        plt.figure(2)
        plt.clf()
        plt.plot(centers, cum_train, label="Train")
        plt.plot(centers, cum_dev, label="Dev")
        plt.legend()
        plt.title(f"Cumulative sum of JSD PDF")
        plt.xlabel("")
        plt.tight_layout()
        pdf.savefig(dpi=300, bbox_inches="tight")

        # # PDF of the relative errors DOESNT make sense for PDF prediction
        # nkde = min(1000, ydev.shape[0])
        # kde_train = gaussian_kde(1 - np.ravel(np.ravel(mtrain[:nkde] / ytrain[:nkde])))
        # kde_dev = gaussian_kde(1 - np.ravel(np.ravel(mdev[:nkde] / ydev[:nkde])))
        # pdf_space = np.linspace(-2e-2, 2e-2, 500)

        # plt.figure(1)
        # plt.clf()
        # plt.semilogy(pdf_space, kde_train(pdf_space), label="Train")
        # plt.semilogy(pdf_space, kde_dev(pdf_space), label="Dev")
        # plt.legend()
        # plt.title(f"{label}: PDF of relative errors")
        # plt.xlabel("")
        # plt.ylim([1e-5, 1e4])
        # plt.tight_layout()
        # pdf.savefig(dpi=300, bbox_inches="tight")


# ========================================================================
def plot_scatter(true, predicted, fname="scatter.pdf"):
    """
    Make a generic scatter plot of true vs predicted
    """

    eps = 1e-13
    error = predicted / (true + eps) - 1
    rmse = rmse_metric(true, predicted)
    lower, upper = np.percentile(error, [5, 95])
    bins = np.linspace(-0.5, 0.5, 100)
    hist, _ = np.histogram(error, bins=bins, density=True)
    centers = utilities.edges_to_midpoint(bins)

    with PdfPages(fname) as pdf:
        plt.close("all")

        plt.figure(0)
        plt.scatter(true, predicted, alpha=0.05)
        plt.plot([0, np.max(true)], [0, np.max(true)], "-k")
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title(f"RMSE: {rmse:.3e}")
        plt.axis("equal")
        plt.tight_layout()
        pdf.savefig(dpi=300, bbox_inches="tight")

        plt.figure(1)
        plt.plot(centers, hist)
        plt.title(f"""PDF of relative errors (90% in [{lower:.3f}, {upper:.3f}])""")
        plt.tight_layout()
        pdf.savefig(dpi=300, bbox_inches="tight")


# ========================================================================
def plot_input_space(pdfs, fname="inputs.pdf"):
    """
    Make plots of the PDF input space
    """

    # Setup
    x = pdfs.Z
    y = pdfs.C

    with PdfPages(fname) as pdf:
        plt.close("all")
        plt.rc("text", usetex=True)

        # definitions for the axes
        pad = 0.02
        left, width = 0.12, 0.65
        bottom, height = 0.1, 0.65
        bottom_h = left + width + pad
        hist_height = 0.2

        # First figure
        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom_h, width, hist_height]

        plt.figure(0, figsize=(10, 8))
        axs0 = plt.axes(rect_scatter)
        axhx = plt.axes(rect_histx)

        # Scatter plot
        im = axs0.scatter(x, y, c=pdfs.Zvar, marker=".", alpha=0.1, cmap="viridis")
        cbar = plt.colorbar(im, ax=axs0)
        cbar.ax.set_title(r"$\widetilde{Z''}$")
        cbar.set_alpha(1.0)
        cbar.ax.tick_params(labelsize=18)
        cbar.draw_all()

        # Histogram on top (resize because of colorbar)
        axhx.hist(x, bins=50, density=True)
        axhx.tick_params(
            axis="both", which="both", bottom=False, top=False, labelbottom=False
        )
        axhx.set_ylabel(r"$P(\widetilde{Z})$", fontsize=22)
        plt.setp(axhx.get_ymajorticklabels(), fontsize=18)
        pos_s0 = axs0.get_position()
        pos_hx = axhx.get_position()
        axhx.set_position([pos_s0.x0, pos_hx.y0, pos_s0.width, pos_hx.height])

        axs0.set_xlabel(r"$\widetilde{Z}$", fontsize=22, fontweight="bold")
        axs0.set_ylabel(r"$\widetilde{c}$", fontsize=22, fontweight="bold")
        plt.setp(axs0.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(axs0.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        pdf.savefig(dpi=300)

        # Second figure
        rect_histy = [left + pos_s0.width + pad, bottom, hist_height, height]

        plt.figure(1, figsize=(10, 8))
        axs1 = plt.axes(rect_scatter)
        axhy = plt.axes(rect_histy)

        # Scatter plot
        im = axs1.scatter(x, y, c=pdfs.Cvar, marker=".", alpha=0.1, cmap="viridis")
        axs1.tick_params(axis="both", which="both", left=False, labelleft=False)
        pos_s1 = axs1.get_position()
        axs1.set_position([pos_s0.x0, pos_s1.y0, pos_s0.width, pos_s1.height])

        # Histogram to the right
        axhy.hist(y, bins=50, density=True, orientation="horizontal")
        axhy.tick_params(
            axis="both",
            which="both",
            bottom=True,
            top=False,
            left=False,
            right=False,
            labelbottom=True,
            labelleft=False,
        )
        axhy.set_xlabel(r"$P(\widetilde{c})$", fontsize=22)
        plt.setp(axhy.get_xmajorticklabels(), fontsize=18)
        pos_hy = axhy.get_position()
        axhy.set_position([pos_hy.x0, pos_hy.y0, pos_hx.height, pos_hy.height])

        # Then colorbar
        cbar = plt.colorbar(im, ax=axhy)
        cbar.ax.set_title(r"$\widetilde{c''}$")
        cbar.set_alpha(1.0)
        cbar.ax.tick_params(labelsize=18)
        cbar.draw_all()

        axs1.set_xlabel(r"$\widetilde{Z}$", fontsize=22, fontweight="bold")
        plt.setp(axs1.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(axs1.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        pdf.savefig(dpi=300)

        # Third figure
        rect_scatter = [left, bottom, width, height]

        plt.figure(2, figsize=(10, 8))
        axs0 = plt.axes(rect_scatter)

        # Scatter plot
        im = axs0.scatter(x, y, c=pdfs.SRC_PV, marker=".", alpha=0.1, cmap="viridis")
        cbar = plt.colorbar(im, ax=axs0)
        cbar.ax.set_title(r"$\widetilde{\dot{\omega}}$")
        cbar.set_alpha(1.0)
        cbar.ax.tick_params(labelsize=18)
        cbar.draw_all()

        axs0.set_xlabel(r"$\widetilde{Z}$", fontsize=22, fontweight="bold")
        axs0.set_ylabel(r"$\widetilde{c}$", fontsize=22, fontweight="bold")
        plt.setp(axs0.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(axs0.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        pdf.savefig(dpi=300)


# ========================================================================
def set_aspect_display_coord(ratio=0.5, ax=None):
    """Set the aspect ratio based on the figure display coordinates"""
    if ax is None:
        ax = plt.gca()
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * ratio)


# ========================================================================
def plot_pdfs(
    pdfs, means, bins, fname="pdfs.pdf", label=False, models=None, legend=False
):
    """
    Make plots of the PDFs
    """

    nc = len(np.unique(bins.Cbins))
    nz = len(np.unique(bins.Zbins))
    zbin_centers = np.unique(bins.Zbins)
    cbin_centers = np.unique(bins.Cbins)
    zbin_edges = utilities.midpoint_to_edges(zbin_centers)
    cbin_edges = utilities.midpoint_to_edges(cbin_centers)
    extent = [zbin_edges.min(), zbin_edges.max(), zbin_edges.min(), cbin_edges.max()]
    y_vars = get_ynames(pdfs)

    nlines = 4
    nskip = len(cbin_centers) // nlines

    with PdfPages(fname) as pdf:
        plt.rc("text", usetex=True)
        plt.close("all")

        fig0 = plt.figure(0)
        fig1 = plt.figure(1)
        fig2 = plt.figure(2)
        ax2 = fig2.add_subplot(111)
        fig3 = plt.figure(3)
        ax3 = fig3.add_subplot(111)
        fig4 = plt.figure(4)
        ax4 = fig4.add_subplot(111)
        fig5 = plt.figure(5)
        ax5 = fig5.add_subplot(111)

        for i, idx in enumerate(pdfs.index):

            pdfi = np.reshape(pdfs.loc[idx, y_vars].values, (nc, nz))
            meansi = np.reshape(means.loc[idx, y_vars].values, (nc, nz))

            # Plot PDF
            plt.figure(0)
            plt.clf()
            ax0 = fig0.add_subplot(111)
            im = ax0.imshow(
                pdfi,
                origin="lower",
                extent=extent,
                aspect="auto",
                interpolation="lanczos",
                cmap="magma",
            )

            if label:
                labels = (
                    f"""$\widetilde{{Z}}  ={pdfs.loc[idx].Z:.4f}$\n"""
                    f"""$\widetilde{{Z''}}={pdfs.loc[idx].Zvar:.4f}$\n"""
                    f"""$\widetilde{{c}}  ={pdfs.loc[idx].C:.4f}$\n"""
                    f"""$\widetilde{{c''}}={pdfs.loc[idx].Cvar:.4f}$\n"""
                    f"""$\widetilde{{\dot{{\omega}}}}={pdfs.loc[idx].SRC_PV:.4f}$"""
                )
                print(labels)
                style = dict(size=10, color="white", ha="left", va="top")
                ax0.text(0.02, 0.2, labels, **style)

            cbar = plt.colorbar(im)
            cbar.ax.set_title(r"$P(Z=Z^*,c = c^*)$")
            ax0.set_yticks([0, 0.05, 0.1, 0.15, 0.2])
            ax0.set_xlabel(r"$Z^*$", fontsize=22, fontweight="bold")
            ax0.set_ylabel(r"$c^*$", fontsize=22, fontweight="bold")
            plt.setp(ax0.get_xmajorticklabels(), fontsize=18, fontweight="bold")
            plt.setp(ax0.get_ymajorticklabels(), fontsize=18, fontweight="bold")
            fig0.subplots_adjust(bottom=0.14)
            fig0.subplots_adjust(left=0.17)
            pdf.savefig(dpi=300)

            # Plot means
            plt.figure(1)
            plt.clf()
            ax1 = fig1.add_subplot(111)
            im = ax1.imshow(
                meansi,
                origin="lower",
                extent=extent,
                aspect="auto",
                interpolation="lanczos",
                cmap="magma",
            )
            cbar = plt.colorbar(im)
            cbar.ax.set_title(r"$\dot{\omega}(Z=Z^*,c = c^*)$")
            ax1.set_yticks([0, 0.05, 0.1, 0.15, 0.2])
            ax1.set_xlabel(r"$Z^*$", fontsize=22, fontweight="bold")
            ax1.set_ylabel(r"$c^*$", fontsize=22, fontweight="bold")
            plt.setp(ax1.get_xmajorticklabels(), fontsize=18, fontweight="bold")
            plt.setp(ax1.get_ymajorticklabels(), fontsize=18, fontweight="bold")
            fig1.subplots_adjust(bottom=0.14)
            fig1.subplots_adjust(left=0.17)
            pdf.savefig(dpi=300)

            # plot marginal distributions
            plt.figure(2)
            p = ax2.plot(
                zbin_centers,
                np.sum(pdfi, axis=0),
                lw=2,
                color=cmap[i],
                label=f"""$\widetilde{{\dot{{\omega}}}}={pdfs.loc[idx].SRC_PV:.2f}$""",
            )
            p[0].set_dashes(dashseq[i])

            if models is not None:
                p[0].set_color(cmap[-1])
                p[0].set_dashes(dashseq[0])
                p[0].set_zorder(10)

                for m, model in enumerate(models):
                    p = ax2.plot(
                        zbin_centers,
                        np.sum(np.reshape(models[model][i, :], (nc, nz)), axis=0),
                        lw=2,
                        color=cmap[m],
                    )
                    p[0].set_dashes(dashseq[m])

            plt.figure(3)
            p = ax3.plot(cbin_centers, np.sum(pdfi, axis=1), lw=2, color=cmap[i])
            p[0].set_dashes(dashseq[i])
            if models is not None:
                p[0].set_color(cmap[-1])
                p[0].set_dashes(dashseq[0])
                p[0].set_zorder(10)

                for m, model in enumerate(models):
                    p = ax3.plot(
                        cbin_centers,
                        np.sum(np.reshape(models[model][i, :], (nc, nz)), axis=1),
                        lw=2,
                        color=cmap[m],
                    )
                    p[0].set_dashes(dashseq[m])

            plt.figure(4)
            p = ax4.plot(zbin_centers, np.sum(meansi, axis=0), lw=2, color=cmap[i])
            p[0].set_dashes(dashseq[i])

            plt.figure(5)
            p = ax5.plot(cbin_centers, np.sum(meansi, axis=1), lw=2, color=cmap[i])
            p[0].set_dashes(dashseq[i])

        plt.figure(2)
        if legend:
            lgd = ax2.legend()
        ax2.set_xlabel(r"$Z^*$", fontsize=22, fontweight="bold")
        ax2.set_ylabel(r"$P(Z=Z^*)$", fontsize=22, fontweight="bold")
        plt.setp(ax2.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax2.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        fig2.subplots_adjust(bottom=0.14)
        fig2.subplots_adjust(left=0.17)
        pdf.savefig(dpi=300)

        plt.figure(3)
        ax3.set_xticks([0, 0.05, 0.1, 0.15, 0.2])
        ax3.set_xlabel(r"$c^*$", fontsize=22, fontweight="bold")
        ax3.set_ylabel(r"$P(c=c^*)$", fontsize=22, fontweight="bold")
        plt.setp(ax3.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax3.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        fig3.subplots_adjust(bottom=0.14)
        fig3.subplots_adjust(left=0.17)
        pdf.savefig(dpi=300)

        plt.figure(4)
        # ax4.set_yscale('symlog', linthreshy=1e-3)
        ax4.set_yscale("log")
        ax4.set_ylim([1e-1, 1e4])
        ax4.set_xlabel(r"$Z^*$", fontsize=22, fontweight="bold")
        ax4.set_ylabel(
            r"$\langle \dot{\omega}|Z=Z^* \rangle$", fontsize=22, fontweight="bold"
        )
        plt.setp(ax4.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax4.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        fig4.subplots_adjust(bottom=0.14)
        fig4.subplots_adjust(left=0.17)
        pdf.savefig(dpi=300)

        plt.figure(5)
        ax5.set_yscale("log")
        ax5.set_ylim([1e-1, 1e4])
        ax5.set_xticks([0, 0.05, 0.1, 0.15, 0.2])
        ax5.set_xlabel(r"$c^*$", fontsize=22, fontweight="bold")
        ax5.set_ylabel(
            r"$\langle \dot{\omega}|c=c^* \rangle$", fontsize=22, fontweight="bold"
        )
        plt.setp(ax5.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax5.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        fig5.subplots_adjust(bottom=0.14)
        fig5.subplots_adjust(left=0.17)
        pdf.savefig(dpi=300)


# ========================================================================
def plot_dice_slices(fname):
    """
    Make plots of slices in the dice

    :param fname: dice file name
    :type fname: str
    """

    # Load dice file
    dat = np.load(fname)

    # Variables
    low = dat["low"]
    high = dat["high"]
    extent = [low[0], high[0], low[1], high[1]]
    dx = dat["dx"]
    z = dat["z"]
    rho = dat["Rho"]

    # index of slice
    slc = np.s_[:, :, rho.shape[2] // 2]

    # Get slices
    rho = rho[slc].T
    Z = np.clip(dat["Z"][slc].T, 0.0, 1.0)
    C = np.clip(dat["C"][slc].T, 0.0, None)
    SRC_PV = np.clip(dat["SRC_PV"][slc].T, 0.0, None)
    rhoZ = rho * Z
    rhoC = rho * C
    rhoSRC_PV = rho * SRC_PV

    # Filter
    width = 32
    rhof = ndimage.uniform_filter(rho, size=width)
    Zf = ndimage.uniform_filter(rhoZ, size=width) / rhof
    Zvarf = ndimage.uniform_filter(rho * (Z - Zf) ** 2, size=width) / rhof
    Cf = ndimage.uniform_filter(rhoC, size=width) / rhof
    Cvarf = ndimage.uniform_filter(rho * (C - Cf) ** 2, size=width) / rhof
    SRC_PVf = ndimage.uniform_filter(rhoSRC_PV, size=width) / rhof

    figname = os.path.splitext(fname)[0] + "_slice.pdf"
    with PdfPages(figname) as pdf:
        plt.close("all")
        plt.rc("text", usetex=True)

        fields = [
            {"field": Zf, "label": "$\\widetilde{Z}$"},
            {"field": Zvarf, "label": "$\\widetilde{Z''}$"},
            {"field": Cf, "label": "$\\widetilde{c}$"},
            {"field": Cvarf, "label": "$\\widetilde{c''}$"},
            {"field": SRC_PVf, "label": "$\\widetilde{\\dot{\\omega}}$"},
        ]

        for i, field in enumerate(fields):
            fig, (ax0) = plt.subplots(1)
            im0 = ax0.imshow(
                field["field"], origin="lower", extent=extent, aspect="equal"
            )
            cbar = plt.colorbar(im0, ax=ax0)
            cbar.ax.set_title(f"""{field["label"]}""")
            ax0.set_xlabel(r"$x~[\mathrm{m}]$", fontsize=22, fontweight="bold")
            ax0.set_ylabel(r"$y~[\mathrm{m}]$", fontsize=22, fontweight="bold")
            ticks = [-0.06, 0, 0.06]
            ax0.set_xticks(ticks)
            ax0.set_yticks(ticks)
            plt.setp(ax0.get_xmajorticklabels(), fontsize=18, fontweight="bold")
            plt.setp(ax0.get_ymajorticklabels(), fontsize=18, fontweight="bold")
            plt.gcf().subplots_adjust(bottom=0.15)
            plt.gcf().subplots_adjust(left=0.17)
            pdf.savefig(dpi=300)


# ========================================================================
def plot_jsd(jsd, legend=False):
    """
    Make plots of JSD for different models

    :param jsd: JSD for different models
    :type jsd: dataframe
    :param legend: Draw legend on plots
    :type legend: bool
    """

    fname = "jsd.pdf"
    pdf_space = np.linspace(0, 0.7, 500)
    bins = np.linspace(0, 0.7, 100)

    with PdfPages(fname) as pdf:
        plt.close("all")
        plt.rc("text", usetex=True)

        fig0 = plt.figure(0)
        ax0 = fig0.add_subplot(111)
        fig1 = plt.figure(1)
        ax1 = fig1.add_subplot(111)
        lst = []
        for k, model in enumerate(jsd):

            kde = gaussian_kde(jsd[model])
            pkde = kde(pdf_space)
            lst.append({"model": model, "jsd90": np.percentile(jsd[model], [90])[0]})
            hist, _ = np.histogram(jsd[model], bins=bins, density=True)
            centers = utilities.edges_to_midpoint(bins)
            cum_hist = np.cumsum(hist) * np.diff(bins)

            plt.figure(0)
            p = ax0.plot(pdf_space, pkde, lw=2, color=cmap[k], label=model)
            p[0].set_dashes(dashseq[k])

            plt.figure(1)
            p = ax1.plot(centers, cum_hist, lw=2, color=cmap[k], label=model)
            p[0].set_dashes(dashseq[k])

        df = pd.DataFrame(lst)
        print(df.to_latex())

        # Format figures
        plt.figure(0)
        if legend:
            lgd = ax0.legend()
        ax0.set_xlabel(r"$J$", fontsize=22, fontweight="bold")
        ax0.set_ylabel(r"$P(J)$", fontsize=22, fontweight="bold")
        plt.setp(ax0.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax0.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        fig0.subplots_adjust(bottom=0.15)
        fig0.subplots_adjust(left=0.17)
        pdf.savefig(dpi=300)

        plt.figure(1)
        ax1.set_xlabel(r"$J$", fontsize=22, fontweight="bold")
        ax1.set_ylabel(r"$CDF(J)$", fontsize=22, fontweight="bold")
        plt.setp(ax1.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax1.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        fig1.subplots_adjust(bottom=0.15)
        fig1.subplots_adjust(left=0.17)
        pdf.savefig(dpi=300)


# ========================================================================
def plot_dice_predictions(predictions, legend=False):
    """
    Make plots for predictions from all dice
    :param legend: Draw legend on plots
    :type legend: bool
    """

    fname = "dice_predictions.pdf"
    with PdfPages(fname) as pdf:
        plt.close("all")
        plt.rc("text", usetex=True)

        fig0 = plt.figure(0)
        ax0 = fig0.add_subplot(111)
        fig1 = plt.figure(1)
        ax1 = fig1.add_subplot(111)

        for k, model in enumerate(predictions):
            plt.figure(0)
            p = ax0.plot(
                predictions[model].z,
                predictions[model].jsd90,
                lw=2,
                color=cmap[k],
                label=model,
                marker=markertype[k],
                ms=10,
            )
            p[0].set_dashes(dashseq[k])

            plt.figure(1)
            p = ax1.plot(
                predictions[model].z,
                predictions[model].rmse,
                lw=2,
                color=cmap[k],
                label=model,
                marker=markertype[k],
                ms=10,
            )
            p[0].set_dashes(dashseq[k])

        # Format figures
        plt.figure(0)
        if legend:
            lgd = ax0.legend()
        ax0.set_xlabel(r"$z~[\mathrm{m}]$", fontsize=22, fontweight="bold")
        ax0.set_ylabel(r"$J_{90}$", fontsize=22, fontweight="bold")
        ax0.set_ylim([0, 0.75])
        plt.setp(ax0.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax0.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        fig0.subplots_adjust(bottom=0.15)
        fig0.subplots_adjust(left=0.17)
        pdf.savefig(dpi=300)

        plt.figure(1)
        ax1.set_xlabel(r"$z~[\mathrm{m}]$", fontsize=22, fontweight="bold")
        ax1.set_ylabel(
            r"RMSE$(\widetilde{\dot{\omega}})$", fontsize=22, fontweight="bold"
        )
        plt.setp(ax1.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax1.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        fig1.subplots_adjust(bottom=0.15)
        fig1.subplots_adjust(left=0.17)
        pdf.savefig(dpi=300)


# ========================================================================
def plot_convolution(true, convolutions, bins, legend=False):
    """
    Make plots of convolution for different models

    :param true: dataframe containing true SRC_PV values
    :type true: dataframe
    :param convolution: convolution for different models
    :type convolution: dataframe
    :param legend: Draw legend on plots
    :type legend: bool
    """

    src_pv = np.asarray(true.SRC_PV, dtype=np.float64)
    zbin_centers = np.unique(bins.Zbins)
    cbin_centers = np.unique(bins.Cbins)
    zbin_edges = utilities.midpoint_to_edges(zbin_centers)
    cbin_edges = utilities.midpoint_to_edges(cbin_centers)

    # Normalization constant (computed on all the data)
    src_pv_norm = src_pv_normalization()

    fname = "convolution.pdf"
    pdf_space = np.linspace(-0.5, 0.5, 500)
    with PdfPages(fname) as pdf:
        plt.close("all")
        plt.rc("text", usetex=True)
        plt.rc("text.latex", preamble=r"\usepackage{nicefrac}")

        fig0 = plt.figure(0)
        ax0 = fig0.add_subplot(111)
        ax0.plot([0, np.max(src_pv)], [0, np.max(src_pv)], lw=1, color=cmap[-1])

        fig1 = plt.figure(1)
        ax1 = fig1.add_subplot(111)

        fig2 = plt.figure(2)
        ax2 = fig2.add_subplot(111)

        fig3 = plt.figure(3)
        ax3 = fig3.add_subplot(111)

        lst = []
        for k, model in enumerate(convolutions):

            # Skip invalid models
            if convolutions[model].isnull().values.any():
                continue

            error = (src_pv - convolutions[model]) / src_pv_norm
            rmse, mae, r2 = error_metrics(src_pv, convolutions[model])
            rmse /= src_pv_norm
            lst.append({"model": model, "rmse": rmse, "r2": r2, "mae": mae})

            # PDF of the error
            kde = gaussian_kde(error)
            pkde = kde(pdf_space)

            # Conditionals
            src_pv_cond_z, _, _ = stats.binned_statistic(
                true.Z, convolutions[model], statistic="mean", bins=zbin_edges
            )
            src_pv_cond_c, _, _ = stats.binned_statistic(
                true.C, convolutions[model], statistic="mean", bins=cbin_edges
            )

            plt.figure(0)
            ax0.scatter(
                src_pv,
                convolutions[model],
                c=cmap[k],
                alpha=0.2,
                s=15,
                marker=markertype[k],
                label=f"""{model}: RMSE = {rmse:.2f}, $R^2$ = {r2:.2f}""",
            )

            plt.figure(1)
            p = ax1.plot(pdf_space, pkde, lw=2, color=cmap[k], label=model)
            p[0].set_dashes(dashseq[k])

            plt.figure(2)
            p = ax2.plot(zbin_centers, src_pv_cond_z, lw=2, color=cmap[k], label=model)
            p[0].set_dashes(dashseq[k])

            plt.figure(3)
            p = ax3.plot(cbin_centers, src_pv_cond_c, lw=2, color=cmap[k], label=model)
            p[0].set_dashes(dashseq[k])

        # True conditionals
        src_pv_cond_z, _, _ = stats.binned_statistic(
            true.Z, src_pv, statistic="mean", bins=zbin_edges
        )
        src_pv_cond_c, _, _ = stats.binned_statistic(
            true.C, src_pv, statistic="mean", bins=cbin_edges
        )

        plt.figure(2)
        p = ax2.plot(zbin_centers, src_pv_cond_z, lw=2, color=cmap[-1], label=model)
        p[0].set_zorder(0)

        plt.figure(3)
        p = ax3.plot(cbin_centers, src_pv_cond_c, lw=2, color=cmap[-1], label=model)
        p[0].set_zorder(0)

        # Make a table from the legend
        df = pd.DataFrame(lst)
        print(df.to_latex())
        # table = r"""\begin{tabular}{ cccc } & model & RMSE & $R^2$ \\\hline"""
        # for index, row in df.iterrows():
        #     table += (
        #         r" \begin{tikzpicture}"
        #         + r" \draw[red,thick,solid,fill=blue]"
        #         + " (0,0) circle (0.1cm); \end{tikzpicture}"
        #         + r" & {0:s} & {1:.2f} & {2:.2f} \\".format(row.model, row.rmse, row.r2)
        #     )
        # table += r"""\end{tabular}"""
        # print(table)
        # ax0.text(9,3.4,table,size=5, color='black',  bbox=dict(facecolor='none', edgecolor='gray', boxstyle='round,pad=0.5'))

        # Format figures
        plt.figure(0)
        if legend:
            lgd = ax0.legend()
        ax0.set_xlabel(r"$\widetilde{\dot{\omega}}$", fontsize=22, fontweight="bold")
        ax0.set_ylabel(r"$\widetilde{\dot{\omega}}_m$", fontsize=22, fontweight="bold")
        plt.setp(ax0.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax0.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        fig0.subplots_adjust(bottom=0.15)
        fig0.subplots_adjust(left=0.17)
        pdf.savefig(dpi=300)

        plt.figure(1)
        ax1.set_xlabel(
            r"$\nicefrac{\epsilon(\widetilde{\dot{\omega}})}{\widetilde{\dot{\Omega}}}$",
            fontsize=22,
            fontweight="bold",
        )
        ax1.set_ylabel(
            r"$P(\nicefrac{\epsilon(\widetilde{\dot{\omega}})}{\widetilde{\dot{\Omega}}})$",
            fontsize=22,
            fontweight="bold",
        )
        plt.setp(ax1.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax1.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        ax1.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
        ax1.set_yscale("log")
        fig1.subplots_adjust(bottom=0.15)
        fig1.subplots_adjust(left=0.17)
        pdf.savefig(dpi=300)

        plt.figure(2)
        if legend:
            lgd = ax2.legend()
        ax2.set_xlabel(r"$\widetilde{Z}$", fontsize=22, fontweight="bold")
        ax2.set_ylabel(
            r"$\langle \widetilde{\dot{\omega}} | Z=\widetilde{Z} \rangle$",
            fontsize=22,
            fontweight="bold",
        )
        plt.setp(ax2.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax2.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        fig2.subplots_adjust(bottom=0.15)
        fig2.subplots_adjust(left=0.17)
        pdf.savefig(dpi=300)

        plt.figure(3)
        ax3.set_xticks([0, 0.05, 0.1, 0.15, 0.2])
        if legend:
            lgd = ax3.legend()
        ax3.set_xlabel(r"$\widetilde{c}$", fontsize=22, fontweight="bold")
        ax3.set_ylabel(
            r"$\langle \widetilde{\dot{\omega}} | c=\widetilde{c} \rangle$",
            fontsize=22,
            fontweight="bold",
        )
        plt.setp(ax3.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax3.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        fig3.subplots_adjust(bottom=0.15)
        fig3.subplots_adjust(left=0.17)
        pdf.savefig(dpi=300)


# ========================================================================
def plot_pdf_distances(datadir="data"):

    bases = ["dice_0004", "dice_0006", "dices_skip"]
    fname = "pdf_distances.pdf"
    pdf_space = np.linspace(0, np.log(2), 500)

    with PdfPages(fname) as pdf:
        plt.close("all")
        plt.rc("text", usetex=True)

        fig1 = plt.figure(1)
        ax1 = fig1.add_subplot(111)

        for i, base in enumerate(bases):
            with open(os.path.join(datadir, f"{base}_pdf_distances.pkl"), "rb") as f:
                distances = pickle.load(f)
            fig0 = plt.figure(0)
            fig0.clf()
            ax0 = fig0.add_subplot(111)

            r90 = np.zeros((2, len(distances)))
            for k, dice in enumerate(distances):

                # 90th percentile
                dat = np.load(os.path.join(datadir, f"{dice}.npz"))
                r90[0, k] = dat["z"]
                r90[1, k] = np.percentile(distances[dice].r, [90])[0]

                # smooth pdf
                kde = gaussian_kde(distances[dice].r)
                pkde = kde(pdf_space)

                plt.figure(0)
                p = ax0.plot(
                    pdf_space, pkde, lw=2, color=cmap[k % len(cmap)], label=dice
                )
                p[0].set_dashes(dashseq[k % len(dashseq)])

            plt.figure(0)
            ax0.set_xlabel(r"$r$", fontsize=22, fontweight="bold")
            ax0.set_ylabel(r"$P(r)$", fontsize=22, fontweight="bold")
            plt.setp(ax0.get_xmajorticklabels(), fontsize=18, fontweight="bold")
            plt.setp(ax0.get_ymajorticklabels(), fontsize=18, fontweight="bold")
            fig0.subplots_adjust(bottom=0.15)
            fig0.subplots_adjust(left=0.17)
            pdf.savefig(dpi=300)

            plt.figure(1)
            p = ax1.plot(
                r90[0, :], r90[1, :], lw=2, color=cmap[i], marker=markertype[i], ms=10
            )
            p[0].set_dashes(dashseq[i])

        plt.figure(1)
        ax1.set_ylim([0, 0.75])
        ax1.set_xlabel(r"$z~[\mathrm{m}]$", fontsize=22, fontweight="bold")
        ax1.set_ylabel(r"$r_{90}$", fontsize=22, fontweight="bold")
        plt.setp(ax1.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax1.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        fig1.subplots_adjust(bottom=0.15)
        fig1.subplots_adjust(left=0.17)
        pdf.savefig(dpi=300)


# ========================================================================
def plot_lrp(lrps, legend=False):
    """
    Make plots for LRP values
    :param legend: Draw legend on plots
    :type legend: bool
    """

    fname = "lrps.pdf"
    with PdfPages(fname) as pdf:
        plt.close("all")
        plt.rc("text", usetex=True)

        fig0 = plt.figure(0)
        ax0 = fig0.add_subplot(111)

        for k, xname in enumerate(get_xnames()):
            plt.figure(0)
            p = ax0.plot(
                lrps.z,
                lrps[xname],
                lw=2,
                color=cmap[k],
                label=xname,
                marker=markertype[k],
                ms=10,
            )
            p[0].set_dashes(dashseq[k])

        # Format figures
        plt.figure(0)
        if legend:
            lgd = ax0.legend()
        ax0.set_xlabel(r"$z~[\mathrm{m}]$", fontsize=22, fontweight="bold")
        ax0.set_ylabel(r"$R$", fontsize=22, fontweight="bold")
        # ax0.set_ylim([0, 1])
        plt.setp(ax0.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax0.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        fig0.subplots_adjust(bottom=0.15)
        fig0.subplots_adjust(left=0.17)
        pdf.savefig(dpi=300)
