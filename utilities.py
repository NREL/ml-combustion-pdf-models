# ========================================================================
#
# Imports
#
# ========================================================================
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.externals import joblib


# ========================================================================
#
# Function definitions
#
# ========================================================================
def split_df(df, train_frac=0.9, dev_frac=0.05, seed=983324):
    """
    Split data frame into a train, dev and test set

    :param df: dataframe to split
    :type df: dataframe
    :param train_frac: fraction of data in training set
    :type train_frac: float
    :param dev_frac: fraction of data in dev set
    :type dev_frac: float
    :param seed: seed for random number generator
    :type seed: int
    :return: train, dev and test dataframes
    :rtype: list
    """

    frac = train_frac + dev_frac

    return np.split(
        df.sample(frac=1, random_state=seed),
        [int(train_frac * len(df)), int(frac * len(df))],
    )


# ========================================================================
def synthetic_data():
    """
    Generate some synthetic data
    """
    x_vars = ["x0", "x1"]
    y_vars = ["y"]

    N = 10000
    x = [
        np.random.uniform(low=0, high=1, size=N),
        np.random.uniform(low=0, high=1, size=N),
    ]
    y = x[0] + 2 * x[1]

    df = pd.DataFrame({x_vars[0]: x[0], x_vars[1]: x[1], y_vars[0]: y})
    return gen_training(df, x_vars, y_vars, oname="synthetic")


# ========================================================================
def gen_training(df, x_vars, y_vars, oname="training"):
    """
    Generate scaled training, dev, test arrays
    :param x_vars: names of input variables
    :type x_vars: list
    :param y_vars: names of ouput variables
    :type y_vars: list
    :param oname: output file name of data
    :type oname: str
    """

    # Split the data into different sets
    train, dev, test = split_df(df)

    Xtrain = train.loc[:, x_vars]
    Ytrain = train.loc[:, y_vars]

    Xdev = dev.loc[:, x_vars]
    Ydev = dev.loc[:, y_vars]

    Xtest = test.loc[:, x_vars]
    Ytest = test.loc[:, y_vars]

    # Scale the data
    scaler = RobustScaler()
    scaler.fit(Xtrain)
    Xtrain = pd.DataFrame(
        scaler.transform(Xtrain), index=Xtrain.index, columns=Xtrain.columns
    )
    Xdev = pd.DataFrame(scaler.transform(Xdev), index=Xdev.index, columns=Xdev.columns)
    Xtest = pd.DataFrame(
        scaler.transform(Xtest), index=Xtest.index, columns=Xtest.columns
    )

    # Save the data
    Xtrain.to_pickle(oname + "_xtrain.gz")
    Xdev.to_pickle(oname + "_xdev.gz")
    Xtest.to_pickle(oname + "_xtest.gz")
    Ytrain.to_pickle(oname + "_ytrain.gz")
    Ydev.to_pickle(oname + "_ydev.gz")
    Ytest.to_pickle(oname + "_ytest.gz")
    joblib.dump(scaler, oname + "_scaler.pkl")

    return Xtrain, Xdev, Xtest, Ytrain, Ydev, Ytest, scaler


# ========================================================================
def switch_scaler(X, original, new=None):
    """
    Switch the scaler
    :param X: data scaled by orig
    :type X: array or dataframe
    :param original: original scaler
    :type original: scaler
    :param new: new scaler
    :type new: scaler
    :return Xnew: new(inverse_original(X))
    :rtype Xnew: array or dataframe
    """

    # new is an identity transform (only do the inverse transform)
    if new is None:
        new = StandardScaler()
        new.mean_ = 0.0
        new.scale_ = 1.0

    # Inverse the transformation
    inverse = original.inverse_transform(X)

    if isinstance(X, pd.DataFrame):
        return pd.DataFrame(new.transform(inverse), index=X.index, columns=X.columns)
    else:
        return new.transform(inverse)


# ========================================================================
def midpoint_to_edges(centers):
    """
    Return the bin edges given the center points of the bins
    """
    centers = np.asarray(centers, dtype=np.float64)
    d = 0.5 * np.diff(centers)
    return np.hstack([centers[0] - d[0], centers[:-1] + d, centers[-1] + d[-1]])


# ========================================================================
def edges_to_midpoint(edges):
    """
    Return the bin centers points given the bin edges
    """
    edges = np.asarray(edges, dtype=np.float64)
    return 0.5 * (edges[1:] + edges[:-1])
