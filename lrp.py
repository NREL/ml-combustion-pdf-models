#!/usr/bin/env python3


# ========================================================================
#
# Imports
#
# ========================================================================
import types
import copy
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
from itertools import chain


# ========================================================================
#
# Function definitions
#
# ========================================================================
def forward_hook(self, input, output):
    """Forward hook method for retrieving intermediate results"""
    self.X = input[0]
    self.Y = output


# ========================================================================
def zplus_relprop(self, R):
    """
    Relevance propagation of positive weights
    """

    Z = torch.clamp(
        torch.transpose(self.weight, 0, 1)[None, :, :] * self.X[:, :, None], min=1e-16
    )
    b = torch.clamp(self.bias, min=1e-16)
    Zs = Z.sum(dim=1)[:, None, :] + b[None, None, :]
    R = ((Z / Zs) * R[:, None, :]).sum(dim=2)
    return R


# ========================================================================
def w2_relprop(self, R):
    """
    Weights-squared relevance propagation
    """
    W2 = self.weight ** 2
    Z = torch.sum(W2, dim=1) + 1e-9
    S = R / Z
    R = torch.mm(S, W2)
    return R


# ========================================================================
def gen_relprop(self, R):
    """
    Basic linear relevance propagation

    Eq(56) in DOI: 10.1371/journal.pone.0130140
    """
    Z = torch.transpose(self.weight, 0, 1)[None, :, :] * self.X[:, :, None]
    Zs = Z.sum(dim=1)[:, None, :] + self.bias[None, None, :]
    Zs[Zs >= 0] += 1e-9
    Zs[Zs < 0] -= 1e-9
    R = ((Z / Zs) * R[:, None, :]).sum(dim=2)
    return R


# ========================================================================
def ab_relprop(self, R):
    """
    Alpha-beta linear relevance propagation

    Eq(60) in DOI: 10.1371/journal.pone.0130140
    """
    alpha = 0.99
    beta = 1 - alpha
    Z = torch.transpose(self.weight, 0, 1)[None, :, :] * self.X[:, :, None]

    if not alpha == 0:
        Zp = torch.clamp(Z, min=1e-16)
        bp = torch.clamp(self.bias, min=1e-16)
        Zsp = Zp.sum(dim=1)[:, None, :] + bp[None, None, :]
        Ralpha = alpha * ((Zp / Zsp) * R[:, None, :]).sum(dim=2)
    else:
        Ralpha = 0

    if not beta == 0:
        Zn = torch.clamp(Z, max=-1e-16)
        bn = torch.clamp(self.bias, max=-1e-16)
        Zsn = Zn.sum(dim=1)[:, None, :] + bn[None, None, :]
        Rbeta = beta * ((Zn / Zsn) * R[:, None, :]).sum(dim=2)
    else:
        Rbeta = 0

    return Ralpha + Rbeta


# ========================================================================
def relu_relprop(self, R):
    return R


# ========================================================================
def leakyrelu_relprop(self, R):
    R[self.X < 0] *= self.negative_slope
    return R


# ========================================================================
def batchnorm1d_relprop(self, R):
    """
    Relevance propagation for batchnorm

    Implements:
             x * (y - beta)     R
        R = ---------------- * ---
                x - mu          y

    Taken from discussion found at:
    https://github.com/albermax/innvestigate/issues/2
    https://github.com/albermax/innvestigate/blob/master/innvestigate/analyzer/relevance_based/relevance_analyzer.py
    https://github.com/sebastian-lapuschkin/lrp_toolbox/issues/10

    """
    num = self.X * (self.Y - self.bias)
    den = (self.X - self.running_mean) * self.Y
    den[den >= 0] += 1e-9
    den[den < 0] -= 1e-9
    R = num / den * R
    return R


# ========================================================================
def softmax_relprop(self, R):
    return R


# ========================================================================
def lrp_dnn(model):
    """
    Add relevance propagation functionality to a DNN
    """

    lrp = copy.deepcopy(model)
    lrp.eval()

    # Add relprop function to each layer in the model
    all_layers = chain(lrp.MLP.children())
    for layer in all_layers:
        if type(layer).__name__ == "Linear":
            # TODO: ab_ or gen_ ?
            layer.relprop = types.MethodType(ab_relprop, layer)

        if type(layer).__name__ == "BatchNorm1d":
            layer.relprop = types.MethodType(batchnorm1d_relprop, layer)

        if type(layer).__name__ == "LeakyReLU":
            # TODO: relu or leakyrelu ?
            layer.relprop = types.MethodType(relu_relprop, layer)

        if type(layer).__name__ == "Softmax":
            layer.relprop = types.MethodType(softmax_relprop, layer)

        layer.register_forward_hook(forward_hook)

    # TODO: gen_ the input layer?
    # lrp.MLP.L0.relprop = types.MethodType(gen_relprop, lrp.MLP.L0)

    # Add relprop function to the model
    def relprop(self, R):

        for l in range(len(self.MLP), 0, -1):
            R = self.MLP[l - 1].relprop(R)

        return R

    lrp.relprop = types.MethodType(relprop, lrp)

    return lrp


# ========================================================================
def eval_lrp(X, model):

    # Model with relprop added in
    lrp = lrp_dnn(model)

    # Loop to evaluate LRP
    X = np.asarray(X, dtype=np.float64)
    lrp_values = np.zeros(X.shape)
    batch_size = 1024
    for batch, i in enumerate(range(0, X.shape[0], batch_size)):

        batch_x = Variable(torch.from_numpy(X[i : i + batch_size, :]))
        y_pred = lrp(batch_x)
        lrp_values[i : i + batch_size, :] = lrp.relprop(y_pred).data.numpy()

    return lrp_values
