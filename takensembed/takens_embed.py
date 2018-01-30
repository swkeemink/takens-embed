"""An implementation of the Taken's embedding theorem.

Based on Sugihara et al (2012) - Detecting Causality in Complex Ecosystems.

Author: Sander Keemink.
"""
import random
import numpy as np
from sklearn.neighbors import NearestNeighbors


def get_delayed_manifold(data, tau=10, ndelay=3):
    """Get the delayed manifolds of the variables in data.

    Parameters
    ----------
    data : array
        ntimepoints*nvariable data array
    tau : int, optional
        how many timepoints per delay step
    ndelay : int
        how many delay steps, optional

    Returns
    -------
    array
        ndim*ndata*ndelay array

    """
    N = data.shape[0]  # number of data points
    ndim = data.shape[1]  # number of input dimensions
    delayed_manifolds = np.zeros((ndim, N-ndelay*tau, ndelay))
    for dim in range(ndim):
        for n in range(ndelay):
            delayed_manifolds[dim, :, n] = np.roll(data[:, dim],
                                                   -n*tau)[:-ndelay*tau]
    return delayed_manifolds


def findknearest(data, k):
    """Find k nearest neighbours.

    Parameters
    ----------
    data : array
        Data to apply NearestNeighbors to
        (see sklearn.neighbors.NearestNeighbors)
    k : int
        How many neighbours to find.

    Returns
    -------
    array
        distances to k nearest points
    array
        ids of k nearest points
    """
    neigh = NearestNeighbors(n_neighbors=k+1)
    neigh.fit(data)
    dists, ids = neigh.kneighbors(data, n_neighbors=k+1)

    return dists[:, 1:], ids[:, 1:]


def do_embedding(delayed_manifolds, rnge=None,
                 randomize_coordinates=False):
    """Do embedding at different time-point-lengths.

    Parameters
    ----------
    delayed_manifolds : array
        ndim*ndata*ndelay array with delayed single time courses
    rnge : array, optional
        At which time points to calculate the predictability of the variables
    randomize_coordinates : bool, optional
        If true, will randomize the delay coordinates as in Tajima et al (2015)

    Returns
    -------
    array
        The correlations

    """
    if rnge is None:
        rnge = range(20, 5000, 20)

    # randomize delay coordinates if requested
    if randomize_coordinates:
        delayed_randomized = np.copy(delayed_manifolds)
        for i in range(delayed_manifolds.shape[0]):
            R = np.random.normal(0, 1, (delayed_manifolds.shape[2],
                                        delayed_manifolds.shape[2]))
            for j in range(delayed_manifolds.shape[1]):
                delayed_randomized[i, j, :] = np.dot(
                                                 R, delayed_manifolds[i, j, :])
        delayed_manifolds = delayed_randomized

    # get some information about data size
    ndelay = delayed_manifolds.shape[2]
    ndims = delayed_manifolds.shape[0]
    N = delayed_manifolds.shape[1]

    # start analysis
    data = delayed_manifolds
    k = ndelay+3  # how many neighbours to find
    cors = np.zeros((ndims, ndims, len(rnge)))
    # loop over time lengths
    for i, l in enumerate(rnge):
        indices = random.sample(range(N), l)
        data_cut = data[:, indices, :]
        dists, ids, weights, preds = {}, {}, {}, {}
        # loop over actual dimensions
        for dim in range(ndims):
            # get nearest neighbours
            dists[dim], ids[dim] = findknearest(data_cut[dim, :, :], k)

            # get weights as per pop paper
            minim = dists[dim].min(axis=1)
            weights[dim] = np.exp(-dists[dim]/minim[:, None])
            weights[dim] /= weights[dim].sum(axis=1)[:, None]
        # get predictions from cross embeddings for all dimension combinations
        for dim1 in range(ndims):  # dimension to use to predict
            for dim2 in range(ndims):  # dimension to predict
                points_to_use = data_cut[dim2, ids[dim1], 0]
                preds[dim1, dim2] = np.sum(weights[dim1][:, :]*points_to_use,
                                           axis=1)
                cors[dim1, dim2, i] = np.corrcoef(preds[dim1, dim2],
                                                  data_cut[dim2, :, 0])[0, 1]

    return cors
