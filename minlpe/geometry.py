import copy, scipy
import numpy as np


def relDist(a, b):
    assert len(a.shape) <= 2 and len(b.shape) <= 2
    if len(a.shape) == 1: a = a.reshape(1, -1)
    if len(b.shape) == 1: b = b.reshape(1, -1)
    sum = np.linalg.norm(a, axis=1) + np.linalg.norm(b, axis=1)
    sum[sum == 0] = 1
    return np.linalg.norm(a-b, axis=1)/sum


def getNNdistsStable(x1, x2):
    dists = scipy.spatial.distance.cdist(x1, x2)
    M = np.max(dists)
    mean = np.mean(dists)
    np.fill_diagonal(dists, M)
    NNdists = np.min(dists, axis=1)
    return NNdists, mean


def getMinDist(x1, x2):
    NNdists, mean = getNNdistsStable(x1, x2)
    # take min element with max index (we want to throw last points first)
    min_dist_ind = np.max(np.where(NNdists == np.min(NNdists))[0])
    min_dist = NNdists[min_dist_ind]
    return min_dist_ind, min_dist, mean


def unique_mulitdim(p, rel_err=1e-6):
    """
    Remove duplicate points from array p

    :param p: points (each row is one point)
    :param rel_err: max difference between points to be equal i.e. dist < np.quantile(all_dists, 0.1) * rel_err
    :returns: new_p, good_ind
    """
    p = copy.deepcopy(p)
    assert len(p.shape) == 2, 'p must be 2-dim (each row is one point). For 1 dim we can\'t determine whether it is row or column'
    good_ind = np.arange(len(p))
    min_dist_ind, min_dist, med_dist = getMinDist(p,p)
    while min_dist <= med_dist * rel_err:
        good_ind = np.delete(good_ind, min_dist_ind)
        min_dist_ind, min_dist, _ = getMinDist(p[good_ind, :], p[good_ind, :])
    return p[good_ind, :], good_ind


def length(x):
    if hasattr(x, "__len__"): return len(x)
    else: return 1   # x is scalar
