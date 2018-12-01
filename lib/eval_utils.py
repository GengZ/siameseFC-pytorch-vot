import numpy as np
from numpy import matlib


def precisionAuc(positions, groundTruth, radius, nStep):
    thres = np.linspace(0, radius, nStep)
    errs = np.zeros([nStep], dtype=np.float32)

    distances = np.sqrt(np.power(positions[:, 0] - groundTruth[:, 0], 2) +
                        np.power(positions[:, 1] - groundTruth[:, 1], 2))
    distances[np.where(np.isnan(distances))] = []

    for p in range(nStep):
        errs[p] = np.shape(np.where(distances > thres[p]))[-1]
    score = np.trapz(errs)
    return score


def centerThrErr(score, labels, oldRes, m):
    radiusInPix = 50
    totalStride = 8
    nStep = 100
    batchSize = score.shape[0]
    # posMask = np.where(labels > 0)
    posMask = np.arange(score.shape[0])
    numPos = posMask.shape[0]

    responses = score           # ?
    half = np.floor(score.shape[-1]/2)
    centerLabel = np.matlib.repmat([half, half], numPos, 1)
    positions = np.zeros([numPos, 2], dtype=np.float32)

    for b in range(0, numPos):
        sc = np.squeeze(responses[b, 0, :, :])
        r = np.where(sc == np.max(sc))
        positions[b, :] = [r[0][0], r[1][0]]

    res = precisionAuc(positions, centerLabel,
                       radiusInPix * 1. / totalStride, nStep)
    res = (oldRes * m + res) / (m + batchSize)
    return res
