# Functions that threshold (smoothen) data, scale data and perform filtering

# packages
import numpy as np

# my functions
from waveletTransform import *
from displayData import *


def performThresholdEps(f, eps):
    # takes in digital coefficients f, decomposes, performs thresholding, recomposes it back
    coeff = decompose(f)  # decompose
    scaledCoeff = scaleCoeff(np.square(coeff))
    # put back average value c00
    scaledCoeff = np.hstack((coeff[0], scaledCoeff[1:]))

    threshold = 4 * (eps**2) * sum(np.square(scaledCoeff))
    # print('\nthrehold is ' + str(threshold))
    maxWeight = 0.1*threshold
    sumDiscarded = 0

    while sumDiscarded < 0.9*threshold:  # 0.9 as a safety mechanism
        i = 1
        updatedDiscarded = sumDiscarded

        while updatedDiscarded < threshold and i < coeff.size:
            if scaledCoeff[i] < maxWeight:
                updatedDiscarded += scaledCoeff[i]

                coeff[i] = 0
            i += 1

        maxWeight += 0.05*threshold

        # if there are no weights were removed in this traversal, we are done.
        if sumDiscarded == updatedDiscarded:
            break
        else:
            sumDiscarded = updatedDiscarded
    return recompose(coeff)


def performThresholdTheta(coeff, theta):
    trimmedCoeff = coeff.copy()
    threshold = theta * max(np.abs(coeff[1:]))
    # print('threshold is ' + str(threshold))

    for i in range(1, coeff.size):
        if np.abs(trimmedCoeff[i]) < threshold:
            trimmedCoeff[i] = 0

    return trimmedCoeff


def scaleCoeff(coeff):  # computes for each entry  2^-l * d_l,k
    scaledCoeff = coeff.copy()  # make a copy

    levels = int(np.log2(scaledCoeff.size))
    stepSize = 2**levels

    for l in range(levels):
        # print('level is ' + str(l))
        # print('stepSize ' +str(stepSize))
        i = stepSize//2

        while i < scaledCoeff.size:
            # print('we are on entry ' + str(i))
            scaledCoeff[i] *= (2**(-l))  # scale
            i += stepSize  # move to next d_l,k

        stepSize //= 2

    return scaledCoeff


def unscaleCoeff(coeff):
    unscaledCoeff = coeff.copy()  # make a copy

    levels = int(np.log2(unscaledCoeff.size))
    stepSize = 2**levels

    for l in range(levels):
        # print('level is ' + str(l))
        # print('stepSize ' + str(stepSize))
        i = stepSize//2

        while i < unscaledCoeff.size:
            # print('we are on entry' + str(i))
            unscaledCoeff[i] *= (2**l)  # scale
            i += stepSize  # move to next d_l,k

        stepSize //= 2

    return unscaledCoeff


def removeBelowLevelK(coeff, k):
    # removes all d coefficients on levels below k
    d = coeff.copy()  # make a copy

    levels = int(np.log2(d.size))
    stepSize = 2**(levels-k)

    while stepSize >= 2:
        i = stepSize//2

        while i < d.size:
            d[i] = 0
            i += stepSize

        stepSize //= 2

    return d


def removeAboveLevelL(coeff, l):
    # removes all d coefficients on levels above l
    d = removeBelowLevelK(coeff, l - 1)
    d = coeff - d

    return d


def smoothenScore(coeff, coarsestLvl, finestLvl):
    # keeps wavelet coefficients between coarsestLvla nd finestLvl inclusive
    # also removes average value c_0,0
    x = removeBelowLevelK(removeAboveLevelL(
        decompose(coeff), coarsestLvl), finestLvl)
    x[0] = 0
    thresholdedScore = recompose(x)

    return thresholdedScore


def findSpikeCoeff(coeff, threshold):
    x = coeff.copy()

    posMax = max([0, max(x)])
    negMin = min([0, min(x)])
    posKept = []
    negKept = []

    for i in range(x.size):
        d = x[i]

        if d > 0:
            if d < threshold*posMax:
                x[i] = 0
            else:
                posKept.append(d)
        elif d < 0:
            if d > threshold*negMin:
                x[i] = 0
            else:
                negKept.append(d)

    return (x, posKept, negKept)
