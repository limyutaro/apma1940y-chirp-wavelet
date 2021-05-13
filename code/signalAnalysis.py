# Functions that find chirp in a given signal, computes scores and spikes.

# packages
import numpy as np
from scipy import stats

# my functions
from waveletTransform import *
from signalConstruction import *
from filterData import *


def windowAnalysis(signal, chirp):
    # score thresholded, with high and low levels removed. average set to 0.
    score = computeScore(signal, chirp)
    thresholdedScore = smoothenScore(score, 4, 8)
    spike, posKept, negKept = findSpikeCoeff(decompose(thresholdedScore), 0.5)

    return thresholdedScore, spike, posKept, negKept


def computeScore(signal, chirp):
    signalDecomp = decompose(signal)
    chirpDecomp = decompose(chirp)

    score = np.ones(signal.size)

    for t in range(signal.size - chirp.size + 1):
        data = signalDecomp[t:t+chirp.size]
        scoreHere = 0  # higher score is worse

        for i in range(chirp.size):
            # if data and expected chirp are of opposite polarity,
            #   then we penalize the score (by increasing variable score)
            pdt = data[i] * chirpDecomp[i]
            if pdt <= 0:
                scoreHere += np.abs(pdt)

        score[t] = scoreHere

    return score


def findChirp(signal, chirp, sigma):
    scores, spikes, _, _ = windowAnalysis(signal, chirp)
    # ignore the last few entries of chirp
    score = scores[:-chirp.size or None]
    spike = spikes[:-chirp.size or None]

    # initialize
    chirpDetected = False
    indexDetected = np.inf

    # search for anomalies and the largest positive and negative spikes
    windowRatio = 5
    posAnomalyPresent, maxIdx = posSpikeAnomaly(spike, chirp.size, windowRatio)
    negAnomalyPresent, minIdx = negSpikeAnomaly(spike, chirp.size, windowRatio)

    # check for anomalous spikes
    posSpikeSize = np.abs(max(spike))
    negSpikeSize = np.abs(min(spike))

    if (negSpikeSize > posSpikeSize and negSpikeSize > 6):
        chirpDetected = True
        indexDetected = minIdx
    elif (posSpikeSize > negSpikeSize and posSpikeSize > 6):
        chirpDetected = True
        indexDetected = maxIdx
    elif negAnomalyPresent:
        chirpDetected = True
        indexDetected = minIdx
    elif posAnomalyPresent:
        chirpDetected = True
        indexDetected = maxIdx

    return chirpDetected, indexDetected


def posSpikeAnomaly(coeff, chirpLength, windowRatio):
    globalMaxIdx = np.argmax(coeff)
    globalMax = coeff[globalMaxIdx]
    secondMax = -np.inf

    for i in range(coeff.size):
        leftBound = globalMaxIdx - chirpLength/windowRatio
        rightBound = globalMaxIdx + chirpLength/windowRatio
        if i < leftBound or rightBound < i:
            x = coeff[i]
            if x > secondMax:
                secondMax = x

    return 0.6 * np.abs(globalMax) > np.abs(secondMax), globalMaxIdx


def negSpikeAnomaly(coeff, chirpLength, windowRatio):
    globalMinIdx = np.argmin(coeff)
    globalMin = coeff[globalMinIdx]
    secondMin = np.inf
    leftBound = globalMinIdx - chirpLength/windowRatio
    rightBound = globalMinIdx + chirpLength/windowRatio

    for i in range(coeff.size):

        if i < leftBound or rightBound < i:
            x = coeff[i]
            if x < secondMin:
                secondMin = x

    return 0.6 * np.abs(globalMin) > np.abs(secondMin), globalMinIdx


# # # # # # # # # # # # # # # UNUSED FUNCTIONS # # # # # # # # # # # # # # #

def sparsityCount(spike, idxDetected, chirpLength, windowRatio):
    posCount = 0
    negCount = 0

    leftBound = idxDetected - chirpLength/windowRatio
    rightBound = idxDetected + chirpLength/windowRatio

    for i in range(spike.size):
        if i < leftBound or rightBound < i:
            if spike[i] > 0:
                posCount += 1
            elif spike[i] < 0:
                negCount += 1

    return posCount, negCount


def checkLargeCoeffSparse(scoreDecomp, idxDetected, chirpLength, windowRatio, threshold):
    posCount, negCount = sparsityCount(
        scoreDecomp, idxDetected, chirpLength, windowRatio)

    topIsSparse = posCount <= threshold
    bottomIsSparse = negCount <= threshold

    return (topIsSparse, bottomIsSparse)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
