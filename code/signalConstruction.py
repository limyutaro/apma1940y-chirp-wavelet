# functions that make signals and noise
import numpy as np


# makes a signal with a given chirp, with random start time
def makeSignalWithChirp(chirp, t, mu, sigma, Mu, Sigma, noiseInterval):
    # chirp is the chirp array
    # t is the time array, t.size should be larger than chirp.size
    # mu, sigma is the mean and s.d. of the fine noise
    # Mu, Sigma is the mean and s.d. of the coarse noise
    # noiseInterval gives the distance of the "corners" of coarse noise
    chirpStartTime = np.random.randint(0, t.size - chirp.size)
    signal = makeRandomNoise(mu, sigma, Mu, Sigma, t.size, noiseInterval) + \
        insertChirp(chirp, chirpStartTime, t.size)

    return signal, chirpStartTime

# makes signal with a given chirp with fixed start time


def insertChirp(chirp, chirpStartTime, totalTime):
    # chirp is the chirp array
    # chirpStartTime can be any integer between 0 and totalTime - chirp.size
    # totalTime is the duration of the entire signal
    leftZeroes = np.zeros(chirpStartTime)
    rightZeroes = np.zeros(totalTime - chirpStartTime - chirp.size)

    return np.hstack((leftZeroes, chirp, rightZeroes))

# makes signal with coarse and fine random noise


def makeRandomNoise(mu, sigma, Mu, Sigma, n, h):
    # mu, sigma is the mean and s.d. of the fine noise
    # Mu, Sigma is the mean and s.d. of the coarse noise
    # n is the length of desired noise array
    # h is the noiseInterval for coarse noise
    # n / h must be a power of 2
    return makeCoarseNoise(Mu, Sigma, n, h) + makeFineNoise(mu, sigma, n)


def makeCoarseNoise(Mu, Sigma, n, h):
    corners = np.random.normal(Mu, Sigma, n//h + 1)
    coarseNoise = np.zeros(n)

    # interpolation
    for i in range(corners.size - 1):
        leftIdx = h*i  # index of left point
        for j in range(h):
            # weighted average
            coarseNoise[leftIdx+j] = ((h-j)*corners[i] + j*corners[i+1]) / h

    return coarseNoise


def makeFineNoise(mu, sigma, n):
    return np.random.normal(mu, sigma, n)
