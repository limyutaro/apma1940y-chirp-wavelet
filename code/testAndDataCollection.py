# Functions that runs tests on the terminal, plots test cases that should be
# flagged, and collects data on the success rate of the algorithm

# packages
import numpy as np
import matplotlib.pyplot as plt
import random

# my functions
from signalConstruction import *
from signalAnalysis import *
from displayData import *


# # # # # # # # # # # # tests on the terminal # # # # # # # # # # # #

# # # # # # chirp present # # # # # #
def runTestsChirpPresentTerminal(n, chirp, t, mu, sigma, Mu, Sigma, noiseInterval):
    correctDetection = np.zeros(n)
    wrongTime = np.zeros(n)

    for i in range(n):
        mySeed = np.random.randint(0, 10000000)
        np.random.seed(mySeed)

        # make signal
        signal, chirpStartTime = makeSignalWithChirp(
            chirp, t, mu, sigma, Mu, Sigma, noiseInterval)

        # search for chirp
        chirpDetected, timeDetected = findChirp(signal, chirp, Sigma)

        # checks if results of algorithm is reasonable
        if chirpDetected:
            leftBound = chirpStartTime - chirp.size/5
            rightBound = chirpStartTime + chirp.size/5
            correctTime = leftBound < timeDetected and timeDetected < rightBound

            # update data
            if chirpDetected and correctTime:
                correctDetection[i] = 1
            elif chirpDetected and not correctTime:
                wrongTime[i] = 1

        # for wrong times or missed detections
        if wrongTime[i] or not chirpDetected:
            print('Seed: ' + str(mySeed))

            if wrongTime[i]:
                # wrong time
                printWrongTimeMessage(i, t, timeDetected, chirpStartTime)
                plotWrongTime(i, signal, chirp, t)
            else:
                # missed detection
                printMissedDectMessage(i, t, chirpStartTime)
                plotMissedDect(i, signal, chirp, t, chirpStartTime)

    print('Success rate is: ' + str(sum(correctDetection)/n))
    print('Rate of wrong time detected is: ' + str(sum(wrongTime)/n) + '\n')

    plt.show()


def plotWrongTime(i, signal, chirp, t):
    # recompute the analysis on signal and pure noise
    score, spike, _, _ = windowAnalysis(signal, chirp)

    # plot
    fig, axs = plt.subplots(2, sharex=True, sharey=True,
                            constrained_layout=True)
    fig.suptitle('Test ' + str(i))
    axs[0].set_title('signal score')
    axs[1].set_title('signal spike')

    axs[0].plot(t, score, 'tab:green')
    axs[1].plot(t, spike, 'tab:red')
    plt.xlabel('time')


def printWrongTimeMessage(i, t, timeDetected, chirpStartTime):
    testnumber = 'Test ' + str(i) + ': '
    s1 = 'Time detected was ' + str(t[timeDetected]) + '. '
    s2 = 'But it was actually ' + str(t[chirpStartTime]) + '. '
    print(testnumber + s1 + s2)


def plotMissedDect(i, signal, chirp, t, chirpStartTime):
    # recompute the analysis on signal and pure noise
    score, spike, _, _ = windowAnalysis(signal, chirp)

    noise = signal - insertChirp(chirp, chirpStartTime, t.size)
    noiseScore, noiseSpike, _, _ = windowAnalysis(noise, chirp)

    # plot
    fig, axs = plt.subplots(4, sharex=True, sharey=True,
                            constrained_layout=True)
    fig.suptitle('Test ' + str(i))
    axs[0].set_title('signal score')
    axs[1].set_title('signal spike')
    axs[2].set_title('noise score')
    axs[3].set_title('noise spike')

    axs[0].plot(t, score, 'tab:green')
    axs[1].plot(t, spike, 'tab:red')
    axs[2].plot(t, noiseScore, 'tab:green')
    axs[3].plot(t, noiseSpike, 'tab:red')
    plt.xlabel('time')


def printMissedDectMessage(i, t, chirpStartTime):
    testnumber = 'Test ' + str(i) + ': '
    s1 = 'Chirp not detected, '
    s2 = 'but there was one at ' + str(t[chirpStartTime]) + '.'
    print(testnumber + s1 + s2)


# # # # # # chirp absent # # # # # #
def runTestsWithoutChirpTerminal(n, chirp, t, mu, sigma, Mu, Sigma, noiseInterval):
    # n is number of trials
    # chirp is the signal that we are detecting for
    # t is the time array
    falsePositives = np.zeros(n)

    for i in range(n):
        # make signal
        signal = makeRandomNoise(mu, sigma, Mu, Sigma, t.size, noiseInterval)

        # search for chirp using algorithm
        chirpDetected, timeDetected = findChirp(signal, chirp, Sigma)

        # checks if false positive attained, if so, plot
        if chirpDetected:
            falsePositives[i] = 1
            printFalsePositiveMessage(i, t, timeDetected)
            plotFalsePositive(i, signal, chirp, t)

    print('False positive rate is: ' + str(sum(falsePositives)/n))

    plt.show()


def plotFalsePositive(i, signal, chirp, t):
    # recompute the analysis on signal and pure noise
    score, spike, _, _ = windowAnalysis(signal, chirp)

    # plot
    fig, axs = plt.subplots(2, sharex=True, sharey=True,
                            constrained_layout=True)
    fig.suptitle('Test ' + str(i), ' false positive')
    axs[0].set_title('noise score')
    axs[1].set_title('noise spike')

    axs[0].plot(t, score, 'tab:green')
    axs[1].plot(t, spike, 'tab:red')
    plt.xlabel('time')


def printFalsePositiveMessage(i, t, timeDetected):
    testnumber = 'Test ' + str(i) + ': '
    s1 = 'Chirp falsely detected at time ' + str(t[timeDetected]) + '. '
    print(testnumber + s1)

# # # # # # # # # # # # # # # data collection # # # # # # # # # # # # # #


def collectDataWithChirp(n, chirp, t, mu, sigma, Mu, Sigma, noiseInterval):
    # data[0] is the number of missed detections
    # data[1] is the number of wrong times
    # data[2] is the number of correct detections
    data = np.zeros(3)

    for i in range(n):
        # make signal
        signal, chirpStartTime = makeSignalWithChirp(
            chirp, t, mu, sigma, Mu, Sigma, noiseInterval)

        # search for chirp using algorithm
        chirpDetected, timeDetected = findChirp(signal, chirp, Sigma)

        leftBound = chirpStartTime - chirp.size/5
        rightBound = chirpStartTime + chirp.size/5
        correctTime = leftBound < timeDetected and timeDetected < rightBound

        if chirpDetected and correctTime:
            data[2] += 1
        elif chirpDetected:
            data[1] += 1
        else:
            data[0] += 1

    return data


def collectDataWithoutChirp(n, chirp, t, mu, sigma, Mu, Sigma, noiseInterval):
    # data[0] is the number of false positives
    # data[1] is the number of correct detections
    data = np.zeros(2)

    for i in range(n):
        # make signal
        signal = makeRandomNoise(mu, sigma, Mu, Sigma, t.size, noiseInterval)

        # search for chirp using algorithm
        chirpDetected, _ = findChirp(signal, chirp, Sigma)

        if chirpDetected:
            data[0] += 1
        else:
            data[1] += 1

    return data
