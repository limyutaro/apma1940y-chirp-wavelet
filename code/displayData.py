# Functions that display plots and coefficients

# packages
import numpy as np
import matplotlib.pyplot as plt

# my functions
from waveletTransform import *
from signalConstruction import *


def printCoeffByLevel(coeff):
    # function that prints wavelet coefficeint by level in the terminal
    print('\nCoefficients by level:\n')

    print('Average: [' + str(np.round(coeff[0], 3)) + ']\n')

    levels = int(np.log2(coeff.size))
    stepSize = 2**levels

    for i in range(levels):  # start from coarsest scale, i.e. lowest level
        j = stepSize//2

        coeffToPrint = np.array([])
        while j < coeff.size:
            coeffToPrint = np.append(coeffToPrint, np.round(coeff[j], 3))
            j += stepSize

        print('Level ' + str(i) + ': ' + str(coeffToPrint) + '\n')
        stepSize //= 2


def plotTwoWithDecomp(x, f1, title1, f2, title2):
    # function plots signals f1 and f2 next to their respective wavelet decompositions
    t1decomp = title1 + ' decomp'
    t2decomp = title2 + ' decomp'

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(x, f1)
    axs[0, 0].set_title(title1)
    axs[0, 1].plot(x, decompose(f1), 'tab:orange')
    axs[0, 1].set_title(t1decomp)
    axs[1, 0].plot(x, f2, 'tab:green')
    axs[1, 0].set_title(title2)
    axs[1, 1].plot(x, decompose(f2), 'tab:red')
    axs[1, 1].set_title(t2decomp)

    for ax in axs.flat:
        ax.set(xlabel='time', ylabel='amplitude')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()


def plotTwoVert(x1, y1, title1, x2, y2, title2):
    # plots two vertically stacked
    fig, axs = plt.subplots(2)
    axs[0].plot(x1, y1)
    axs[0].set_title(title1)
    axs[1].plot(x2, y2)
    axs[1].set_title(title2)
