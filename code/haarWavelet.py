# decomposition and recomposition for Haar Wavelet
import numpy as np


def decompose_haar(f):
    coeff = f.copy()  # make a copy

    levels = int(np.log2(coeff.size))
    # print(levels)
    stepSize = 1

    for i in range(levels):
        coeff = decomposeOnce_haar(coeff, stepSize)
        stepSize *= 2

    return coeff


def decomposeOnce_haar(coeff, stepSize):
    i = 0

    while i < coeff.size:
        coeff[i] += coeff[i+stepSize]
        coeff[i+stepSize] *= -2
        coeff[i+stepSize] += coeff[i]
        coeff[i] *= 0.5
        i += 2*stepSize

    return coeff


def recompose_haar(coeff):
    f_tilde = coeff.copy()  # make a copy

    levels = int(np.log2(f_tilde.size))
    stepSize = f_tilde.size // 2

    for i in range(levels):
        f_tilde = recomposeOnce_haar(f_tilde, stepSize)
        stepSize //= 2

    return f_tilde


def recomposeOnce_haar(coeff, stepSize):
    i = 0

    while i < coeff.size:
        coeff[i] += 0.5 * coeff[i+stepSize]
        coeff[i+stepSize] *= -1
        coeff[i+stepSize] += coeff[i]
        i += 2*stepSize

    return coeff
