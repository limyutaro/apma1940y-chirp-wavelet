# Functions for wavelet decomposition and recomposition
# import other wavelets and insert them into the respective functions if needed

# packages
import numpy as np

# my functions
from haarWavelet import *


def decompose(f):
    coeff = decompose_haar(f)

    return coeff


def recompose(coeff):
    f = recompose_haar(coeff)

    return f
