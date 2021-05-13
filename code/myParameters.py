# parameters that will be common to all files.
import numpy as np


# # # # # # # # # # EXAMPLE 1, CHIRP SIGNAL # # # # # # # # # # #

# # # # # # TIME PARAMETERS
l = 10  # number of levels to expect in the wavelet decomposition
t_chirp = np.linspace(-3, -1, 2**8, endpoint=False)
t = np.linspace(0, 8, 2**l, endpoint=False)


# # # # # # SCALING FUNCTIONS
# # # gaussian
g_1 = -1  # gives center of gaussian
g_2 = 1  # width of gaussian
g_3 = 2  # steepness of gaussian, choose even numbers
gaussianScaling = np.exp(-((t_chirp-g_1)/g_2)**g_3)

# # # hyperbolic
# h_1 = 1  # height of hyperbolic
# h_2 = 2  # steepness of hyperbolic
# hyperbolicScaling = h_1 / (t_chirp - 3)**h_2


# # # # # # CHIRP
amplitude = 1  # height of chirp
chirp = amplitude*np.cos(100/t_chirp) * gaussianScaling
# c_1 = 5  # frequency of chirp
# c_2 = -2  # translation of chirp
# oldchirp = amplitude*np.cos((c_1*t_chirp - c_2)**2) * gaussianScaling


# # # # # # NOISE PARAMETERS
ratio = 2
mu, sigma = 0, amplitude/4  # param for fine noise
Mu, Sigma = 0, ratio*amplitude  # param for coarse noise
noiseInterval = 2**4  # FIX AT 2^4 for project. interval for coarse noise
