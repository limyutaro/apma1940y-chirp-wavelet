# Script to run tests

# my functions and parameters
from myParameters import *
from testAndDataCollection import *

n = 10  # number of tests

# comment/uncomment accordingly
runTestsChirpPresentTerminal(n, chirp, t, mu, sigma, Mu, Sigma, noiseInterval)
# runTestsWithoutChirpTerminal(n, chirp, t, mu, sigma, Mu, Sigma, noiseInterval)
