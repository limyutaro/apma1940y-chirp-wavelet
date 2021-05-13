# Script that runs tests and collects data on the success rate of chirp detection

# packages
import numpy as np
import matplotlib.pyplot as plt

# my functions and parameters
from myParameters import *
from testAndDataCollection import *

n = 500


x = ['Missed Detection', 'Wrong Time', 'Correct Detection']
dataPresent = collectDataWithChirp(
    n, chirp, t, mu, sigma, Mu, Sigma, noiseInterval)
print(dataPresent/n)
plt.figure()
plt.bar(x, dataPresent/n)
plt.ylabel("Fraction of trials")
plt.ylim([0, 1])


y = ['False Positive', 'Correct Detection']
dataAbsent = collectDataWithoutChirp(
    n, chirp, t, mu, sigma, Mu, Sigma, noiseInterval)
print(dataAbsent/n)
plt.figure()
plt.bar(y, dataAbsent/n)
plt.ylabel("Fraction of trials")
plt.ylim([0, 1])


plt.show()
