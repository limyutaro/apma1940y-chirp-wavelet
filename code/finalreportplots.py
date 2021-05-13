# Script that plots and saves figure generated for the final report

# packages
import numpy as np
import matplotlib.pyplot as plt
import random

# my functions and parameters
from myParameters import *
from waveletTransform import *
from signalConstruction import *
from signalAnalysis import *


#   #   #   #   #        SET TO APPROPRIATE FILEPATH    #   #   #   #   #   #
filepath = '/Users/loser/Documents/_APMA1940Y/finalproject/images/'
#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #


# # # # # # # Seed
# # obvious V-shape in score: 1, 8, 12
# # good spike: 14, 22, 26, 27
np.random.seed(27)

# # # # # # # Create noisy signal with chirp
coarseNoise = makeCoarseNoise(Mu, Sigma, t.size, noiseInterval)
fineNoise = makeFineNoise(mu, sigma, t.size)
w = coarseNoise + fineNoise
c = insertChirp(chirp, t.size//2, t.size)
f = c + w

# # # # # # # Compute score and spike coefficients
score = computeScore(f, chirp)
thresholdedScore, spike, _, _ = windowAnalysis(f, chirp)
thresholdedScoreDecomp = decompose(thresholdedScore)


# # # # # # # # # # # # # # # # #  chirp.png  # # # # # # # # # # # # # # # # #
plt.figure(figsize=(9, 3))
plt.title('Chirp')
plt.plot(t, c)
plt.xlabel('time')
plt.savefig(filepath + 'chirp.png', bbox_inches='tight')


# # # # # # # # # # # # # # # # #  noise.png  # # # # # # # # # # # # # # # # #
fig, axs = plt.subplots(3, sharex=True, sharey=True, constrained_layout=True)
# fig.suptitle('Noise and its components')
axs[0].set_title('total noise')
axs[1].set_title('coarse noise')
axs[2].set_title('fine noise')

axs[0].plot(t, w)
axs[1].plot(t, coarseNoise)
axs[2].plot(t, fineNoise)
plt.xlabel('time')
# Hide x labels and tick labels for all but bottom plot.
for ax in axs:
    ax.label_outer()
plt.savefig(filepath + 'noise.png', bbox_inches='tight')


# # # # # # # # # # # # # # # # noisychirp.png # # # # # # # # # # # # # # # # #
fig, axs = plt.subplots(2, sharex=True, figsize=(9, 4))
fig.suptitle('Chirp masked in noise')
axs[0].plot(t, c)
axs[1].plot(t, f)
plt.xlabel('time')
plt.savefig(filepath + 'noisychirp.png', bbox_inches='tight')


# # # # # # # # # # # # # # # noisychirpdecomp.png # # # # # # # # # # # # # # #
fig, axs = plt.subplots(2, sharex=True, sharey=True,
                        figsize=(9, 5), constrained_layout=True)
fig.suptitle('Wavelet coefficients')
axs[0].set_title('chirp')
axs[1].set_title('raw signal')

axs[0].plot(t, decompose(c), 'tab:orange')
axs[1].plot(t, decompose(f), 'tab:orange')
plt.xlabel('time')
plt.savefig(filepath + 'noisychirpdecomp.png', bbox_inches='tight')


# # # # # # # # # # # # # # # # #  score.png  # # # # # # # # # # # # # # # # #
fig, axs = plt.subplots(2, sharex=True, figsize=(9, 5),
                        constrained_layout=True)
axs[0].set_title('score')
axs[1].set_title('thresholded score')
axs[0].plot(t, score, 'tab:green')
axs[1].plot(t, thresholdedScore, 'tab:green')
plt.xlabel('time')
plt.savefig(filepath + 'score.png', bbox_inches='tight')


# # # # # # # # # # # # # # # # #  spike.png  # # # # # # # # # # # # # # # # #
scoreDecomp = decompose(score)
fig, axs = plt.subplots(3, sharex=True, constrained_layout=True)
# fig.suptitle('Relating score and spike')
axs[0].set_title('score')
axs[1].set_title('wavelet coefficients of score')
axs[2].set_title('spike')

axs[0].plot(t, thresholdedScore, 'tab:green')
axs[1].plot(t, thresholdedScoreDecomp, 'tab:red')
axs[2].plot(t, spike, 'tab:red')
plt.xlabel('time')
plt.savefig(filepath + 'spike.png', bbox_inches='tight')


# # # # # # # # # # # # # # anomalouscomparison.png # # # # # # # # # # # # # #
noisescore, noisespike, _, _ = windowAnalysis(w, chirp)

fig, axs = plt.subplots(2, 2, sharex='col', sharey='row',
                        figsize=(9, 5), constrained_layout=True)
fig.suptitle('Score and spike affected by presence of chirp')
axs[0, 0].set_title('score (chirp present)')
axs[0, 1].set_title('score (chirp absent)')
axs[1, 0].set_title('spike (chirp present)')
axs[1, 1].set_title('spike (chirp absent)')

axs[0, 0].plot(t, thresholdedScore, 'tab:green')
axs[1, 0].plot(t, spike, 'tab:red')
axs[0, 1].plot(t, noisescore, 'tab:green')
axs[1, 1].plot(t, noisespike, 'tab:red')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
plt.savefig(filepath + 'anomalouscomparison.png', bbox_inches='tight')


# # # # # # # # # # # # # # # chirppresentscores.png # # # # # # # # # # # # # #
scores = np.zeros((6, t.size))
spikes = np.zeros((6, t.size))
for i in range(6):
    # make signal
    signal, _ = makeSignalWithChirp(
        chirp, t, mu, sigma, Mu, Sigma, noiseInterval)
    score, spike, _, _ = windowAnalysis(signal, chirp)
    scores[i, :] = score
    spikes[i, :] = spike

fig, axs = plt.subplots(3, 2, sharex=True, sharey=True,
                        figsize=(9, 5), constrained_layout=True)
fig.suptitle('Chirp present scores')
axs[0, 0].set_title('P1')
axs[0, 1].set_title('P2')
axs[1, 0].set_title('P3')
axs[1, 1].set_title('P4')
axs[2, 0].set_title('P5')
axs[2, 1].set_title('P6')

axs[0, 0].plot(t, scores[0, :], 'tab:green')
axs[0, 1].plot(t, scores[1, :], 'tab:green')
axs[1, 0].plot(t, scores[2, :], 'tab:green')
axs[1, 1].plot(t, scores[3, :], 'tab:green')
axs[2, 0].plot(t, scores[4, :], 'tab:green')
axs[2, 1].plot(t, scores[5, :], 'tab:green')
plt.ylim([-10, 10])

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
plt.savefig(filepath + 'chirppresentscores.png', bbox_inches='tight')


# # # # # # # # # # # # # # # chirppresentspike.png # # # # # # # # # # # # # #
fig, axs = plt.subplots(3, 2, sharex=True, sharey=True,
                        figsize=(9, 5), constrained_layout=True)
fig.suptitle('Chirp present spikes')
axs[0, 0].set_title('Pa')
axs[0, 1].set_title('Pb')
axs[1, 0].set_title('Pc')
axs[1, 1].set_title('Pd')
axs[2, 0].set_title('Pe')
axs[2, 1].set_title('Pf')

axs[0, 0].plot(t, spikes[0, :], 'tab:red')
axs[0, 1].plot(t, spikes[1, :], 'tab:red')
axs[1, 0].plot(t, spikes[2, :], 'tab:red')
axs[1, 1].plot(t, spikes[3, :], 'tab:red')
axs[2, 0].plot(t, spikes[4, :], 'tab:red')
axs[2, 1].plot(t, spikes[5, :], 'tab:red')
plt.ylim([-10, 10])

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
plt.savefig(filepath + 'chirppresentspike.png', bbox_inches='tight')


# # # # # # # # # # # # # # # chirpabsentscores.png # # # # # # # # # # # # # #
scores = np.zeros((6, t.size))
spikes = np.zeros((6, t.size))
for i in range(6):
    # make signal
    signal = makeRandomNoise(mu, sigma, Mu, Sigma, t.size, noiseInterval)
    score, spike, _, _ = windowAnalysis(signal, chirp)
    scores[i, :] = score
    spikes[i, :] = spike

fig, axs = plt.subplots(3, 2, sharex=True, sharey=True,
                        figsize=(9, 5), constrained_layout=True)
fig.suptitle('Chirp absent scores')
axs[0, 0].set_title('A1')
axs[0, 1].set_title('A2')
axs[1, 0].set_title('A3')
axs[1, 1].set_title('A4')
axs[2, 0].set_title('A5')
axs[2, 1].set_title('A6')

axs[0, 0].plot(t, scores[0, :], 'tab:green')
axs[0, 1].plot(t, scores[1, :], 'tab:green')
axs[1, 0].plot(t, scores[2, :], 'tab:green')
axs[1, 1].plot(t, scores[3, :], 'tab:green')
axs[2, 0].plot(t, scores[4, :], 'tab:green')
axs[2, 1].plot(t, scores[5, :], 'tab:green')
plt.ylim([-10, 10])

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
plt.savefig(filepath + 'chirpabsentscores.png', bbox_inches='tight')


# # # # # # # # # # # # # # # chirpabsentspike.png # # # # # # # # # # # # # # #
fig, axs = plt.subplots(3, 2, sharex=True, sharey=True,
                        figsize=(9, 5), constrained_layout=True)
fig.suptitle('Chirp absent spikes')
axs[0, 0].set_title('Aa')
axs[0, 1].set_title('Ab')
axs[1, 0].set_title('Ac')
axs[1, 1].set_title('Ad')
axs[2, 0].set_title('Ae')
axs[2, 1].set_title('Af')

axs[0, 0].plot(t, spikes[0, :], 'tab:red')
axs[0, 1].plot(t, spikes[1, :], 'tab:red')
axs[1, 0].plot(t, spikes[2, :], 'tab:red')
axs[1, 1].plot(t, spikes[3, :], 'tab:red')
axs[2, 0].plot(t, spikes[4, :], 'tab:red')
axs[2, 1].plot(t, spikes[5, :], 'tab:red')
plt.ylim([-10, 10])

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
plt.savefig(filepath + 'chirpabsentspike.png', bbox_inches='tight')


# # # # # # # # # # # # # # # misseddetection.png # # # # # # # # # # # # # # #

plt.show()
