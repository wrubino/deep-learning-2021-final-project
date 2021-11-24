import datetime
import numpy as np
import toolbox.dsp

from toolbox.initialization import *
from toolbox.dsp import db2mag, mag2db, mel2hz, hz2mel, erb, magnitude_spectrum




# %% Analysis filterbank
# Eq 1 in resources/literature/DIFFERENTIABLE HEARING AID SPEECH PROCESSING.pdf

# Number of filter frequencies
I = 32

# The centre frequencies of the analysis filterbank f_a are in the Mel scale
# covering the range from 80 Hz to 8 kHz:
f_a = mel2hz(np.linspace(
    hz2mel(80),
    hz2mel(8000),
    I
))

# Hearing loss for outer cells at f_a:
HL = np.zeros_like(f_a)

# Maximum hearing loss is set to 50
HL[HL > 50] = 50

# The bandwidths b_NH_a are in the equivalent rectangular bandwidth (ERB)
# scale for the normal hearing model:
b_NH_a = erb(f_a)

# The bandwidths b_NH a are in the equivalent rectangular bandwidth (ERB)
# scale for the normal hearing model. To approximate the behaviour that the
# auditory filter bandwidths increase along with the hearing loss the
# bandwidths b_HL_a of the hearing loss model is expressed as:
b_HL_a = (1
          + HL / 50
          + 2 * np.power(HL / 50, 6)
          ) * b_NH_a

# Amplitude required to normalise the frequency response of the filter
A_a = np.ones(32)

# Filter order
N = 4


# Transfer function of the analysis filterbank
def calculate_h_a(
        n_taps,
        N,
        f_a,
        b_a,
        fs):

    t = np.arange(n_taps) * 1 / fs
    I = len(f_a)
    h = np.zeros((len(f_a), len(t)))

    # calculate h for each filter

    for i in range(I):
        h_i = (np.power(t, N - 1)
               * np.exp(-2 * np.pi * b_a[i] * t)
               * np.cos(2 * np.pi * f_a[i] * t))

        # Normalize so that peak gain = !
        w = 2 * np.pi * f_a[i]
        scale = np.linalg.norm(np.array([
            np.sum(h_i * np.cos(w * t)),
            np.sum(h_i * np.sin(w * t))
            ]))

        h[i, :] = h_i / scale

    return h