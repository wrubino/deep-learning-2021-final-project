# %% Imports
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as tnnf
import torchaudio as ta
import torchaudio.functional as taf

from IPython import get_ipython, InteractiveShell
from toolbox.initialization import *
from toolbox.dsp import envelope, magnitude_spectrum, mag2db, db2mag, rms
from toolbox.plotting import plot_magnitude_spectrum
from toolbox.type_conversion import np2torch, ensure_torch
from typing import Union
from scipy.interpolate import interp1d
from scipy.signal import kaiserord, lfilter, firwin, freqz

# %% Load Audio
# Load
fs_model = 24e3
audio_samples, _ = t.data_loading.get_waveforms_synthetic_speech(
    dict(
        speaker='joanna',
        length='5s',
        segment=10
    ),
    fs_model
)

# Get the first 3 seconds of the clean sample.
waveform = audio_samples['clean'][: int(fs_model) * 3]

# %% Initialize the dhasp model
dhasp = t.dhasp.DHASP(fs_model)

# %% Get the output of the auditory model
output_model, envelope_model = dhasp.calculate_output(waveform)

#%% Smoothing
# Convert the envelope model to 32-bit float
envelope_model = envelope_model.to(torch.float32)

# Number of samples corresponding to 16 ms.
n_samples_16_ms = int(16e-3 * fs_model)
stride = n_samples_16_ms // 2

# Get the weights of the Hanning window we will use for smoothing
weights = torch.hann_window(n_samples_16_ms)
weights = weights / weights.sum()

# Smooth the envelope
envelope_smoothed = tnnf.conv1d(
    envelope_model.reshape(envelope_model.shape[0],
                           1,
                           envelope_model.shape[1]),
    weights.reshape(1, 1, -1),
    stride=stride
).squeeze(1)


#%% Visualize

# The index of the filter and number of samples to show
idx_filter = 2

# Create the time axis for the plot.
time_original = np.arange(envelope_model.shape[1]) / fs_model
time_smoothed = np.linspace(
    0,
    stride / fs_model * envelope_smoothed.shape[1],
    envelope_smoothed.shape[1]
)

# Plot filter output and its envelope.
figure, axes = plt.subplots(figsize=(12, 8))
axes_right = axes.twinx()

# Initialize a list of curve handles.
curves = list()
curves += axes.plot(
    time_original,
    envelope_model[idx_filter, :],
    label=f'Original envelope of the output of the model [dB]'
)
curves += axes.plot(
    time_smoothed,
    envelope_smoothed[idx_filter, :],
    label='Smoothed envelope [dB]',
    color='red'
)

axes.legend(curves, [curve.get_label() for curve in curves])
axes.set_title(f'Filter number: {idx_filter + 1}, '
               f'$f_c$={dhasp.f_a[idx_filter].numpy()[0]:,.0f} Hz')
axes.set_xlabel('Time')
