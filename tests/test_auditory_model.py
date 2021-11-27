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


# %% Apply gain
output_model, envelope_model = dhasp.calculate_output(waveform)


# %% Visualize

# The index of the filter and number of samples to show
idx_filter = 2
n_samples = int(waveform.shape[-1])

# Create the time axis for the plot
time = np.arange(n_samples) / fs_model

# Initialize a list of curve handles.
curves = list()

# Plot filter output and its envelope.
figure, axes = plt.subplots(figsize=(12, 8))
axes_right = axes.twinx()

curves += axes.plot(
    time,
    output_model.squeeze()[idx_filter, :n_samples],
    label=f'Output of the auditory model'
)

curves += axes_right.plot(
    time,
    envelope_model[idx_filter, :n_samples],
    label='Envelope of the output [dB]',
    color='red'
)

axes.legend(curves, [curve.get_label() for curve in curves])
axes.set_title(f'Filter number: {idx_filter + 1}, '
               f'$f_c$={dhasp.f_a[idx_filter].numpy()[0]:,.0f} Hz')
axes.set_xlabel('Time [s]')