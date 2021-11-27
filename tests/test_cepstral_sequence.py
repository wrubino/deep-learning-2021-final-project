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

# %% Initialize the dhasp model.
dhasp = t.dhasp.DHASP(fs_model)

# %% Get the output of the auditory model.
output, output_envelope = dhasp.calculate_output(waveform)

# %% Get the smoothed envelope

# Get the smoothed envelope and the stride used when smoothing.
output_envelope_smoothed, stride = \
    dhasp.smooth_output_envelope(output_envelope)

# Get the time resolution of the smoothed envelope.
t_smoothed = stride / fs_model

# %% Compute cepstral sequence
C = dhasp.calculate_C(output_envelope_smoothed,
                      from_value_type='smoothed_output_envelope')


#%% Visualize
# The index of the filter and number of samples to show
idx_filter = 2

# Create the time axis for the plot.
time_smoothed = np.linspace(
    0,
    stride / fs_model * output_envelope_smoothed.shape[1],
    output_envelope_smoothed.shape[1]
)

# Plot filter output and its envelope.
figure, axes = plt.subplots(figsize=(12, 8))
axes_right = axes.twinx()

# Initialize a list of curve handles.
curves = list()

curves += axes.plot(
    time_smoothed,
    output_envelope_smoothed[idx_filter, :],
    label='Smoothed envelope [dB]',

)

curves += axes_right.plot(
    time_smoothed,
    C[idx_filter, :],
    label=f'Cepstral sequence',
    color='red'
)

axes.legend(curves, [curve.get_label() for curve in curves])
axes.set_title(f'Filter number: {idx_filter + 1}, '
               f'$f_c$={dhasp.f_a[idx_filter].numpy()[0]:,.0f} Hz')
axes.set_xlabel('Time [s]')


