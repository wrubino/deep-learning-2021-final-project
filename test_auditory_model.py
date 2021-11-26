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

# %% Get the outputs of the filters
outputs_control_filter = dhasp.apply_filter('control', waveform)
outputs_analysis_filter = dhasp.apply_filter('analysis', waveform)
envelopes_control_filter = envelope(outputs_control_filter, axis=1)

# %% Apply gain
G = dhasp.calculate_G(outputs_control_filter)
output_model_per_band = outputs_analysis_filter * db2mag(G)
output_model = output_model_per_band.sum(dim=0)
envelope_model = mag2db(envelope(output_model))


# %% Visualize

# The index of the filter and number of samples to show
idx_filter_to_show = 2
n_samples_to_show = int(waveform.shape[-1])

# Create the time axis for the plot
time = np.arange(n_samples_to_show) / fs_model

# Initialize a list of curve handles.
curves = list()

# Plot filter output and its envelope.
figure, axes = plt.subplots(figsize=(12, 8))
axes_right = axes.twinx()

curves += axes.plot(
    time,
    waveform.squeeze()[:n_samples_to_show],
    label=f'Waveform'
)

curves += axes_right.plot(
    time,
    envelope_model[: n_samples_to_show],
    label='Envelope',
    color='red'
)

# curves += axes_right.plot(time,
#                           G[idx_filter_to_show, :n_samples_to_show],
#                           color='red',
#                           label='Dynamic range compression gain')

axes.legend(curves, [curve.get_label() for curve in curves])
# axes.set_title(f'Filter number {idx_filter_to_show + 1}')
axes.set_xlabel('Time [s]')
# %% Play

sd.play(output_model.squeeze(), fs_model)