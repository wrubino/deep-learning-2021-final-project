# %% Imports
import matplotlib.pyplot as plt

from IPython import get_ipython, InteractiveShell
from toolbox.initialization import *
from toolbox.dsp import envelope, magnitude_spectrum

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

# Get the waveform of the clean signal
waveform = audio_samples['clean']
display(waveform)


# %% Envelope
time_to_show_s = 0.3
samples_to_show = int(time_to_show_s * fs_model)
waveform = waveform.squeeze()[:samples_to_show]
envelope_waveform = envelope(waveform).squeeze()[:samples_to_show]

# Plot filter output and its envelope.
figure, axes = plt.subplots(figsize=(12, 8))
axes.plot(np.arange(len(waveform)) / fs_model,
          waveform,
          label='Waveform')

axes.plot(np.arange(len(envelope_waveform)) / fs_model,
          envelope_waveform,
          label='Envelope')

axes.legend()
axes.set_xlabel('Time [s]')
axes.set_ylabel('Signal value')
