# %% Imports

from toolbox.initialization import *

# %% Load Audio

# Load variations of the same speech segment in different conditions.
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

# Make 2 versions, quiet and loud.
waveforms = torch.vstack([
    waveform,
    2 * waveform
])

# %% Calculate envelopes

envelopes = t.dsp.envelope(waveforms)

# %% Visualize

# Create a time axis.
time_to_show_s = 0.1
samples_to_show = int(time_to_show_s * fs_model)
time_axis = np.arange(samples_to_show) / fs_model

# Plot filter output and its envelope.
figure, axess = plt.subplots(2, 1, figsize=(12, 8))

axess[0].plot(time_axis,
              waveforms[0, :samples_to_show],
              label='Waveform, 0 dB')

axess[1].plot(time_axis,
              waveforms[1, :samples_to_show],
              label='Waveform, 6 dB')

axess[0].plot(time_axis,
              envelopes[0, :samples_to_show],
              label='Envelope, 0 dB')

axess[1].plot(time_axis,
              envelopes[1, :samples_to_show],
              label='Envelope, 6 dB')

for axes in axess.flatten():
    axes.legend()
    axes.set_xlabel('Time [s]')
    axes.set_ylabel('Signal')
    axes.grid()

axess[0].set_title('Calculation of envelope for signals with 2 '
                   'different amplitudes')

t.plotting.apply_standard_formatting(figure)
plt.show()

