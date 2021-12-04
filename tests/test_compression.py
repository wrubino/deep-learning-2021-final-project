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
    0.1 * waveform,
    10 * waveform
])

# Assign names for plotting.
waveform_names = ['-20 dB', '+20 dB']


# %% Initialize the dhasp model

dhasp = t.dhasp.DHASP(fs_model)

# %% Get the outputs of the filters

outputs_control_filter = dhasp.apply_filter('control', waveforms)
envelopes_control_filter = t.dsp.envelope(outputs_control_filter)


# %% Get dynamic range compression gain
G = dhasp.calculate_G_comp(outputs_control_filter)


# %% Visualize
# The index of the filter and number of samples to show
idx_filter = 2
n_samples = int(3e3)

# Create the time axis for the plot
time = np.arange(n_samples) / fs_model

# Initialize a list of curve handles.
curves = list()

# Plot filter output and its envelope.
figure, axess = plt.subplots(2, 1, figsize=(12, 8))

for idx_waveform, (waveform_name, axes) \
        in enumerate(zip(waveform_names, axess.flatten())):
    curves = list()
    axes_right = axes.twinx()

    curves += axes.plot(
        time,
        outputs_control_filter[idx_waveform, idx_filter, :n_samples],
        label=f'Left axis: Output of control filter'
    )

    curves += axes.plot(
        time,
        envelopes_control_filter[idx_waveform, idx_filter, :n_samples],
        label='Left axis: Envelope'
    )

    curves += axes_right.plot(
        time,
        G[idx_waveform, idx_filter, :n_samples],
        color='red',
        label='Right axis: Compression gain [dB]'
    )

    axes.grid()
    if idx_waveform == 1:
        axes.legend(curves, [curve.get_label() for curve in curves],
                    bbox_to_anchor=(0, -0.15, 1, 0),
                    loc="upper left",
                    mode="expand",
                    ncol=len(curves),
                    frameon=False
                    )

    axes.set_title(f'Waveform: {waveform_name}, '
                   f'Filter number: {idx_filter + 1}, '
                   f'$f_c$={dhasp.f_a[idx_filter].numpy()[0]:,.0f} Hz')
    axes.set_xlabel('Time [s]')
    axes.set_ylabel('Signal value')
    axes_right.set_ylabel('Compression gain [dB]')

t.plotting.apply_standard_formatting(figure, include_grid=False)
figure.tight_layout(rect=[0, 0, 1, 1])
plt.show()
