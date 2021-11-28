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
    10 * waveform
])

# Assign names for plotting.
waveform_names = ['0 dB', '+20 dB']


# %% Initialize the dhasp model
dhasp = t.dhasp.DHASP(fs_model)


# %% Get the output of the auditory model
output_auditory_model, envelope_auditory_model = \
    dhasp.calculate_output(waveforms)


# %% Visualize

# The index of the filter and number of samples to show
idx_filter = 2
n_samples = int(waveform.shape[-1])

# Create the time axis for the plot
time = np.arange(n_samples) / fs_model

# Create figure and axes for the plots.
figure, axess = plt.subplots(2, 1, figsize=(12, 8))

for idx_waveform, (waveform_name, axes) \
        in enumerate(zip(waveform_names, axess.flatten())):

    # Initialize a list of curve handles.
    curves = list()

    # Plot filter output and its envelope.
    axes_right = axes.twinx()

    curves += axes.plot(
        time,
        waveforms[idx_waveform, :n_samples],
        label=f'Original waveform'
    )

    curves += axes.plot(
        time,
        output_auditory_model[idx_waveform, idx_filter, :n_samples],
        label=f'Output of the auditory model'
    )

    curves += axes_right.plot(
        time,
        envelope_auditory_model[idx_waveform, idx_filter, :n_samples],
        label='Envelope of the output [dB]',
        color='red'
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

    axes.set_title(
        f'Waveform: {waveform_name}, '
        f'Filter number: {idx_filter + 1}, '
        f'$f_c$={dhasp.f_a[idx_filter].numpy()[0]:,.0f} Hz')
    axes.set_xlabel('Time [s]')

t.plotting.apply_standard_formatting(figure, include_grid=False)
figure.tight_layout(rect=[0, 0, 1, 1])
plt.show()
