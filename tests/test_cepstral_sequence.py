# %% Imports

from toolbox.initialization import *

# %% Load Audio

# Load audio samples representing the same speech segment in different
# acoustic conditions.
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

# %% Initialize the dhasp model.
dhasp = t.dhasp.DHASP(fs_model)

# %% Get the output of the auditory model.
output, output_envelope = dhasp.calculate_output(waveforms)

# %% Get the smoothed envelope

# Get the smoothed envelope and the stride used when smoothing.
output_envelope_smoothed, stride = \
    dhasp.smooth_output_envelope(output_envelope)

# %% Compute cepstral sequence
C = dhasp.calculate_C(
    output_envelope_smoothed,
    from_value_type='smoothed_output_envelope'
)


# %% Visualize
# The index of the filter and number of samples to show
idx_filter = 10
j = 5

# Create the time axis for the plot.
time_smoothed = \
    np.arange(output_envelope_smoothed.shape[-1]) / fs_model * stride

# Create figure and axes for the plots.
figure, axess = plt.subplots(2, 1, figsize=(12, 8))

for idx_waveform, (waveform_name, axes) \
        in enumerate(zip(waveform_names, axess.flatten())):

    axes_right = axes.twinx()

    # Initialize a list of curve handles.
    curves = list()

    curves += axes.plot(
        time_smoothed,
        output_envelope_smoothed[idx_waveform, idx_filter, :],
        label='Left axis: Smoothed envelope [dB]',
    )

    curves += axes_right.plot(
        time_smoothed,
        C[idx_waveform, j - 2, :],
        label=f'Right axis: Cepstral sequence for j = {j - 2}',
        color='red'
    )

    curves += axes_right.plot(
        time_smoothed,
        C[idx_waveform, j - 2 + 1, :],
        label=f'Right axis: Cepstral sequence for j = {j + 1}',
        color='green'
    )

    # # Show grid
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
        f'$f_c$={dhasp.f_a[idx_filter].numpy()[0]:,.0f} Hz'
    )
    axes.set_xlabel('Time [s]')

t.plotting.apply_standard_formatting(figure, include_grid=False)
figure.tight_layout(rect=[0, 0, 1, 1])
plt.show()