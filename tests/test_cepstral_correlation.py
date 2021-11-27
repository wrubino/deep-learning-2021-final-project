# %% Imports
from IPython import get_ipython, InteractiveShell
from toolbox.initialization import *
from toolbox.dsp import envelope, magnitude_spectrum, mag2db, db2mag, rms
from toolbox.plotting import plot_magnitude_spectrum
from toolbox.type_conversion import np2torch, ensure_torch
from typing import Union


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

# %% Initialize the dhasp model.
dhasp = t.dhasp.DHASP(fs_model)

# %% Define on which samples to do the test

# Get the first 3 seconds of the clean sample.
n_samples_3s = int(fs_model) * 3

# Define the speech variants to compare
speech_variants = ['clean', 'babble']

# %% Compute cepstral sequence for each variant
# Initialize the dict where the cepstral sequences will be stored.
C = dict()
E = dict()

# Calculate the cepstral sequences
for variant_name in speech_variants:

    waveform = audio_samples[variant_name][:n_samples_3s]
    _, output_envelope = dhasp.calculate_output(waveform)
    smoothed_envelope, stride = dhasp.smooth_output_envelope(output_envelope)
    E[variant_name] = smoothed_envelope
    C[variant_name] = dhasp.calculate_C(
        smoothed_envelope,
        from_value_type='smoothed_output_envelope'
    )

# %% Calculate the correlation
R = (
        (C[speech_variants[0]] * C[speech_variants[1]]).sum(dim=1)
        / (
                torch.linalg.norm(C[speech_variants[0]], dim=1)
                * torch.linalg.norm(C[speech_variants[1]], dim=1)
        )
)

# %% Calculate the envelope loss

L_e = E[speech_variants[1]] - E[speech_variants[0]]
L_e[L_e < 0] = 0
L_e = L_e.sum(dim=1)


# %% Calculate the total loss
alpha = 1e-4
L = -R.mean() + alpha * L_e.sum()



# %% Visualize
# # The index of the filter and number of samples to show
# idx_filter = 2
#
# # Create the time axis for the plot.
# # time_original = np.arange(envelope_model.shape[1]) / fs_model
# time_smoothed = np.linspace(
#     0,
#     stride / fs_model * output_envelope_smoothed.shape[1],
#     output_envelope_smoothed.shape[1]
# )
#
# # Plot filter output and its envelope.
# figure, axes = plt.subplots(figsize=(12, 8))
# axes_right = axes.twinx()
#
# # Initialize a list of curve handles.
# curves = list()
#
# curves += axes.plot(
#     time_smoothed,
#     output_envelope_smoothed[idx_filter, :],
#     label='Smoothed envelope [dB]',
#
# )
#
# curves += axes_right.plot(
#     time_smoothed,
#     C[idx_filter, :],
#     label=f'Cepstral sequence',
#     color='red'
# )
#
# axes.legend(curves, [curve.get_label() for curve in curves])
# axes.set_title(f'Filter number: {idx_filter + 1}, '
#                f'$f_c$={dhasp.f_a[idx_filter].numpy()[0]:,.0f} Hz')
# axes.set_xlabel('Time [s]')
#
#
