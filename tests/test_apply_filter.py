# %% Imports
from toolbox.initialization import *

# %% Initialize the dhasp model.
fs_model = 24e3
dhasp = t.dhasp.DHASP(fs_model)


# %% Create a test signal (white noise)
time = torch.arange(0, 2, 1 / fs_model)
white_noise = 2 * torch.rand_like(time).unsqueeze(0) - 0.25
white_noise_20_dB_louder = 10 * white_noise
white_noise_filtered = dhasp.apply_filter('analysis', torch.vstack(
    [white_noise, white_noise_20_dB_louder]))


# %% Test the analysis filterbank on audio

# The index of the filter to apply
idx_filter = 3
for filterbank in ['analysis', 'control', 'eq']:
    white_noise_filtered = dhasp.apply_filter(filterbank, torch.vstack(
        [white_noise, white_noise_20_dB_louder]))

    if filterbank == 'analysis':
        frequencies = dhasp.f_a
    elif filterbank == 'control':
        frequencies = dhasp.f_c
    elif filterbank == 'eq':
        frequencies = dhasp.f_eq

    t.plotting.plot_magnitude_spectrum(
        {'Unfiltered, 0 dB': white_noise,
         'Filtered, 0 dB': white_noise_filtered[0, idx_filter, :],
         'Filtered, 20 dB': white_noise_filtered[1, idx_filter, :]},
        fs_model,
        title=f'Application of filter number {idx_filter + 1} '
              f'with $f_c$={frequencies.numpy().flatten()[idx_filter]:.0f} Hz '
              f'from the filterbank: "{filterbank}" on white noise'
)
