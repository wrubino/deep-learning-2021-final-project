# %% Imports
from toolbox.initialization import *

# %% Initialize the dhasp model.
fs_model = 24e3
dhasp = t.dhasp.DHASP(fs_model)


# %% Test the analysis filterbank on audio
time = torch.arange(0, 2, 1 / fs_model)
white_noise = 2 * torch.rand_like(time).unsqueeze(0) - 0.25
white_noise_20_dB_louder = 10 * white_noise
white_noise_filtered = dhasp.apply_filter('analysis', torch.vstack(
    [white_noise, white_noise_20_dB_louder]))

t.plotting.plot_magnitude_spectrum(
    {'Unfiltered, 0 dB': white_noise,
     'Filtered, 0 dB': white_noise_filtered[0, 10, :],
     'Filtered, 20 dB': white_noise_filtered[1, 10, :]},
    fs_model,
    title='Application of filter 11 from the analysis filterbank on white noise'
)
