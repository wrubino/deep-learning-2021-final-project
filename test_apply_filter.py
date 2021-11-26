# %% Imports
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as tnnf
import torchaudio as ta
import torchaudio.functional as taf

from IPython import get_ipython, InteractiveShell
from toolbox.initialization import *
from toolbox.type_conversion import np2torch, ensure_torch
from scipy.signal import kaiserord, lfilter, firwin, freqz

fs_model = 24e3
dhasp = t.dhasp.DHASP(fs_model)


# %% Test the analysis filterbank on audio
time = torch.arange(0, 2, 1 / fs_model)
white_noise = 2 * torch.rand_like(time).unsqueeze(0) - 0.25
white_noise_filtered = dhasp.apply_filter('analysis', white_noise)

t.plotting.plot_magnitude_spectrum(
    {'Unfiltered': white_noise,
     'Filtered': white_noise_filtered[10, :]},
    fs_model
)
