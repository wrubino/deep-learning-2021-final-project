# %% Imports
import torch
import torchaudio as ta
import torchaudio.functional as taf

from toolbox.initialization import *
from IPython import get_ipython, InteractiveShell

import torch.nn.functional as nnf
import torch.nn as nn

from toolbox.type_conversion import ensure_torch, np2torch


# %%
def test_fun(fn, x):
    inputs = {'np': x,
              'torch': np2torch(x)}

    out = dict()
    for var_type, var in inputs.items():
        print()
        print(f'{var_type}: ')
        out[var_type] = fn(var)
        display(out[var_type])

    return out


# %%
audio, fs = t.data_loading.get_waveforms_synthetic_speech(dict(
    speaker='joanna',
    length='5s',
    segment=10
))
fs_model = 24e3

audio_np = resample(audio['clean'], fs, fs_model)
out_rms = test_fun(t.dsp.rms, audio_np)

#%%
out_db2mag = test_fun(t.dsp.db2mag, np.array([20, 6]))

#%%
out_mag2db = test_fun(t.dsp.mag2db, np.array([0.0001, 10]))

#%%
out_mel2hz = test_fun(t.dsp.mel2hz, np.array([10, 500]))

#%%
out_hz2mel = test_fun(t.dsp.hz2mel, np.array([10, 500]))

#%%
out_erb = test_fun(t.dsp.erb, np.array([10, 500]))

#%%
out_envelope = test_fun(t.dsp.envelope, audio_np[:100])
