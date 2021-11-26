# %% Imports
import matplotlib.pyplot as plt

from toolbox.initialization import *
from toolbox.dsp import db2mag, mag2db, mel2hz, hz2mel, erb, magnitude_spectrum
from toolbox.type_conversion import np2torch, ensure_torch

import torch
import torch.nn.functional as tnnf
import torchaudio as ta
import torchaudio.functional as taf

from scipy.signal import kaiserord, lfilter, firwin, freqz
from IPython import get_ipython, InteractiveShell

fs_model = 24e3
dhasp = t.dhasp.DHASP(fs_model)

#%%
print('Analysis filterbank:')
dhasp.show_filterbank_responses('analysis', show_every=2)
dhasp.show_filterbank_joint_response('analysis')

print('Control filterbank:')
dhasp.show_filterbank_responses('control', show_every=2)
dhasp.show_filterbank_joint_response('control')