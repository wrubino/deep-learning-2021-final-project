# %% Imports
import time

import pandas as pd

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

# Specify the target (clean) waveform
waveform_target = audio_samples['clean']

# Get the other versions of the waveform
waveforms = torch.vstack(list(audio_samples.values()))

# Get the waveform names for plotting.
waveform_names = list(audio_samples.keys())

# %% Initialize the dhasp model.
dhasp = t.dhasp.DHASP(fs_model)

# % Get C and E for target and the other waveforms

# Get the cepstral sequences and the smoothed envelope of the output of the
# auditory model
C_r, E_r = dhasp.calculate_C(waveform_target, from_value_type='waveforms')
C_r = C_r.squeeze(0)
E_r = E_r.squeeze(0)

C_p, E_p = dhasp.calculate_C(waveforms, from_value_type='waveforms')

# %% Calculate the correlations.
R = dhasp.calculate_R(C_r, C_p)

# %% Calculate the energy loss.
L_e = dhasp.calculate_L_e(E_r, E_p)

# %% Calculate the total loss
alpha = 1e-5
L = dhasp.calculate_L(R, L_e, alpha)

# %% Calculate everything in one hook.

start_time = time.time()

L_2, R_2, L_e_2 = dhasp.calculate_loss(waveforms, waveform_target, alpha)

stop_time = time.time()
print(f'Execution time: {stop_time - start_time:.2f} s.')

# %% Show results
data = {
    'Variant': waveform_names,
    'R': R.mean(dim=1).numpy(),
    'L_e': L_e.sum(dim=1).numpy(),
    'L': L.numpy()
    }

df_results = pd.DataFrame(data)

display(df_results)