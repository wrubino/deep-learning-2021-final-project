# %% Imports
from toolbox.initialization import *

# %% Load audio

# Load audio samples representing the same speech segment in different
# acoustic conditions.
fs_model = 24e3
audio_samples, _ = t.data_loading.get_waveforms_synthetic_speech(
    dict(
        variant=['clean', 'tv'],
        length='5s',
        segment=10
    ),
    fs_model
)

# Show the shapes of the tensors containing the clean and noisy speech.
display(audio_samples['clean'].shape)
display(audio_samples['tv'].shape)

# %% Initialize the dhasp model.
dhasp = t.dhasp.DHASP(fs_model)


#%%
g = torch.arange(9).reshape(-1, 1) * torch.ones(9, 8)

a = dhasp.apply_filter('eq',
    audio_samples['clean'][0, :].to(torch.float64).repeat(9, 1),
    gain=g,
    joint_response=False
)

#%%
for i in range(a.shape[0]):
    print(torch.max(a[i, :]))
