import torch

from toolbox.initialization import *

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


# %% Get cepstral sequences and smoothed output envelopes
X_sample = audio_samples['tv']
X_target = audio_samples['clean']

C_target, E_target = dhasp.calculate_C(X_target, from_value_type='waveforms')

C_sample, E_sample = dhasp.calculate_C(X_sample, from_value_type='waveforms')

# %%
G_eq = 2 * torch.rand_like(dhasp.f_eq)


#%% Loss
def calculate_loss(X_sample, G_eq, C_target, E_target):

    X_sample_eq = dhasp.apply_filter('eq', X_sample, G_eq, joint_response=True)
    C_sample_eq, E_sample_eq = dhasp.calculate_C(X_sample_eq,
                                                 from_value_type='waveforms')

    R_sample_eq = dhasp.calculate_R(C_target, C_sample_eq)
    L_e_sample_eq = dhasp.calculate_L_e(E_target, E_sample_eq)

    return dhasp.calculate_L(R_sample_eq, L_e_sample_eq)

#%%

start_time = time.time()
loss = calculate_loss(X_sample, G_eq, C_target, E_target)

stop_time = time.time()
time_passed = stop_time - start_time
print(f'Time passed: {time_passed:.2f}')



#%%
idx = 5
sd.play(audio_samples['clean'][idx, :].squeeze().numpy(), fs_model, blocking=True)
sd.play(audio_samples['tv'][idx, :].squeeze().numpy(), fs_model, blocking=True)