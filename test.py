from toolbox.initialization import *
from toolbox.dsp import db2mag, mag2db, rms

# %%
df_timit = pd.read_pickle(paths.cache.sent)


# %%
sentence_data = df_timit.iloc[0, :]
path_recording = sentence_data['audio_path']
start_sample = sentence_data['start_sample']
end_sample = sentence_data['end_sample']


# %%

recording, fs = sf.read(path_recording)

recording_clean = recording[start_sample: end_sample].copy()
SNR = 10

noise = np.random.default_rng().random(recording_clean.shape) - 0.5
noise = db2mag(mag2db(rms(recording_clean) / rms(noise)) - SNR) * noise

recording_with_noise = recording_clean + noise

# sd.play(recording_clean, fs, blocking=True)
# sd.play(recording_with_noise, fs, blocking=True)

