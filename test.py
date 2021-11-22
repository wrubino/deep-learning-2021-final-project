import matplotlib.pyplot as plt
import pandas as pd
import soundfile as sf
import sounddevice as sd

from pathlib import Path

import HASPI_v2
from toolbox.initialization import *


# %%
df_sentences = pd.read_pickle(paths.cache.sentence_data)


# %%
sentence_data = df_sentences.iloc[0, :]
path_recording = sentence_data['audio_path']
start_sample = sentence_data['start_sample']
end_sample = sentence_data['end_sample']


# %%
def rms(signal: np.ndarray):
    return np.sqrt(np.mean(np.power(signal, 2)))


def db2mag(magnitude_db: np.ndarray):
    return np.power(10, magnitude_db / 20)


def mag2db(magnitude: np.ndarray):
    return 20 * np.log10(magnitude)


recording, fs = sf.read(path_recording)

recording_clean = recording[start_sample: end_sample].copy()
SNR = 10

noise = np.random.default_rng().random(recording_clean.shape) - 0.5
noise = db2mag(mag2db(rms(recording_clean) / rms(noise)) - SNR) * noise

recording_with_noise = recording_clean + noise

# sd.play(recording_clean, fs, blocking=True)
# sd.play(recording_with_noise, fs, blocking=True)


# %%
haspi, _ = HASPI_v2.HASPI_v2(
    recording_clean,
    fs,
    recording_clean,
    fs,
    HL=np.zeros(6)
)

print(f'HASPI: {haspi}')
