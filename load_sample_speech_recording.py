# %% Imports
import numpy as np
from toolbox.initialization import *


def get_waveforms_synthetic_speech(parameters):
    # %% Load the dataframe
    df_synthetic_speech = pd.read_pickle(paths.cache.df_synthetic_speech)

    # Set up a filter.
    mask = np.ones(len(df_synthetic_speech), dtype=bool)
    for parameter_name, parameter_value in parameters.items():
        mask = mask & (df_synthetic_speech[parameter_name] == parameter_value)

    # Get a dataframe containing data from segment.
    df_segment = (
        df_synthetic_speech
            .loc[mask, :]
    )

    # Load the audio data.
    audio = dict()
    for idx_row, row in df_segment.iterrows():
        variant = row['variant']
        fullness = row['fullness']
        clarity = row['clarity']
        path = paths.data.synthetic_speech.root / row['audio_path']

        if variant == 'zoom_augmented':
            variant = f'{variant}_f{fullness}_c{clarity}'

        audio[variant], fs = sf.read(path)

    return audio, fs


# %% Play some of the samples
audio, fs = get_waveforms_synthetic_speech(dict(
    speaker='joanna',
    length='5s',
    segment=10
))

sd.play(audio['clean'], fs)
sd.play(audio['zoom'], fs)
sd.play(audio['babble'], fs)
sd.play(audio['zoom_augmented_f4_c4'], fs)

