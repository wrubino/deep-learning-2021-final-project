import pandas as pd
import soundfile as sf
import sounddevice as sd
from toolbox.paths import ProjectPaths


def play_timit(row_timit_data: pd.Series,
               project_paths: ProjectPaths = ProjectPaths(),
               blocking=False):

    # In case a dataframe is passed, process only the first row.
    if isinstance(row_timit_data, pd.DataFrame):
        row_synthetic_speech_data = row_timit_data.iloc[0, :]

    # Get the path to the audio file:
    path_recording = project_paths.data.timit.root / row_timit_data[
        'audio_path']
    start_sample = row_timit_data['start_sample']
    end_sample = row_timit_data['end_sample']

    # Read the audio file.
    recording, fs = sf.read(path_recording)
    recording = recording[start_sample:end_sample]

    # Play the sound
    sd.play(recording, fs, blocking=blocking)


def play_synthetic_speech(row_synthetic_speech_data: pd.Series,
                          project_paths: ProjectPaths = ProjectPaths(),
                          blocking=False):

    # In case a dataframe is passed, process only the first row.
    if isinstance(row_synthetic_speech_data, pd.DataFrame):
        row_synthetic_speech_data = row_synthetic_speech_data.iloc[0, :]

    # Get the path to the audio file:
    path_audio = (
            project_paths.data.synthetic_speech.root
            / row_synthetic_speech_data['audio_path']
    )

    # Read the audio file.
    audio, fs = sf.read(path_audio)

    # Play the sound
    sd.play(audio, fs, blocking=blocking)
