import numpy as np

import toolbox.paths
from toolbox.initialization import *
from toolbox.dsp import db2mag, mag2db, rms

# %%


def load_synthetic_speech_data(
        project_paths: toolbox.paths.ProjectPaths = None):
    """
    The function returns a dataframe containing the metadata about the
    synthetic speech dataset.
    :param project_paths:
    :type project_paths:
    :return:
    :rtype:
    """

    # Default project paths.
    if project_paths is None:
        project_paths = toolbox.paths.ProjectPaths()

    # Define the names of the columns to include in the recordings dataframe
    column_names = [
        'speaker',
        'length',
        'variant',
        'segment_number',
        'fullness',
        'clarity',
        'audio_path',
    ]

    # Initialize the dataframe in which the data will be stored
    df_synthetic_speech = pd.DataFrame(columns=column_names)

    # Define the folder paths for the training and test data.
    data_folders = dict(
        full_length=project_paths.data.synthetic_speech.full_length,
        cut_5_s=project_paths.data.synthetic_speech.cut_5_s
    )

    def get_variant(variant_code):
        """
        Interpret variant code and reurn variant, fullness, clarity, and
        file extension.

        :param variant_code:
        :type variant_code:
        :return:
        :rtype:
        """
        # Initialize the values for clarity and fullness that only have
        # relevance for the variant "augmented"
        clarity = np.nan
        fullness = np.nan

        # Get the values of clarity and fullness for the variant: augmented
        if variant_code in variant_codes_augmented:
            variant = 'zoom_augmented'
            fullness, clarity = [
                int(item) for item in variant_code.split('-')]
        else:
            if variant_code == 'orig':
                variant = 'clean'
            elif variant_code == 'zoom':
                variant = 'zoom'
            elif variant_code == 'babb':
                variant = 'babble'
            elif variant_code == 'musi':
                variant = 'music'
            elif variant_code == 'tele':
                variant = 'tv'
            elif variant_code == 'bitc':
                variant = 'distorted'
            else:
                variant = None

        # Define the file extension
        if variant_code == 'orig':
            extension = 'mp3'
        else:
            extension = 'wav'

        return variant, fullness, clarity, extension

    # Get the speaker names.
    speakers = [
        item.stem
        for item in data_folders['full_length'].iterdir()
        if item.is_dir()
    ]

    # Get the codes of the variants of the recordings.
    folder_path_first_speaker = data_folders['full_length'] / speakers[0]

    variant_codes = [
        path.stem
        for path in (
                list(folder_path_first_speaker.glob('**/*.wav'))
                + list(folder_path_first_speaker.glob('**/*.mp3'))
        )
    ]

    # Define the variant codes for the variant "augmented_zoom":
    variant_codes_augmented = [
        f'{fullness}-{clarity}'
        for fullness in range(1, 5)
        for clarity in range(1, 5)
    ]

    # Load the metadata into the dataframe
    for speaker in speakers:
        for variant_code in variant_codes:
            # Interpret the variant code.
            variant, fullness, clarity, extension = get_variant(variant_code)

            # If the variant is unsupported, skip the file
            if variant is None:
                continue

            # Get the path of the audio file:
            audio_path_full_length = (
                    data_folders['full_length'] / speaker /
                    f'{variant_code}.{extension}'
            ).relative_to(paths.data.synthetic_speech.root)

            # Append the information about the full length audio.
            df_synthetic_speech = df_synthetic_speech.append(
                dict(speaker=speaker,
                     length='full',
                     variant=variant,
                     segment_number=np.nan,
                     fullness=fullness,
                     clarity=clarity,
                     audio_path=audio_path_full_length),
                ignore_index=True
            )

            # Create the path to the folder containing the 5s samples of
            # the full recordings
            path_folder_5s_samples = (
                    data_folders['cut_5_s'] / speaker / variant_code)

            # Get the paths of all the audio files (they are all wav)
            audio_paths = [
                path.relative_to(paths.data.synthetic_speech.root)
                for path in path_folder_5s_samples.glob('**/*.wav')]

            # Append the information about the 5s segment of the audio.
            for idx_segment, audio_path_segment in enumerate(audio_paths):

                # Append the information about the full length recording
                df_synthetic_speech = df_synthetic_speech.append(
                    dict(speaker=speaker,
                         length='5s',
                         variant=variant,
                         segment_number=idx_segment,
                         fullness=fullness,
                         clarity=clarity,
                         audio_path=audio_path_segment),
                    ignore_index=True
                )

    # Save the dataframe to cache.
    df_synthetic_speech.to_pickle(
        project_paths.cache.df_synthetic_speech)

    return df_synthetic_speech
