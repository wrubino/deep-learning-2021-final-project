import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio as ta
import torchaudio.functional as taf

from pathlib import Path
from toolbox.paths import ProjectPaths
from toolbox.type_conversion import np2torch


def get_sentence_data(sentence_code: str,
                      path_folder: Path):
    """
    A function that returns the metadata about a sentence from a TIMIT dataset
    on the basis of the files present in the provided folder. 
    :param sentence_code: 
    :type sentence_code: 
    :param path_folder: 
    :type path_folder: 
    :return: 
    :rtype: 
    """

    # Define the extensions of the files containing
    # the data about the sentences.
    file_extensions = ['wav', 'txt', 'wrd', 'phn']

    # Create a path for each file type.
    sentence_paths = {
        extension: path_folder / f'{sentence_code}.{extension}'
        for extension in file_extensions
    }

    # Sentence data.
    # Get the content of the 'txt' file as a list of lines.
    file_content = (
        sentence_paths['txt']
            .read_text()
            .split('\n')
    )

    # Get the data about the sentence
    sentence = dict()
    sentence_data = file_content[0].strip().split()
    sentence['start_sample'] = int(sentence_data[0])
    sentence['end_sample'] = int(sentence_data[1])
    sentence['text'] = ' '.join(sentence_data[2:])
    sentence['audio_path'] = sentence_paths['wav']

    # Word data
    # Get the content of the 'wrd' as a list of lines.
    file_content = (
        sentence_paths['wrd']
            .read_text()
            .split('\n')
    )

    # Get the data about the words in the sentence.
    words = list()
    for line in file_content:
        # Skip empty lines.
        if not line.strip():
            continue

        # Initialize the dict in which the data about the words
        # will be saved.
        word = dict()

        # Extract the data about the words in the sentence.
        word_data = line.strip().split()
        word['start_sample'] = int(word_data[0])
        word['end_sample'] = int(word_data[1])
        word['text'] = ' '.join(word_data[2:])

        # Append the extracted data to the list of words.
        words.append(word)

    # Phoneme data.
    # Get the content of the 'wrd' as a list of lines.
    file_content = (
        sentence_paths['phn']
            .read_text()
            .split('\n')
    )

    # Get the data about the words sentence.
    phonemes = list()
    for line in file_content:
        # Skip empty lines.
        if not line.strip():
            continue

        # Initialize the dict in which the data about the words
        # will be saved.
        phoneme = dict()

        # Extract the data about the words in the sentence.
        phoneme_data = line.strip().split()
        phoneme['start_sample'] = int(phoneme_data[0])
        phoneme['end_sample'] = int(phoneme_data[1])
        phoneme['text'] = ' '.join(phoneme_data[2:])

        # Append the extracted data to the list of words.
        phonemes.append(phoneme)

    return sentence, words, phonemes


def timit_sentence_type(sentence_type_code: str):
    if sentence_type_code.lower() == 'si':
        sentence_type = 'phonetically-diverse'

    elif sentence_type_code.lower() == 'sa':
        sentence_type = 'dialect'

    elif sentence_type_code.lower() == 'sx':
        sentence_type = 'phonetically-compact'

    else:
        raise ValueError(
            f'Invalid sentence code: '
            f'"{sentence_type_code}"'
        )

    return sentence_type


def timit_dialect(dialect_code: str):
    if dialect_code.lower() == 'dr1':
        dialect = 'New England'

    elif dialect_code.lower() == 'dr2':
        dialect = 'Northern'

    elif dialect_code.lower() == 'dr3':
        dialect = 'North Midland'

    elif dialect_code.lower() == 'dr4':
        dialect = 'South Midland'

    elif dialect_code.lower() == 'dr5':
        dialect = 'Southern'

    elif dialect_code.lower() == 'dr6':
        dialect = 'New York City'

    elif dialect_code.lower() == 'dr7':
        dialect = 'Western'

    elif dialect_code.lower() == 'dr8':
        dialect = 'Army Brat'

    else:
        raise ValueError(
            f'Invalid dialect code: '
            f'"{dialect_code}"'
        )

    return dialect


def load_timit_data(
        project_paths: ProjectPaths = ProjectPaths()):
    """
    The function returns a dataframe containing the metadata about the TIMIT
    dataset.
    :return: 
    :rtype: 
    """

    # Define the names of the columns to include in the TIMIT dataframe
    column_names = [
        'sentence_number',
        'data_group',
        'dialect',
        'gender',
        'speaker',
        'type',
        'text',
        'audio_path',
        'start_sample',
        'end_sample',
        'words_text',
        'words_start_sample',
        'words_end_sample',
        'phonemes_text',
        'phonemes_start_sample',
        'phonemes_end_sample'
    ]

    # Initialize thd dataframe where the metadata will be stored.
    df_timit = pd.DataFrame(columns=column_names)

    # Define the folder paths for the training and test data.
    data_folders = dict(
        train=project_paths.data.timit.train,
        test=project_paths.data.timit.test
    )

    # Put the data in a dataframe
    for data_group, path_data_folder in data_folders.items():
        # Get the dialects.
        dialect_codes = [
            folder.stem
            for folder in path_data_folder.iterdir()
            if folder.is_dir()
        ]

        # Get the speaker codes.
        for dialect_code in dialect_codes:

            # Get the dialect.
            dialect = timit_dialect(dialect_code)

            # Get the path to the folder representing
            # a given dialect.
            path_dialect_folder = path_data_folder / dialect_code

            # Get the speaker codes present in the folder.
            speaker_codes = [
                folder.stem
                for folder in path_dialect_folder.iterdir()
                if folder.is_dir()
            ]

            # Get information for each speaker.
            for speaker_code in speaker_codes:
                # Get the gender and ID code for the speaker.
                gender = speaker_code[0]
                speaker = speaker_code[1:]

                # Define the path to the folder containing the
                # sentences spoken by the speaker.
                speaker_folder = path_dialect_folder / speaker_code

                # Get the codes for all the sentences present in the
                # speaker folder.
                sentence_codes = [
                    path.stem
                    for path in speaker_folder.glob('**/*.wav')
                ]

                # Get the information for each sentence.
                for sentence_code in sentence_codes:
                    # Extract and interpret the sentence types.
                    sentence_type_code = sentence_code[:2]

                    # Get the sentence type.
                    sentence_type = timit_sentence_type(sentence_type_code)

                    # Extract the sentence number.
                    sentence_number = int(sentence_code[2:])

                    # Get the data about the phonemes.
                    sentence, words, phonemes = \
                        get_sentence_data(sentence_code, speaker_folder)

                    # Put the data about the phonemes in a dict.
                    new_row = {
                        'sentence_number': sentence_number,
                        'data_group': data_group,
                        'dialect': dialect,
                        'gender': gender,
                        'speaker': speaker,
                        'type': sentence_type,
                        'text': sentence['text'],
                        'audio_path': (
                            sentence['audio_path']
                                .relative_to(
                                project_paths.data.timit.root)
                        ),
                        'start_sample': sentence['start_sample'],
                        'end_sample': sentence['end_sample'],
                        'words_text': [word['text']
                                       for word in words],
                        'words_start_sample': [word['start_sample']
                                               for word in words],
                        'words_end_sample': [word['end_sample']
                                             for word in words],
                        'phonemes_text': [phoneme['text']
                                          for phoneme in phonemes],
                        'phonemes_start_sample': [phoneme['start_sample']
                                                  for phoneme in phonemes],
                        'phonemes_end_sample': [phoneme['end_sample']
                                                for phoneme in phonemes]
                    }

                    # Append the data to the sentence dataframe.
                    df_timit = df_timit.append(new_row,
                                               ignore_index=True)

    # # Save the dataframe to cache.
    df_timit.to_pickle(project_paths.cache.df_timit)

    return df_timit


def load_synthetic_speech_data(
        project_paths: ProjectPaths = ProjectPaths()):
    """
    The function returns a dataframe containing the metadata about the
    synthetic speech dataset.
    :param project_paths:
    :type project_paths:
    :return:
    :rtype:
    """

    # Define the names of the columns to include in the recordings dataframe
    column_names = [
        'speaker',
        'length',
        'variant',
        'segment',
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
            ).relative_to(project_paths.data.synthetic_speech.root)

            # Append the information about the full length audio.
            df_synthetic_speech = df_synthetic_speech.append(
                dict(speaker=speaker,
                     length='full',
                     variant=variant,
                     segment=np.nan,
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
                path.relative_to(project_paths.data.synthetic_speech.root)
                for path in path_folder_5s_samples.glob('**/*.wav')]

            # Append the information about the 5s segment of the audio.
            for idx_segment, audio_path_segment in enumerate(audio_paths):
                # Append the information about the full length recording
                df_synthetic_speech = df_synthetic_speech.append(
                    dict(speaker=speaker,
                         length='5s',
                         variant=variant,
                         segment=int(audio_path_segment.stem),
                         fullness=fullness,
                         clarity=clarity,
                         audio_path=audio_path_segment),
                    ignore_index=True
                )

    # Save the dataframe to cache.
    df_synthetic_speech.to_pickle(
        project_paths.cache.df_synthetic_speech)

    return df_synthetic_speech


def get_waveforms_synthetic_speech(parameters,
                                   fs_output=None,
                                   paths: ProjectPaths = ProjectPaths()):
    # Load the dataframe
    df_synthetic_speech = pd.read_pickle(paths.cache.df_synthetic_speech)

    # Set up a filter.
    mask = np.ones(len(df_synthetic_speech), dtype=bool)
    for parameter_name, parameter_values in parameters.items():
        if not isinstance(parameter_values, list):
            parameter_values = [parameter_values]

        mask = mask & df_synthetic_speech[parameter_name].isin(parameter_values)

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

        if path.suffix == '.mp3':
            # Read the file using soundfile.
            waveform, fs = sf.read(path)

            # Transform the waveform int
            waveform = (
                np2torch(waveform)
                    .to(torch.float64)
                    .unsqueeze(0)
            )
        elif path.suffix == '.wav':
            waveform, fs = ta.load(path)
        else:
            raise ValueError(f'Invalid file type: "{path.suffx}".')

        # Resample to the output fs.
        if fs_output is None:
            fs_output = fs

        if fs_output != fs:
            resample = ta.transforms.Resample(fs, fs_output)
            waveform = resample(waveform)

        waveform = waveform.to(torch.float64)

        if variant not in audio.keys():
            audio[variant] = waveform
        else:
            audio[variant] = torch.cat([
                audio[variant],
                waveform
            ])


    return audio, fs_output
