import pandas as pd
import toolbox.paths

from pathlib import Path


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


def load_timit_data(project_paths: toolbox.paths.ProjectPaths = None):
    """
    The function returns a dataframe containing the metadata about the TIMIT
    dataset
    :return: 
    :rtype: 
    """

    if project_paths is None:
        project_paths = toolbox.paths.ProjectPaths()

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
                        'audio_path': sentence['audio_path'],
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
