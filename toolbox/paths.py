from pathlib import Path


# Define all the project paths
class ProjectPaths:

    def __init__(self, root=None):

        if root is None:
            # Get the path to this file
            path_to_file = Path(__file__)

            root = path_to_file.parent.parent

        self.root = root

        # Resources folder.
        self.resources = self.root / 'resources'

        # Data
        class Data:
            def __init__(self, paths):
                # Root folder for the data.
                self.root = paths.resources / 'data'

                # TIMIT
                class TIMIT:
                    def __init__(self, data):
                        # Root folder for the data.
                        self.root = data.root / 'TIMIT'

                        # Folder for the train data.
                        self.train = self.root / 'train'

                        # Folder for the test data.
                        self.test = self.root / 'test'

                self.timit = TIMIT(self)

                # Recordings
                class SyntheticSpeech:
                    def __init__(self, data):
                        # Root folder for the data.
                        self.root = data.root / 'synthetic_speech'

                        # Folder for the full length versions.
                        self.full_length = self.root / 'full_length'

                        # Folder for the full length versions.
                        self.cut_5_s = self.root / 'cut_5_s'

                self.synthetic_speech = SyntheticSpeech(self)

        self.data = Data(self)

        # Cache
        class Cache:
            def __init__(self, paths):
                # Root folder for the cache.
                self.root = paths.resources / 'cache'

                # TIMIT meta data (pandas dataframe)
                self.df_timit = self.root / 'df_timit.pkl'

                # Synthetic speech data (pandas dataframe)
                self.df_synthetic_speech = self.root / 'df_synthetic_speech.pkl'

        self.cache = Cache(self)
