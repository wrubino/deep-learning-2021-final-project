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

        # Graphics
        class Graphics:
            def __init__(self, paths):
                # Root folder for the graphics.
                self.root = paths.resources / 'graphics'

                self.dhasp_proposed_framework = \
                    self.root / 'dhasp_proposed_framework.jpg'

                self.dhasp_original_framework = \
                    self.root / 'dhasp_original_framework.jpg'

                self.dhasp_auditory_model = \
                    self.root / 'dhasp_differentiable_auditory_model.jpg'

                # Folder for the test data.
                self.test = self.root / 'test'

        self.graphics = Graphics(self)

        # Cache
        class Cache:
            def __init__(self, paths):
                # Root folder for the cache.
                self.root = paths.resources / 'cache'

                # TIMIT meta data (pandas dataframe)
                self.df_timit = self.root / 'df_timit.pkl'

                # Synthetic speech data (pandas dataframe)
                self.df_synthetic_speech = self.root / 'df_synthetic_speech.pkl'

                # Best model
                self.model_best = self.root / 'model_best.pkl'

                # Last model
                self.model_last = self.root / 'model_last.pkl'

                # Network input parameters
                self.X_input = self.root / 'X_input.pt'
                self.X_target = self.root / 'X_target.pt'
                self.C_input = self.root / 'C_input.pt'
                self.E_input = self.root / 'E_input.pt'
                self.C_target = self.root / 'C_target.pt'
                self.E_target = self.root / 'E_target.pt'





        self.cache = Cache(self)
