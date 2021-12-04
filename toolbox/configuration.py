import warnings
import pandas as pd
import torch

from IPython import get_ipython, InteractiveShell

# %%

# Check whether the code is run from a Jupyter notebook
run_from_jupyter = (
        (get_ipython() is not None)
        & ('PyDevTerminal' not in str(type(get_ipython())))
)

# %% ## Notebook options

# Decide which output is shown below the cells.
if run_from_jupyter:
    InteractiveShell.ast_node_interactivity = "none"
else:
    InteractiveShell.ast_node_interactivity = "all"

# %% ## Matplotlib options

# Show matplotlib plots inline.
if run_from_jupyter:
    get_ipython().run_line_magic('matplotlib', 'inline')

# %% ## Pandas options

# Define the format in which the numbers will be shown in
# the pandas dataframes.
pd.options.display.float_format = '{:,.2f}'.format

# Decide how to handle the "SettingWithCopyWarning" warning
pd.options.mode.chained_assignment = None  # default='warn'

# At multirow, top-align (False) or center-align (True)
pd.options.display.latex.multirow = False

# Set the maximum number of rows and columns to show when
# displaying a Pandas dataframe.
pd.options.display.max_rows = 150
pd.options.display.max_columns = 200

# %% Torch options
torch.set_default_dtype(torch.float64)

# %% ## Warnings

# Decide how to handle warnings.
if run_from_jupyter:
    warnings.filterwarnings(action='ignore',
                            category=UserWarning)
    warnings.filterwarnings(action='ignore',
                            category=pd.errors.PerformanceWarning)
