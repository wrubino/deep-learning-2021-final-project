# %% # Imports

import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysepm
import requests
import soundfile as sf
import sounddevice as sd
import warnings

from IPython.core.interactiveshell import InteractiveShell
from IPython.display import display
from IPython.display import HTML
from IPython.display import IFrame
from IPython.display import Markdown
from IPython.display import YouTubeVideo
from IPython import get_ipython
from operator import itemgetter
from pathlib import Path
from tqdm import tqdm

# %% # Configuration

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

# %% ## Warnings

# Decide how to handle warnings.
if run_from_jupyter:
    warnings.filterwarnings(action='ignore',
                            category=UserWarning)
    warnings.filterwarnings(action='ignore',
                            category=pd.errors.PerformanceWarning)


# %% # Paths

# Define all the project paths
class Paths:

    def __init__(self, root=None):
        self.root = Path(
            r'G:\My Drive\DTU\Kurser\Deep_Learning_02456\final_project')

        # Resources folder.
        self.resources = self.root / 'resources'

        # Data
        class Data():
            def __init__(self, paths):
                # Root folder for the data.
                self.root = paths.resources / 'data'

                # TIMIT
                class TIMIT():
                    def __init__(self, data):
                        # Root folder for the data.
                        self.root = data.root / 'TIMIT'

                        # Folder for the train data.
                        self.train = self.root / 'train'

                        # Folder for the test data.
                        self.test = self.root / 'test'

                self.timit = TIMIT(self)

                # Recordings
                class Recordings():
                    def __init__(self, data):
                        # Root folder for the data.
                        self.root = data.root / 'recordings'

                        # Folder for the full length versions.
                        self.full_length = self.root / 'full_length'

                        # Folder for the full length versions.
                        self.cut_5_s = self.root / 'cut_5_s'

                self.recordings = Recordings(self)

        self.data = Data(self)

        # Cache
        class Cache():
            def __init__(self, paths):
                # Root folder for the cache.
                self.root = paths.resources / 'cache'

                # Sentence data (pandas dataframe)
                self.sentence_data = self.root / 'sentence_data.pkl'

        self.cache = Cache(self)


paths = Paths()

# %% # Function definitions
#
pass


# %% ## General functions

# A function that returns a dict of object attributes.
def get_obj_attributes(obj):
    return {attribute_name: getattr(obj, attribute_name)
            for attribute_name in dir(obj)
            if (not attribute_name.startswith('__')
                and not callable(getattr(obj, attribute_name)))}


# A function that returns a dict of object methods.
def get_obj_methods(obj):
    return {method_name: getattr(obj, method_name)
            for method_name in dir(obj)
            if (not method_name.startswith('__')
                and callable(getattr(obj, method_name)))}


# A function that prints a string in markdown format.
def printmd(string):
    # Define a function that will print a markdown text.
    # A function that prints a string in markdown format.
    def printmd(markdown_text,
                font_family='courier',
                font_size=14):
        # Initialize the html formatting markup.
        pre_html = '<span style="'

        # Set the font family in the HTML markup.
        if font_family is not None:
            pre_html += f'font-family: {font_family.lower()}; '
            html_applied = True

        # Set the font size in the HTML markup.
        if font_size is not None:
            pre_html += f'font-size: {font_size}px; '
            html_applied = True

        # Finish the HTML markup
        pre_html += '">'
        post_html = '</span>'

        # Create the final text string to be displayed.
        final_markdown_text = f'{pre_html}{markdown_text}{post_html}'

        # Display.
        display(Markdown(final_markdown_text))


# A function that returns unique values from a text.
def unique(list_):
    return list(set(list_))


# %% ## Plotting functions

# A function that applies default formatting to an axes.
def format_axes(axes: plt.Axes,
                keep_box=False):
    if not keep_box:
        axes.spines['top'].set_color('white')
        axes.spines['right'].set_color('white')

    axes.set_facecolor("white")


# A function that applies default formatting to annotation
# of an axes.
def format_axes_annotation(axes: plt.Axes):
    axes.xaxis.label.set_fontsize(14)
    axes.yaxis.label.set_fontsize(14)
    axes.title.set_fontsize(16)


# A function for creating common x-label for the figure.
def figure_x_label(figure: plt.Figure,
                   label: str,
                   y_position=0.04,
                   font_size=16):
    figure.text(0.5, y_position, label,
                ha='center',
                fontdict={'size': font_size})


# A function for creating common y-label for the figure.
def figure_y_label(figure: plt.Figure,
                   label: str,
                   x_position=0.04,
                   font_size=16):
    figure.text(x_position, 0.5, label,
                va='center',
                rotation='vertical',
                fontdict={'size': font_size})


# A function that draws a horizontal line across the entire axes.
def draw_threshold(value: float,
                   axes: plt.Axes,
                   linewidth=1,
                   linestyle='-',
                   color=None,
                   title=None):
    # Get axes limits and ranges.
    x_min, x_max = axes.get_xlim()
    x_range = x_max - x_min
    y_min, y_max = axes.get_ylim()
    y_range = y_max - y_min

    # Plot the threshold line.
    axes.plot([x_min, x_max], [value, value],
              linewidth=1,
              linestyle='-',
              color=color)

    # Write a title above the threshold line
    if title is not None:
        axes.text(x_min + 0.01 * x_range,
                  value + 0.02 * y_range,
                  title)


# %% ## Dataframe functions

# A function that gets column names of a dataframe.
def get_df_columns(df, exclude=None):
    # If columns to be excluded have not been defined,
    # represent it as an empty list.
    if exclude is None:
        exclude = list()

    # If the columns to be excluded are not specified using a list
    # or a tuple, represent them as a list.
    elif not isinstance(exclude, (list, tuple)):
        exclude = [exclude]

    # Return all column names except the ones to exclude.
    return [column for column in df.columns.to_list()
            if column not in exclude]


# A function that gets rows names of a dataframe.
def get_df_rows(df, exclude=None):
    # If columns to be excluded have not been defined,
    # represent it as an empty list.
    if exclude is None:
        exclude = list()

    # If the columns to be excluded are not specified using a list
    # or a tuple, represent them as a list.
    elif not isinstance(exclude, (list, tuple)):
        exclude = [exclude]

    # Return all column names except the ones to exclude.
    return [row for row in df.index.to_list()
            if row not in exclude]


# A function that resets the names of indices
def reset_df_index_names(df):
    return (
        df
            .rename_axis(['' for level in range(df.columns.nlevels)],
                         axis="columns")
            .rename_axis(['' for level in range(df.index.nlevels)],
                         axis="rows")
    )


# A function that sorts the columns in alphabethical order
# and puts the user-chosen columns first
def df_sort_columns(df: pd.DataFrame,
                    first_columns=None):
    # Define a function that moves the chosen element to the
    # front of the list.
    def move_to_front(element, list_):
        if element in list_:
            list_.insert(0, list_.pop(list_.index(element)))

    # Make sure that the columns that are to be put in front
    # are represented as a list.
    if first_columns is None:
        first_columns = list()

    # Sort the columns in alphabetical order.
    sorted_columns = list(df.columns)
    sorted_columns.sort()

    # Move the user-chosen columns to the front.
    for column in first_columns[::-1]:
        move_to_front(column, sorted_columns)

    # Assign the ordered columns to the dataframe.
    df = df[sorted_columns]

    return df


# A function that creates a new column representing data in 'value_columns'
# for every unique value in 'category_columns'.
def df_create_column_for_each_unique_value(df,
                                           category_columns,
                                           value_columns,
                                           aggfunc='first'):
    # Always represent category and value columns as a list or tuple.
    if not isinstance(category_columns, (list, tuple)):
        category_columns = [category_columns]

    if not isinstance(value_columns, (list, tuple)):
        value_columns = [value_columns]

    # Create a column order for grouping so that all the value columns
    # come last and category columns second last. We leave out 1
    # value column for the result
    cat_and_value_columns = category_columns + value_columns
    column_order = (
            get_df_columns(df, exclude=cat_and_value_columns)
            + cat_and_value_columns[:-1]
    )

    # Create columns from unique values by grouping and unstacking.
    df = (
        df
            .groupby(column_order)
            .first()
            .unstack(list(np.arange(-len(cat_and_value_columns) + 1,
                                    0)))
            .reset_index()
    )

    # Delete the names of the index levels
    df = df.rename_axis(['' for level in range(df.columns.nlevels)],
                        axis="columns")
    return df


# A function that flattens the multiindex of a dataframe.
def flatten_multiindex(df, axis='columns'):
    # Get the desired index
    if axis in [1, 'columns']:
        index = df.columns
    elif axis in [0, 'rows']:
        index = df.index
    else:
        raise ValueError(f'Invalid axis: "{axis}".')

    # Join all the levels except the empty ones with a ', '
    flat_index = list()
    for element in index.values:
        if not isinstance(element, (tuple, list)):
            flat_index.append(element)
        else:
            flat_element = ''
            for idx, subelement in enumerate(element):
                if subelement:
                    if idx == 0:
                        flat_element += subelement
                    else:
                        flat_element += ', ' + subelement

            flat_index.append(flat_element)

    # Assign the index to the dataframe
    if axis in [1, 'columns']:
        df.columns = flat_index
    elif axis in [0, 'rows']:
        df.index = flat_index

    return df


# A function that creates an empty multiindex of a given depth
def create_empty_multiindex(n_levels, names=None):
    return pd.MultiIndex.from_arrays(
        arrays=[list() for _ in range(n_levels)],
        names=names
    )


# %% ## Data science functions

# A function for balancing a dataframe so that the number of rows
# containing each value present in the designated column will be the same.
def balance_dataframe(df: pd.DataFrame, column_name):
    # Get the count of the least frequent occurrence in column.
    lowest_frequency = df[column_name].value_counts().min()

    # Create an empty dataframe for storing the balanced data
    df_balanced = pd.DataFrame()

    # For each value in column, randomly choose the number of samples
    # that corresponds to the least frequent value in the column.
    for value in df[column_name].unique():
        df_balanced = df_balanced.append(
            df
                .loc[df[column_name] == value]
                .sample(lowest_frequency)
        )

    return df_balanced
