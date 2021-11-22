import pandas as pd
import numpy as np


def get_df_columns(df, exclude=None):
    # If columns to be excluded have not been defined,
    # represent it as an empty list.
    if exclude is None:
        exclude = list()

    # If the columns to be excluded are not ½ using a list
    # or a tuple, represent them as a list.
    elif not isinstance(exclude, (list, tuple)):
        exclude = [exclude]

    # Return all column names except the ones to exclude.
    return [column for column in df.columns.to_list()
            if column not in exclude]


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


def reset_df_index_names(df):
    return (
        df
            .rename_axis(['' for level in range(df.columns.nlevels)],
                         axis="columns")
            .rename_axis(['' for level in range(df.index.nlevels)],
                         axis="rows")
    )


def df_sort_columns(df: pd.DataFrame,
                    first_columns=None):

    """
    # A function that sorts the columns in alphabetical order
    # and puts the user-chosen columns first
    :param df:
    :type df:
    :param first_columns:
    :type first_columns:
    :return:
    :rtype:
    """

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


def df_create_column_for_each_unique_value(df,
                                           category_columns,
                                           value_columns,
                                           aggfunc='first'):
    """
    A function that creates a new column representing data in 'value_columns'
    for every unique value in 'category_columns'.
    NB: Might not alsays work well!
    :param df:
    :type df:
    :param category_columns:
    :type category_columns:
    :param value_columns:
    :type value_columns:
    :param aggfunc:
    :type aggfunc:
    :return:
    :rtype:
    """

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


def create_empty_multiindex(n_levels, names=None):
    return pd.MultiIndex.from_arrays(
        arrays=[list() for _ in range(n_levels)],
        names=names
    )
