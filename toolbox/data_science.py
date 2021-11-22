import pandas as pd


def balance_dataframe(df: pd.DataFrame, column_name):
    """
    A function for balancing a dataframe so that the number of rows
    containing each value present in the designated column will be the same.
    :param df:
    :type df:
    :param column_name:
    :type column_name:
    :return:
    :rtype:
    """

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
