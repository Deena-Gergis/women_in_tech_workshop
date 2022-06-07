"""Preprocessing function"""

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def split_answers(data_series: pd.Series, delimiter=";") -> pd.Series:
    """
    Split multiple answers in a single string
    to a list of single strings each representing a single answers

    Parameters:
    * data_series (pd.Series): String series with answers
    * delimiter (string): Another decimal integer
                          Defaults to ";"

    Returns: (pd.Series): If column contains
    """

    # Sub functions
    def is_splittable(pd_series, delimiter):
        """ Check if results multiple should be splitted - Returns boolean """
        return pd_series.str.contains(delimiter)

    def split_answer(pd_series, delimiter):
        """Function to split single answer"""
        return pd_series.str.split(delimiter)

    # --------------------

    # Check if multiple answers exist - if none: return original
    splittable_values = is_splittable(data_series, delimiter)
    if not splittable_values.any():
        return data_series

    # Else, split each value to a list
    modified_series = split_answer(data_series, delimiter)

    # Replace NAs with empty lists
    mask_null = modified_series.isnull()
    modified_series.loc[mask_null] = modified_series.loc[mask_null].apply(lambda x: [])

    return modified_series


def one_hot_encode(df: pd.DataFrame, columns):
    """One-hot-encode columns with multiple answers"""
    df = df.copy()

    if not isinstance(columns, list):
        raise ValueError('arg: column has to be a list')

    encoded_dfs = {}
    for column in columns:
        binarizer = MultiLabelBinarizer()
        encoded_df = pd.DataFrame(binarizer.fit_transform(df[column]),
                                  columns=binarizer.classes_,
                                  index=df[column].index)
        encoded_dfs[column] = encoded_df

    # Merge 1-hot encoded dfs and return
    encoded_dfs = pd.concat(encoded_dfs, axis=1)
    return encoded_dfs
