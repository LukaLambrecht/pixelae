# **Tools for reading and writing dataframes from/to parquet files**

import os
import sys
import numpy as np
import pandas as pd


def read_parquet(path, **kwargs):
    """
    Read a parquet file into a dataframe.
    Input arguments:
    - path: path to the parquet file to read.
    - kwargs: passed down to pandas.read_parquet,
      see https://pandas.pydata.org/docs/reference/api/pandas.read_parquet.html.
    Returns:
    - dataframe with the contents of the read file.
    Note: for now this function is just a trivial wrapper around pandas.read_parquet,
    but maybe extend later.
    """
    df = pd.read_parquet(path, **kwargs)
    return df