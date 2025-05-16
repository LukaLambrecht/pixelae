# **Tools for reading and writing dataframes from/to parquet files**

import os
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow.parquet import ParquetFile, ParquetDataset


def read_parquet(path, verbose=False, 
                 columns=None, batch_size=None, first_batch=0, last_batch=0):
    """
    Read a parquet file into a dataframe.
    Input arguments:
    - path: path to the parquet file to read.
    - kwargs: passed down to pandas.read_parquet,
      see https://pandas.pydata.org/docs/reference/api/pandas.read_parquet.html.
    Returns:
    - dataframe with the contents of the read file.
    """
    if batch_size is None:
        # standard case where all rows are read
        df = pd.read_parquet(path, columns=columns)
    else:
        # more involved case where only a section of rows is read
        pf = ParquetFile(path)
        if verbose:
            print('Found following parquet metadata:')
            print(pf.metadata)
            
        # check internal consistency
        if last_batch < first_batch:
            last_batch = first_batch
            if verbose:
                msg = f'WARNING: setting last_batch to {last_batch}'
                msg += ' as values smaller than first_batch are not supported.'
                print(msg)
        
        # check available rows
        num_rows = pf.metadata.num_rows
        first_row = first_batch * batch_size
        if first_row >= num_rows:
            msg = f'Requested to read first row {first_row},'
            msg += f' but the data has only {num_rows} rows.'
            raise Exception(msg)
        first_row_last_batch = last_batch * batch_size
        if first_row_last_batch >= num_rows:
            last_batch = int((num_rows-1) / batch_size)
            if verbose:
                msg = f'WARNING: setting last_batch to {last_batch}'
                msg += f' as the data only has {num_rows} rows.'
                print(msg)
        
        # iterate through the batches
        iterobj = pf.iter_batches(batch_size = batch_size)
        for counter in range(first_batch): _ = next(iterobj)
        batches = [next(iterobj) for counter in range(last_batch+1-first_batch)]
        df = pa.Table.from_batches(batches).to_pandas()
        
    if verbose:
        print(f'Read dataframe with {len(df)} rows and {len(df.columns)} columns.')
    return df


def read_lumisections(path, run_numbers, ls_numbers,
                      verbose=False, columns=None,
                      run_column='run_number', ls_column='ls_number'):
    """
    Read specific lumisections from a file or list of files
    Note: can be slow, do not use for many lumisections.
    """
    
    # make filter
    lsfilter = []
    for run_number, ls_number in zip(run_numbers, ls_numbers):
        lsfilter.append( [(run_column, '=', run_number), (ls_column, '=', ls_number)] )
    
    # read dataframe
    ds = ParquetDataset(path, filters=lsfilter)
    df = ds.read().to_pandas()
        
    return df