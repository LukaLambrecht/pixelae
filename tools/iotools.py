# **Tools for reading and writing dataframes from/to parquet files**

import os
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow.parquet import ParquetFile, ParquetDataset


def read_parquet(path, verbose=False, 
                 columns=None, batch_size=None, first_batch=None, last_batch=None, batch_ids=None):
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
            
        # check if contradictory arguments were provided
        if batch_ids is not None:
            if first_batch is not None or last_batch is not None:
                if verbose:
                    msg = 'WARNING in read_parquet: cannot provide both batch_ids and first_batch / last_batch;'
                    msg += ' first_batch and last_batch will be ignored.'
                    print(msg)
        else:
            if first_batch is None or last_batch is None:
                msg = 'ERROR in read_parquet: in batched mode, either batch_ids, or first_batch and last_batch'
                msg += ' must be provided; returning None.'
                print(msg)
                return None
            if last_batch < first_batch:
                last_batch = first_batch
                if verbose:
                    msg = f'WARNING in read_parquet: setting last_batch to {last_batch}'
                    msg += ' as values smaller than first_batch are not supported.'
                    print(msg)
            batch_ids = list(range(first_batch, last_batch+1))
        
        # check available rows and batches
        num_rows = pf.metadata.num_rows
        num_batches = int((num_rows-1)/batch_size)+1
        if max(batch_ids) >= num_batches:
            if verbose:
                msg = f'WARNING in read_parquet: batch indices greater than {num_batches-1} will be ignored.'
                print(msg)
        
        # iterate through the batches
        iterobj = pf.iter_batches(batch_size = batch_size)
        batches = []
        for batch_idx in range(num_batches):
            batch = next(iterobj)
            if batch_idx in batch_ids: batches.append(batch)
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


def read_runs(path, run_numbers,
              verbose=False, columns=None,
              run_column='run_number'):
    """
    Read specific runs from a file or list of files
    """
    
    # make filter
    runfilter = []
    runfilter.append( [(run_column, 'in', run_numbers)] )
    
    # read dataframe
    ds = ParquetDataset(path, filters=runfilter)
    df = ds.read().to_pandas()
        
    return df