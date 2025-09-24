# Implementation of data loaders for datasets that do not fit in memory

# external modules
import os
import sys
import numpy as np
import pandas as pd

# local modules
thisdir = os.path.abspath(os.path.dirname(__file__))
topdir = os.path.abspath(os.path.join(thisdir, '../../'))
import tools.iotools as iotools


class MEDataLoader(object):
    
    def __init__(self, parquet_files, run_column='run_number', verbose=False):
        '''
        Initialize an MEDataLoader object
        Input arguments:
        - parquet_files: list of paths to parquet files
        - run_column: name of the column in the parquet files specifying the run number
        '''
        
        # parse input arguments
        if isinstance(parquet_files, str): parquet_files = [parquet_files]
            
        # copy attributes
        self.parquet_files = parquet_files[:]
        self.run_column = run_column
        
        # initialize random number generator
        self.rng = np.random.default_rng()
        
        # existence check
        for f in self.parquet_files:
            if not os.path.exists(f):
                msg = f'Provided parquet file {f} does not exist.'
                raise Exception(msg)
                
        # read number of rows per file
        self.nrows = []
        for f in self.parquet_files:
            this_nrows = len(iotools.read_parquet(f, columns=[self.run_column]))
            self.nrows.append(this_nrows)
        
        # printouts
        if verbose:
            msg = 'Initialized MEDataLoader object\n'
            msg += f'  - {len(self.parquet_files)} files\n'
            msg += f'  - {sum(self.nrows)} rows'
            print(msg)
        
    def read_random_batch(self, batch_size=32, columns=None, mode='batched', num_subbatches=10):
        '''
        Read a random batch from all instances in this data loader
        Input arguments:
        - mode: choose from the following options:
               - batched: partition the file in batches of size batch_size, then read one.
               - subbatched: same as batched but concatenate multiple subbatches of smaller size
                 (slower than batched but returned data is more shuffled)
        '''
        
        # pick a random file
        # todo: maybe weight probabilities by number of rows in each file?
        file_idx = self.rng.integers(0, high=len(self.parquet_files), endpoint=False)
        nrows = self.nrows[file_idx]
        
        # handle case where the batch contains the full file
        if batch_size >= nrows:
            df = iotools.read_parquet(self.parquet_files[file_idx], columns=columns)
            return df
        
        # handle different modes
        if mode=='batched':
            # partition the file in provided batch size and pick one
            max_first_batch = int((nrows-1)/batch_size)
            first_batch = self.rng.integers(0, high=max_first_batch, endpoint=True)
            last_batch = first_batch
            return iotools.read_parquet(self.parquet_files[file_idx], columns=columns,
                                        batch_size=batch_size, first_batch=first_batch, last_batch=last_batch)
        
        elif mode=='subbatched':
            # partition the file in smaller subbatches and concatenate a few
            # to make a single batch of size batch_size
            subbatch_size = int(batch_size / num_subbatches)
            max_subbatch_idx = int((nrows-1)/subbatch_size)
            replace = True if num_subbatches > max_subbatch_idx+1 else False
            subbatch_ids = self.rng.choice(np.arange(max_subbatch_idx+1), size=num_subbatches, replace=replace)
            return iotools.read_parquet(self.parquet_files[file_idx], columns=columns,
                                        batch_size=subbatch_size, batch_ids=subbatch_ids)
        
    def read_batch(self, batch_params, **kwargs):
        '''
        Read a batch specified by parameters
        Input arguments:
        - batch_params: tuple of the form (file index, batch size, batch index)
          OR tuple of the form (file index, list of run numbers)
        '''
        if len(batch_params)==3:
            file_idx, batch_size, batch_idx = batch_params
            file = self.parquet_files[file_idx]
            return iotools.read_parquet(file, batch_size=batch_size,
                      first_batch=batch_idx, last_batch=batch_idx, **kwargs)
        elif len(batch_params)==2:
            file_idx, runs = batch_params
            file = self.parquet_files[file_idx]
            return iotools.read_runs(file, runs, **kwargs)
                
    def prepare_sequential_batches(self, batch_size=32, **kwargs):
        '''
        Prepare batch parameters that, when read in order, cover the entire data loader sequentially.
        Returns:
        - list of tuples of the form (file index, batch size, batch index)
        '''
        
        # check input arguments and handle special cases
        if batch_size=='run': return self.prepare_run_batches(**kwargs)
        elif isinstance(batch_size, int): pass
        else:
            msg = f'Batch size "{batch_size}" not recognized.'
            raise Exception(msg)
            
        # prepare batch parameters
        res = []
        for file_idx, file in enumerate(self.parquet_files):
            nbatches = int((self.nrows[file_idx]-1)/batch_size)+1
            for batch_idx in range(nbatches): res.append((file_idx, batch_size, batch_idx))
        return res
    
    def read_sequential_batches(self, batch_size=32, **kwargs):
        '''
        Read the entire data loader in sequential batches
        '''
        
        # check input arguments and handle special cases
        if batch_size=='run': return self.read_run_batches(**kwargs)
        
        # prepare and read batches
        batches = self.prepare_sequential_batches(batch_size=batch_size)
        for batch in batches: yield self.read_batch(batch, **kwargs)
            
    def prepare_run_batches(self, target_size=None):
        '''
        Same as prepare_sequential_batches, but batches are done per run instead of a fixed size
        Input arguments:
        - target_size: preferred approximate batch size; batches will be chosen as the smallest
                       possible sequential number of runs that equals or exceeds the target_size
                       (with a minimum of one run per batch).
                       set to None (or a very small value) to use 1 run per batch, regardless of the size.
        Returns:
        - list of tuples of the form (file index, list of run numbers)
        '''
        res = []
        # loop over files
        for file_idx, file in enumerate(self.parquet_files):
            # find runs
            temp = iotools.read_parquet(file, columns=[self.run_column])
            run_numbers = temp[self.run_column].values
            # simple case of one run per batch
            if target_size is None:
                unique_runs = np.unique(run_numbers)
                for run in runs: res.append((file_idx, [run]))
                continue
            # else group runs in batches of target size
            run_number_change_ids = np.where(np.diff(run_numbers))[0]+1
            batch_start_idx = 0
            batch = [run_numbers[0]]
            for idx in run_number_change_ids:
                if idx - batch_start_idx >= target_size:
                    res.append((file_idx, batch))
                    batch_start_idx = idx
                    batch = [run_numbers[idx]]
                else: batch.append(run_numbers[idx])
            res.append((file_idx, batch))
        return res
                
    def read_run_batches(self, target_size=None, **kwargs):
        '''
        Same as read_sequential_batches, but batches are done per run instead of a fixed size
        '''
        
        batches = self.prepare_run_batches()
        for batch in batches: yield self.read_batch(batch, **kwargs)