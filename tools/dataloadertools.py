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
    
    def __init__(self, parquet_files, run_column='run_number'):
        self.parquet_files = parquet_files
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
        msg = 'Initialized MEDataLoader object\n'
        msg += f'  - {len(self.parquet_files)} files\n'
        msg += f'  - {sum(self.nrows)} rows'
        print(msg)
        
    def read_random_batch(self, batch_size=32, columns=None, mode='batched', num_subbatches=10):
        # mode: choose from the following options:
        #       - batched: partition the file in batches of size batch_size, then read one.
        #       - subbatched: same as batched but concatenate multiple subbatches of smaller size
        #         (slower than batched but returned data is more shuffled)
        
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
        
    def read_sequential_batches(self, batch_size=32, columns=None):
        for file_idx, file in enumerate(self.parquet_files):
            nbatches = int((self.nrows[file_idx]-1)/batch_size)+1
            for batch_idx in range(nbatches):
                yield iotools.read_parquet(file, columns=columns, batch_size=batch_size,
                        first_batch=batch_idx, last_batch=batch_idx)
                
    def prepare_sequential_batches(self, batch_size=32):
        res = []
        for file_idx, file in enumerate(self.parquet_files):
            nbatches = int((self.nrows[file_idx]-1)/batch_size)+1
            for batch_idx in range(nbatches): res.append((file_idx, batch_size, batch_idx))
        return res
                
    def read_sequential_batch(self, batch_params, columns=None):
        file_idx, batch_size, batch_idx = batch_params
        file = self.parquet_files[file_idx]
        return iotools.read_parquet(file, columns=columns, batch_size=batch_size,
                  first_batch=batch_idx, last_batch=batch_idx)