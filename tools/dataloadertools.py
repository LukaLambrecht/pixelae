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
        self.nrows = ([len(iotools.read_parquet(f, columns=[self.run_column])) for f in self.parquet_files])
        
        # printouts
        msg = 'Initialized MEDataLoader object\n'
        msg += f'  - {len(self.parquet_files)} files\n'
        msg += f'  - {sum(self.nrows)} rows'
        print(msg)
        
    def read_random_batch(self, batch_size=32, columns=None, batched=False):
        
        # pick a random file
        # todo: maybe weight probabilities by number of rows in each file?
        file_idx = self.rng.integers(0, high=len(self.parquet_files), endpoint=False)
        nrows = self.nrows[file_idx]
        
        # handle case where the batch contains the full file
        if batch_size >= nrows:
            return iotools.read_parquet(self.parquet_files[file_idx], columns=columns)
        
        # make random start index
        if not batched:
            # case 1: fully random start index (can be slow)
            max_start_idx = nrows - batch_size
            start_idx = self.rng.integers(0, high=max_start_idx, endpoint=True)
            stop_idx = start_idx + batch_size
            return iotools.read_parquet(self.parquet_files[file_idx], columns=columns,
                                        batch_size=1, first_batch=start_idx, last_batch=stop_idx-1)
        else:
            # case 2: partition the file in provided batch size and pick one (much faster)
            max_first_batch = int((nrows-1)/batch_size)
            first_batch = self.rng.integers(0, high=max_first_batch, endpoint=True)
            last_batch = first_batch
            return iotools.read_parquet(self.parquet_files[file_idx], columns=columns,
                                        batch_size=batch_size, first_batch=first_batch, last_batch=last_batch)
        
    def read_sequential_batches(self, batch_size=32, columns=None):
        for file_idx, file in enumerate(self.parquet_files):
            nbatches = int((self.nrows[file_idx]-1)/batch_size)+1
            for batch_idx in range(nbatches):
                yield iotools.read_parquet(file, columns=columns, batch_size=batch_size,
                        first_batch=batch_idx, last_batch=batch_idx)