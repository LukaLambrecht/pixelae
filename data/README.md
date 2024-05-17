# Collect input data

Read datasets from DAS and extract MEs of interest.
Original idea was to read the DQMIO datasets remotely and extract the data directly.
However, the datasets seem to be not or very instably available via DAS.
Therefore, new approach is to request a temporary copy of the datasets on T2B via rucio.
See https://t2bwiki.iihe.ac.be/Rucio.
From there, the relevant data can be extracted and stored locally.

### Which datasets
The study will be conducted with 2023 data.
Per-lumisection DQMIO was enabled starting from era 2023C,
so ignore 2023A and 2023B (for now).
In detail, we have the following datasets:
- `/ZeroBias/Run2023C-PromptReco-v1/DQMIO`
- `/ZeroBias/Run2023C-PromptReco-v2/DQMIO`
- `/ZeroBias/Run2023C-PromptReco-v3/DQMIO`
- `/ZeroBias/Run2023C-PromptReco-v4/DQMIO`
- `/ZeroBias/Run2023D-PromptReco-v1/DQMIO`
- `/ZeroBias/Run2023D-PromptReco-v2/DQMIO`
- `/ZeroBias/Run2023E-PromptReco-v1/DQMIO`
- `/ZeroBias/Run2023F-PromptReco-v1/DQMIO`

### Status of Rucio requests
Rucio transfers were requested on 16 Oct 2023.
All transfers seem to be successfully completed on 17 Oct 2023,
i.e. `rucio list-rules` gives `OK` for each dataset.
Check via DAS web interface -> ok, all datasets fully present on T2B.
The local datasets are stored as follows:
- `/pnfs/iihe/cms/ph/sc4/store/data/Run2023C/ZeroBias/DQMIO/PromptReco-v1`
- `/pnfs/iihe/cms/ph/sc4/store/data/Run2023C/ZeroBias/DQMIO/PromptReco-v2`
- `/pnfs/iihe/cms/ph/sc4/store/data/Run2023C/ZeroBias/DQMIO/PromptReco-v3`
- `/pnfs/iihe/cms/ph/sc4/store/data/Run2023C/ZeroBias/DQMIO/PromptReco-v4`
- `/pnfs/iihe/cms/ph/sc4/store/data/Run2023D/ZeroBias/DQMIO/PromptReco-v1`
- `/pnfs/iihe/cms/ph/sc4/store/data/Run2023D/ZeroBias/DQMIO/PromptReco-v2`
- `/pnfs/iihe/cms/ph/sc4/store/data/Run2023E/ZeroBias/DQMIO/PromptReco-v1`
- `/pnfs/iihe/cms/ph/sc4/store/data/Run2023F/ZeroBias/DQMIO/PromptReco-v1`

### Conversion to plain ROOT files
Performed on 17 Oct 2023.
Note: seems to give non-deterministic errors when run with CMSSW 12.4.6 and python3,
while it runs fine with CMSSW 10.6.29 and python2, probably because of ROOT...
Note: seems to give non-deterministic errors when run in multithreaded mode in jobs,
instead run with one thread.
Size of resulting data: 23 GB.
Note: sorting of lumisections in the output files was added on 1 Nov 2023,
but rerunning not yet done, so lumisections in current ROOT files are likely unsorted.
Instead of rerunning the conversion to ROOT files, 
add sorting in conversion to parquet as well and only rerun that step.

### Conversion to parquet files
Performed on 17 Oct 2023.
Note: parquet files apparently are not automatically overwritten,
instead the script fails and raises an error upon writing attempt,
so need to manually remove parquet files before re-running.
Size of resulting data: 15 GB.
Note: sorting of lumisections in the output files was added on 1 Nov 2023,
and this step was rerun to have correctly sorted dataframes.

### Check of available lumisections
No missing lumisections in parquet files with respect to DAS,
except for 5037 missing lumisections in Run2023C-PromptReco-v2.
See DAS: part of the dataset is not available on both tape and disk.
However, this missing fraction corresponds to Run2023C-PromptReco-v3
(which was overlapping with the end part of Run2023C-PromptReco-v2 according to DAS),
so overall conclusion is that the data in the parquet files is complete and non-overlapping.

### Update for 2024 data
We perform a new iteration of the study (with more appropriate MEs) with early 2024 data.
At the time of writing, the following datasets are available on DAS:
- `/ZeroBias/Run2024B-PromptReco-v1/DQMIO` (34 GB, 156 files)
- `/ZeroBias/Run2024C-PromptReco-v1/DQMIO` (102 GB, 438 files)
- `/ZeroBias/Run2024D-PromptReco-v1/DQMIO` (86 GB, 371 files)
We do not include era 2024A, as the new MEs were only included by the end of that era.
Because of the relatively small dataset size, we do not use rucio transfers this time,
but directly copy the datasets to a local location using [this tool](https://github.com/LukaLambrecht/ML4DQMDC-PixelAE/blob/master/dqmio/copydastolocal/copy_das_to_local_set.py).
The remaining steps are similar to the above for the 2023 iteration.
