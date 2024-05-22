# Retreiving input data using DIALS

### Introduction
These scripts are simple wrappers around the [DIALS Python API](https://github.com/cms-DQM/dials-py/tree/develop) to retrieve a specified set of monitoring elements for specified datasets.
You can use these scripts directly, or as an inspiration on how to use the DIALS API for your own purposes.

### How to use
You need two `json` files as an input: one that specifies the dataset(s), and one that specifies the monitoring element(s) (MEs).
Both of these can contain regex-style special characters to include multiple datasets or MEs in a single line.
See `jsons/datasets_zerobias_2024_promptreco_das.json` and `jsons/mes_digioccupancy.json` for examples on the correct formatting.

Next, run `python3 get_data_dials_loop.py` with the following options:
- `-d / --datasets`: path to json file with dataset names
- `-m / --menames`: path to json file with ME names
- `-o / --outputdir`: output directory
- `--splitdatasets`: split into separate jobs per dataset
- `--splitmes`: split into separate jobs per ME
- `--runmode`: choose from `local` (to run in terminal) or `condor` (to run in job)

If everything goes well, one `parquet` file per ME and per dataset will be created in the output directory, containing a `DataFrame` with the requested monitoring elements.
It is advised to always try first running locally (i.e. not in job submission) on a single dataset and for a single ME, to see if everything runs fine.
You can also add the option `--test` to truncate the data and make the test run much faster.

### Authentication
Authentication happens interactively through a device authentication flow.
See [the dials-py documentation](https://github.com/cms-DQM/dials-py/tree/develop?tab=readme-ov-file#usage) for more information.
I'm not sure how authentication is handled exactly in job submission, but it seems to work automagically if the credentials are already cached.
So first run a small test locally to do the authentication, then afterwards jobs can be submitted without authentication issues.

### Gateway time-out errors
It is observed that jobs sometimes go into error with following message:
```
INFO: Api raw response: <html>
<head><title>504 Gateway Time-out</title></head>
<body>
<center><h1>504 Gateway Time-out</h1></center>
<hr><center>nginx/1.25.5</center>
</body>
</html>
```
followed by a JSON decoder error.

While this is still under investigation, it seems this can be mitigated by not having too many parallel jobs.
So try to aim for a smaller number of longer-running jobs rather than a high number of short jobs.
If all else fails, try to run locally (e.g. in a screen session).
