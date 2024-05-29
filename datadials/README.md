# Retreiving input data using DIALS

### Introduction
These scripts are simple wrappers around the [DIALS Python API](https://github.com/cms-DQM/dials-py/tree/develop) to retrieve a specified set of monitoring elements for specified datasets.
You can use these scripts directly, or as an inspiration on how to use the DIALS API for your own purposes.

### How to use
You need two `json` files as an input: one that specifies the dataset(s), and one that specifies the monitoring element(s) (MEs).
Both of these can contain regex-style special characters to include multiple datasets or MEs in a single line.
See `jsons/datasets_zerobias_2024_promptreco_das.json` and `jsons/mes_digioccupancy.json` for examples on the correct formatting.
Note: regex pattern matching is enabled by default, so some special characters (such as `+`) need to be escaped using a double backslash if you want to pass them literally.
See `jsons/mes_charge.json` for examples.

Next, run `python3 get_data_dials_loop.py` with the following options:
- `-d / --datasets`: path to json file with dataset names.
- `-m / --menames`: path to json file with ME names.
- `-t / --metype`: type of MEs (choose from "h1d" or "h2d"), needed for correct DIALS syntax. Note: --menames json files with mixed 1D and 2D MEs are not supported, they should be splitted and submitted separately.
- `-w / --workspace`: DIALS-workspace, see [the documentation](https://github.com/cms-DQM/dials-py?tab=readme-ov-file#workspace), default is `tracker`.
- `-o / --outputdir`: output directory.
- `--splitdatasets`: split into separate jobs per dataset (Note: any regex-style wildcards are transfered verbatim to the DIALS API, and hence jobs are split before rather than after the expansion of these wildcards).
- `--splitmes`: split into separate jobs per ME (Note: any regex-style wildcards are transfered verbatim to the DIALS API, and hence jobs are split before rather than after the expansion of these wildcards).
- `--runmode`: choose from `local` (to run in terminal) or `condor` (to run in job)

If everything goes well, one `parquet` file per ME and per dataset will be created in the output directory, containing a `pandas` `DataFrame` with the requested monitoring elements.
It is advised to always try first running locally (i.e. not in job submission) on a single dataset and for a single ME, to see if everything runs fine.
You can also add the option `--test` to truncate the data and make the test run much faster.

### Authentication
Authentication happens interactively through a device authentication flow.
See [the dials-py documentation](https://github.com/cms-DQM/dials-py/tree/develop?tab=readme-ov-file#usage) for more information.
I'm not sure how authentication is handled exactly in job submission, but it seems to work automagically if the credentials are already cached.
So first run `init_credentials.py` (or a small local test with `get_data_dials.py`) to do the authentication interactively, then afterwards jobs can be submitted without authentication issues.

### Progress checks
You can use the script `python3 check_jobstatus.py` to check the status and progress of jobs.
When all jobs are finished, you can use `check_lumis.py` to double check that all lumisections in the dataset on DAS are present in the produced `.parquet` files. 
Note: this check requires a valid grid proxy to use the DAS API.

### Transient errors
Sometimes transient errors might occur and crash a job. 
In those cases, you can add the `--resubmit` argument to `get_data_dials_loop.py` to submit only the jobs corresponding to datasets and MEs that do not have a corresponding output file yet.
If the errors persist, they are probably not transient and you might want to have a more detailed look.

### Example: getting all cluster charge MEs for 2024 data
As a practical example, we retrieve the cluster charge monitoring elements for 2024 data.

The first step is to make a json file listing the MEs of interest. You can find it in `jsons/mes_charge.json`.
Next, we make another json file listing the datasets of interest. The result is in `jsons/datasets_zerobias_2024_promptreco_das.json`.
Note: at the time of writing, 2024E data taking is ongoing.

Initialize the credentials using `python3 init_credentials.py`.

Submit the jobs using `python3 get_data_dials_loop.py -d jsons/datasets_zerobias_2024_promptreco_das.json -m jsons/mes_charge.json -t h1d -o output_test --splitdatasets --splitmes --runmode condor`.
While the jobs are running, you can check their status with `condor_q`, `python3 ../jobsubmission/jobcheck.py --notags` (when the jobs are finished, you can omit the `--notags` argument), and `python3 check_jobstatus.py -i cjob_get_data_out_*`

When all jobs are successfully finished, check the lumisections in the resulting files using `python3 check_lumis.py -d jsons/datasets_zerobias_2024_promptreco_das.json -p output_test/*.parquet`
(but make sure to use `voms-proxy-init --voms cms` before that if you don't have a valid proxy already).
