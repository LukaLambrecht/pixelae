# Pixel-AE: anomaly detection in the CMS pixel tracker DQM data

<img src="docs/nmf_example.png" width="1000" height="125" />

### Introduction
This is a repository for anomaly detection studies in the DQM data of the CMS pixel tracker.

It was originally developed with an autoencoder in mind (hence the ae in pixelae),
but the current studies use non-negative matrix factorization (NMF) instead.
See [studies/clusters_2024](https://github.com/LukaLambrecht/pixelae/tree/main/studies/clusters_2024)
for the specific implementation of the latest ongoing studies (at the time of writing),
and in particular [studies/clusters_2024/nmf](https://github.com/LukaLambrecht/pixelae/tree/main/studies/clusters_2024/nmf)
for the NMF definition, training, and evaluation.

Regardless of the specific method, this repository contains some [tools](https://github.com/LukaLambrecht/pixelae/tree/main/tools)
(e.g. for loading the input data, getting and handling relevant info from OMS, and common data manipulations).
There is also some code for [plotting](https://github.com/LukaLambrecht/pixelae/tree/main/plotting) (to be improved),
[job submission to HTCondor](https://github.com/LukaLambrecht/pixelae/tree/main/jobsubmission) from `lxplus`,
and handling [automasks](https://github.com/LukaLambrecht/pixelae/tree/main/automasking).

### Input data
Most of the functionality in this repository starts from per-lumisection DQMIO data,
stored in the form of [pandas dataframes](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) in [parquet](https://parquet.apache.org/) files.

The retrieval and parsing of the input data is not within the scope of this repository,
but here are some useful links:
- For data retrieval from DIALS \[[1](https://cmsdials.web.cern.ch/?ws=tracker), [2](https://gitlab.cern.ch/cms-dqmdc/services/dials-service)\] (recommended), see [dialstools/datasets](https://github.com/LukaLambrecht/dialstools/tree/main/datasets).
- For data retrieval directly from the DQMIO datasets centrally maintained by CMS (no longer recommended but might work as backup), see [dqmiotools/datasets](https://github.com/LukaLambrecht/dqmiotools/tree/main/datasets).
