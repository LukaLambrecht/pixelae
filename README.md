# Pixel-AE: anomaly detection in pixel tracker DQM data

### Introduction
To do

### Input data
Most of the functionality in this repository starts from per-lumisection DQM data,
stored in the form of [pandas dataframes](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) in [parquet](https://parquet.apache.org/) files.

The retrieval of the input data is not within the scope of this repository,
but here are some useful links:
- For data retrieval from DIALS \[[1](https://cmsdials.web.cern.ch/?ws=tracker), [2](https://gitlab.cern.ch/cms-dqmdc/services/dials-service)\] (recommended), see [dialstools/datasets](https://github.com/LukaLambrecht/dialstools/tree/main/datasets).
- For data retrieval directly from the DQMIO datasets centrally maintained by CMS (no longer recommended but might work as backup), see [dqmiotools/datasets](https://github.com/LukaLambrecht/dqmiotools/tree/main/datasets).
