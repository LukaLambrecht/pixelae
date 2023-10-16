#!/usr/bin/env bash

# make rucio requests
# see here: https://t2bwiki.iihe.ac.be/Rucio
# first make a proxy before running this script!

# set environment
source /cvmfs/cms.cern.ch/cmsset_default.sh
source /cvmfs/cms.cern.ch/rucio/setup-py3.sh
export RUCIO_ACCOUNT=`whoami`

# make list of datasets
declare -a datasets=(
  "/ZeroBias/Run2023C-PromptReco-v1/DQMIO"
  "/ZeroBias/Run2023C-PromptReco-v2/DQMIO"
  "/ZeroBias/Run2023C-PromptReco-v3/DQMIO"
  "/ZeroBias/Run2023C-PromptReco-v4/DQMIO"
  "/ZeroBias/Run2023D-PromptReco-v1/DQMIO"
  "/ZeroBias/Run2023D-PromptReco-v2/DQMIO"
  "/ZeroBias/Run2023E-PromptReco-v1/DQMIO"
  "/ZeroBias/Run2023F-PromptReco-v1/DQMIO" 
)

# make rucio requests
for dataset in "${datasets[@]}"
do
    rucio add-rule cms:"$dataset" 1 T2_BE_IIHE --asynchronous --ask-approval --lifetime 7776000
done
