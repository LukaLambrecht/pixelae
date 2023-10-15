# Collect input data

Read datasets from DAS and extract MEs of interest.
Original idea was to read the DQMIO datasets remotely and extract the data directly.
However, the datasets seem to be not or very instably available via DAS.
Therefore, new approach is to request a temporary copy of the datasets on T2B via rucio.
See https://t2bwiki.iihe.ac.be/Rucio.
From there, the relevant data can be extracted and stored locally.
