#!/usr/bin/env python3


# Initialize DIALS API credentials
# - The credentials are stored in ./.cache/creds
# - They remain valid for a couple of hours,
#   after which you need to rerun this script.


# dials imports
from cmsdials.auth.bearer import Credentials


if __name__=='__main__':

  # do authentication
  creds = Credentials.from_creds_file()
