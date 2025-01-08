

# qsub -I -q normal -l walltime=04:00:00,ncpus=1,mem=192GB,storage=gdata/v46+scratch/v46
# module load python3/3.12.1
# /apps/python3/3.12.1/bin/python3


# region get command line options

import argparse
parser=argparse.ArgumentParser()
parser.add_argument('-y', '--year', type=int, required=True,)
parser.add_argument('-m', '--month', type=int, required=True,)
args = parser.parse_args()

year=args.year
month=args.month

# endregion


# region import packages

import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
import pandas as pd
import glob
from datetime import datetime
import os
import calendar

# endregion


# not yet



