

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


# region import data

# year=2016
# month=11

CL_Frequency_fl = sorted(glob.glob(f'/scratch/v46/qg8515/data/obs/jaxa/clp/{year}{month:02d}/??/CL_Frequency_{year}{month:02d}??.nc'))
CL_Frequency = xr.open_mfdataset(CL_Frequency_fl)

CL_Frequency_mon = CL_Frequency.CL_Frequency.chunk(chunks={'time': 31, 'latitude': 10, 'longitude': 10, }).sum(dim='time').compute()

ofile = f'/scratch/v46/qg8515/data/obs/jaxa/clp/{year}{month:02d}/CL_Frequency_{year}{month:02d}.nc'
if os.path.exists(ofile): os.remove(ofile)
CL_Frequency_mon.to_netcdf(ofile)


# endregion
