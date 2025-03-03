

# qsub -I -q normal -l walltime=04:00:00,ncpus=1,mem=192GB,jobfs=192GB,storage=gdata/v46+scratch/v46
# module load python3/3.12.1
# /apps/python3/3.12.1/bin/python3


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

year=2016

CL_Frequency_fl = sorted(glob.glob(f'/scratch/v46/qg8515/data/obs/jaxa/clp/{year}??/CL_Frequency_{year}??.nc'))
CL_Frequency = xr.open_mfdataset(CL_Frequency_fl, combine='nested', concat_dim='time')

CL_Frequency_ann = CL_Frequency.CL_Frequency.chunk(chunks={'latitude': 100, 'longitude': 100, }).sum(dim='time').compute()

ofile = f'/scratch/v46/qg8515/data/obs/jaxa/clp/CL_Frequency_{year}.nc'
if os.path.exists(ofile): os.remove(ofile)
CL_Frequency_ann.to_netcdf(ofile)



'''
# check
CL_Frequency_ann = xr.open_dataset('scratch/data/obs/jaxa/clp/CL_Frequency_2016.nc')

(CL_Frequency_ann.CL_Frequency[0] == CL_Frequency_ann.CL_Frequency[1:].sum(axis=0)).all().values


'''
# endregion
