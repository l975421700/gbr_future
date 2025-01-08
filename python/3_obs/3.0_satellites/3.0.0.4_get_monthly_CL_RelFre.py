

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
# month=1

CL_Frequency = xr.open_dataset(f'/scratch/v46/qg8515/data/obs/jaxa/clp/{year}{month:02d}/CL_Frequency_{year}{month:02d}.nc').CL_Frequency

# endregion


# region get CL_RelFreq

CL_RelFreq = xr.DataArray(
    name='CL_RelFreq',
    data=np.zeros((CL_Frequency.shape[0]-2,
                   CL_Frequency.shape[1], CL_Frequency.shape[2])),
    dims=['types', 'latitude', 'longitude',],
    coords={
        'types': CL_Frequency.types.values[1:-1],
        'latitude': CL_Frequency.latitude.values,
        'longitude': CL_Frequency.longitude.values,})


for itype in CL_Frequency.types.values[1:-1]:
    print(itype)
    CL_RelFreq.loc[{'types': itype}][:, :] = CL_Frequency.sel(types=itype) / CL_Frequency.sel(types='finite') * 100

ofile = f'/scratch/v46/qg8515/data/obs/jaxa/clp/{year}{month:02d}/CL_RelFreq_{year}{month:02d}.nc'
if os.path.exists(ofile): os.remove(ofile)
CL_RelFreq.to_netcdf(ofile)


'''
# check
from scipy import stats
year=2016
CL_RelFreq = xr.open_dataset(f'/scratch/v46/qg8515/data/obs/jaxa/clp/CL_RelFreq_{year}.nc').CL_RelFreq
np.min(CL_RelFreq.sum(dim='types').values)
np.max(CL_RelFreq.sum(dim='types').values)
'''
# endregion
