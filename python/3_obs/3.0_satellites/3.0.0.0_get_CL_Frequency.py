

# qsub -I -q normal -l walltime=04:00:00,ncpus=1,mem=192GB,jobfs=192GB,storage=gdata/v46+scratch/v46


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


# region get CL_Frequency

# year=2016
# month=12

ISCCP_types = {'Clear': 0,
               'Cirrus': 1, 'Cirrostratus': 2, 'Deep convection': 3,
               'Altocumulus': 4, 'Altostratus': 5, 'Nimbostratus':6,
               'Cumulus':7, 'Stratocumulus': 8, 'Stratus': 9,
               'Unknown':10}

daterange = pd.date_range(
    start=f"{year}-{month:02d}-01",
    end=f"{year}-{month:02d}-{calendar.monthrange(year, month)[1]}",)

for idate in daterange:
    # idate = daterange[0]
    print(idate)
    day=str(idate)[8:10]
    
    clp_fl = sorted(glob.glob(f'/scratch/v46/qg8515/data/obs/jaxa/clp/{year}{month:02d}/{day}/*/CLP_{year}{month:02d}{day}????.nc'))
    clp_ds = xr.open_mfdataset(clp_fl)
    CLTYPE_values = clp_ds.CLTYPE.values
    
    CL_Frequency = xr.DataArray(
        name='CL_Frequency',
        data=np.zeros((1, 12, clp_ds.CLTYPE.shape[1], clp_ds.CLTYPE.shape[2])),
        dims=['time', 'types', 'latitude', 'longitude',],
        coords={
            'time': [datetime.strptime(f'{year}-{month:02d}-{day}', '%Y-%m-%d')],
            'types': ['finite'] + list(ISCCP_types.keys()),
            'latitude': clp_ds.CLTYPE.latitude.values,
            'longitude': clp_ds.CLTYPE.longitude.values,})
    CL_Frequency.loc[{'types': 'finite'}][0] = np.isfinite(CLTYPE_values).sum(axis=0)
    for itype in list(ISCCP_types.keys()):
        CL_Frequency.loc[{'types': itype}][0] = (CLTYPE_values == ISCCP_types[itype]).sum(axis=0)
    
    print((CL_Frequency[0, 0] == CL_Frequency[0, 1:].sum(axis=0)).all().values)
    print(CL_Frequency[0, 0].sum().values)
    
    ofile = f'/scratch/v46/qg8515/data/obs/jaxa/clp/{year}{month:02d}/{day}/CL_Frequency_{year}{month:02d}{day}.nc'
    if os.path.exists(ofile): os.remove(ofile)
    CL_Frequency.to_netcdf(ofile)


'''
aaa = xr.open_dataset('/scratch/v46/qg8515/data/obs/jaxa/clp/201601/01/CL_Frequency_20160101.nc')

himawari_fl = sorted(glob.glob('/home/563/qg8515/data/obs/jaxa/clp/*/*/*/NC_*'))
clp_fl = sorted(glob.glob('/scratch/v46/qg8515/data/obs/jaxa/clp/*/*/*/CLP_*'))
print(len(himawari_fl))
print(len(clp_fl))
'''
# endregion

