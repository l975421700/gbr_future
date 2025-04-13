

# region import packages

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
from cdo import Cdo
cdo=Cdo()
import tempfile
import joblib
import argparse

# management
import os
import sys  # print(sys.path)
sys.path.append(os.getcwd() + '/code/gbr_future/module')
import glob
import pickle
import datetime
from calculations import (
    mon_sea_ann,
    )

from namelist import cmip6_units, zerok, seconds_per_d

# endregion


# region get BARRA-R2 alltime hourly data


for var in ['rsdt', 'rsut', 'rlut']:
    # var = 'cll'
    print(f'#-------------------------------- {var}')
    
    fl = sorted(glob.glob(f'scratch/data/sim/um/barra_r2/{var}/{var}_hourly_*.nc'))
    barra_r2_hourly = xr.open_mfdataset(fl)[var].sel(time=slice('1979', '2023'))
    barra_r2_hourly_alltime = mon_sea_ann(
        var_monthly=barra_r2_hourly, lcopy=False, mm=True, sm=True, am=True)
    
    ofile = f'data/sim/um/barra_r2/barra_r2_hourly_alltime_{var}.pkl'
    if os.path.exists(ofile): os.remove(ofile)
    with open(ofile,'wb') as f:
        pickle.dump(barra_r2_hourly_alltime, f)
    
    del barra_r2_hourly, barra_r2_hourly_alltime



'''
#-------------------------------- check
var = 'prw' # ['clivi', 'clwvi', 'prw', 'cll', 'clh', 'clm', 'clt', 'pr', 'tas']
with open(f'data/sim/um/barra_r2/barra_r2_hourly_alltime_{var}.pkl','rb') as f:
    barra_r2_hourly_alltime = pickle.load(f)

fl = sorted(glob.glob(f'scratch/data/sim/um/barra_r2/{var}/{var}_hourly_*.nc'))
ifile = 3
ds = xr.open_dataset(fl[ifile])[var]

print((barra_r2_hourly_alltime['mon'][ifile] == ds.squeeze()).all().values)
del barra_r2_hourly_alltime
'''
# endregion
