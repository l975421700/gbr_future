

# region import packages

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()

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


# region get era5 pl mon data

for var in ['q', 't', 'z']:
    # var = 'pv'
    print(var)
    
    fl = sorted([
        file for iyear in np.arange(1979, 2024, 1)
        for file in glob.glob(f'/g/data/rt52/era5/pressure-levels/monthly-averaged/{var}/{iyear}/*.nc')])
    # print(xr.open_dataset(fl[0])[var].units)
    
    era5_pl_mon = xr.open_mfdataset(fl, parallel=True).rename({'latitude': 'lat', 'longitude': 'lon'})[var]
    
    if var in ['q']:
        era5_pl_mon = era5_pl_mon * 1000
    elif var in ['t']:
        era5_pl_mon = era5_pl_mon - zerok
    elif var in ['z']:
        era5_pl_mon = era5_pl_mon / 9.80665
    
    era5_pl_mon_alltime = mon_sea_ann(
        var_monthly=era5_pl_mon, lcopy=False, mm=True, sm=True, am=True,)
    
    ofile = f'data/obs/era5/mon/era5_pl_mon_alltime_{var}.pkl'
    if os.path.exists(ofile): os.remove(ofile)
    with open(ofile, 'wb') as f:
        pickle.dump(era5_pl_mon_alltime, f)
    
    del era5_pl_mon, era5_pl_mon_alltime




'''
# check
era5_pl_mon_alltime = {}
for var in ['pv', 'q', 'r', 't', 'u', 'v', 'w', 'z']:
    # var = 'pv'
    print(var)
    
    with open(f'data/obs/era5/mon/era5_pl_mon_alltime_{var}.pkl', 'rb') as f:
        era5_pl_mon_alltime[var] = pickle.load(f)
    
    print(era5_pl_mon_alltime[var]['mon'].units)
    del era5_pl_mon_alltime[var]

'''
# endregion
