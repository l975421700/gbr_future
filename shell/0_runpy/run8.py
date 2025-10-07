

# region import packages

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
import joblib

# management
import os
import sys  # print(sys.path)
sys.path.append(os.getcwd() + '/code/gbr_future/module')
import glob
import pickle
import datetime
# import psutil
# process = psutil.Process()
# print(process.memory_info().rss / 2**30)

from calculations import (
    mon_sea_ann,
    )

from namelist import zerok, seconds_per_d

# endregion


# region get BARPA-R alltime hourly data
# Memory Used: 47.5GB, Walltime Used: 00:42:54

years = '2016'
yeare = '2023'
for var in ['clivi']:
    # var = 'cll'
    # 'cll', 'clm', 'clh', 'clt', 'rsut', 'rsutcs', 'clwvi', 'clivi', 'rlut', 'rlutcs', 'pr', 'hfls', 'hfss', 'hurs', 'huss'
    print(f'#-------------------------------- {var}')
    
    fl = sorted(glob.glob(f'data/sim/um/barpa_r/{var}/{var}_hourly_*.nc'))
    barpa_r_hourly = xr.open_mfdataset(fl)[var].sel(time=slice(years, yeare))
    barpa_r_hourly_alltime = mon_sea_ann(
        var_monthly=barpa_r_hourly, lcopy=False, mm=True, sm=True, am=True)
    
    ofile = f'data/sim/um/barpa_r/barpa_r_hourly_alltime_{var}.pkl'
    if os.path.exists(ofile): os.remove(ofile)
    with open(ofile,'wb') as f:
        pickle.dump(barpa_r_hourly_alltime, f)
    
    del barpa_r_hourly, barpa_r_hourly_alltime




'''
#-------------------------------- check
ifile = -1
for var in ['rlut', 'rlutcs', 'pr', 'cll', 'clm', 'clh', 'clt', 'rsut', 'rsutcs', 'clwvi', 'clivi']:
    print(f'#-------------------------------- {var}')
    
    with open(f'data/sim/um/barpa_c/barpa_c_hourly_alltime_{var}.pkl','rb') as f:
        barpa_c_hourly_alltime = pickle.load(f)
    
    fl = sorted(glob.glob(f'data/sim/um/barpa_c/{var}/{var}_hourly_*.nc'))
    ds = xr.open_dataset(fl[ifile])[var]
    print((barpa_c_hourly_alltime['mon'][ifile] == ds.squeeze()).all().values)
    
    del ds, barpa_c_hourly_alltime




'''
# endregion
