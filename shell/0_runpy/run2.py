

# region import packages

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
# from dask.diagnostics import ProgressBar
# pbar = ProgressBar()
# pbar.register()
import joblib

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


# region get era5 alltime hourly data


for var in ['mtnswrf', 'mtdwswrf', 'mtnlwrf']:
    # var = 'lcc'
    # ['tcwv', 'tclw', 'tciw', 'lcc', 'mcc', 'hcc', 'tcc', 'tp', '2t']
    print(f'#-------------------------------- {var}')
    odir = f'scratch/data/obs/era5/{var}'
    
    if var == '2t': var='t2m'
    if var == '10si': var='si10'
    if var == '2d': var='d2m'
    if var == '10u': var='u10'
    if var == '10v': var='v10'
    if var == '100u': var='u100'
    if var == '100v': var='v100'
    
    fl = sorted(glob.glob(f'{odir}/{var}_hourly_*.nc'))
    era5_hourly = xr.open_mfdataset(fl)[var].sel(time=slice('1979', '2023'))
    era5_hourly_alltime = mon_sea_ann(
        var_monthly=era5_hourly, lcopy=False, mm=True, sm=True, am=True)
    
    ofile = f'data/obs/era5/hourly/era5_hourly_alltime_{var}.pkl'
    if os.path.exists(ofile): os.remove(ofile)
    with open(ofile,'wb') as f:
        pickle.dump(era5_hourly_alltime, f)
    
    del era5_hourly, era5_hourly_alltime




'''
#-------------------------------- check
ifile = 10
for var in ['tcwv', 'tclw', 'tciw']:
    # var = 'lcc'
    # ['lcc', 'mcc', 'hcc', 'tcc', 'tp', '2t']
    print(f'#-------------------------------- {var}')
    odir = f'scratch/data/obs/era5/{var}'
    
    if var == '2t': var='t2m'
    if var == '10si': var='si10'
    if var == '2d': var='d2m'
    if var == '10u': var='u10'
    if var == '10v': var='v10'
    if var == '100u': var='u100'
    if var == '100v': var='v100'
    
    fl = sorted(glob.glob(f'{odir}/{var}_hourly_*.nc'))
    ds = xr.open_dataset(fl[ifile])
    
    with open(f'data/obs/era5/hourly/era5_hourly_alltime_{var}.pkl','rb') as f:
        era5_hourly_alltime = pickle.load(f)
    
    print((era5_hourly_alltime['mon'][ifile] == ds[var].squeeze()).all().values)
    del era5_hourly_alltime


'''
# endregion
