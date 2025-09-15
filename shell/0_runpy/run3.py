

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
import argparse
import calendar
from metpy.calc import geopotential_to_height, relative_humidity_from_dewpoint
from metpy.units import units

# management
import os
import sys  # print(sys.path)
sys.path.append(os.getcwd() + '/code/gbr_future/module')
import glob
import pickle
import datetime

from calculations import (
    mon_sea_ann,
    get_inversion, get_inversion_numba,
    get_LCL,
    get_LTS,
    get_EIS, get_EIS_simplified,
    )

from namelist import cmip6_units, zerok, seconds_per_d, cmip6_era5_var

# endregion


# region get monthly era5 inversionh, LCL, LTS, EIS
# 4 vars and 12 month: Memory Used: 960.0GB; Walltime Used: 00:04:30

vars = ['inversionh', 'LCL', 'LTS', 'EIS']

def get_mon_from_hour(var, year, month):
    # var = 'LTS'; year=2024; month=1
    print(f'#---------------- {var} {year} {month:02d}')
    
    ifile = f'data/obs/era5/hourly/{var}/{var}_hourly_{year}{month:02d}.nc'
    ofile = f'data/obs/era5/hourly/{var}/{var}_monthly_{year}{month:02d}.nc'
    
    ds_in = xr.open_dataset(ifile)[var]
    ds_out = ds_in.resample({'time': '1ME'}).mean(skipna=True).compute()
    
    if os.path.exists(ofile): os.remove(ofile)
    ds_out.to_netcdf(ofile)
    
    del ds_in, ds_out
    return f'Finished processing {ofile}'

joblib.Parallel(n_jobs=48)(joblib.delayed(get_mon_from_hour)(var, year, month) for var in vars for year in range(2024, 2025) for month in range(1, 13))


'''
#---- check
var = 'EIS'
year = 2024
month = 1
ds_in = xr.open_dataset(f'data/obs/era5/hourly/{var}/{var}_hourly_{year}{month:02d}.nc')[var]
ds_out = xr.open_dataset(f'data/obs/era5/hourly/{var}/{var}_monthly_{year}{month:02d}.nc')[var]

ilat = 100
ilon = 100
print(np.nanmean(ds_in[:, ilat, ilon].values) - ds_out[0, ilat, ilon].values)

'''
# endregion

