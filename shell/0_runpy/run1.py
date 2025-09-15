

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
import argparse
from metpy.calc import specific_humidity_from_dewpoint, relative_humidity_from_dewpoint, vertical_velocity_pressure, mixing_ratio_from_specific_humidity, relative_humidity_from_specific_humidity
from metpy.units import units
import time

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
    get_inversion, get_inversion_numba,
    get_LCL,
    get_LTS,
    get_EIS, get_EIS_simplified,
    )

from namelist import zerok, seconds_per_d

# endregion


# region get monthly BARRA-C2 inversionh, LCL, LTS, EIS

vars = ['inversionh', 'LCL', 'LTS', 'EIS']

def get_mon_from_hour(var, year, month):
    # var = 'LTS'; year=2024; month=1
    print(f'#---------------- {var} {year} {month:02d}')
    
    ifile = f'data/sim/um/barra_c2/{var}/{var}_hourly_{year}{month:02d}.nc'
    ofile = f'data/sim/um/barra_c2/{var}/{var}_monthly_{year}{month:02d}.nc'
    
    ds_in = xr.open_dataset(ifile)[var]
    ds_out = ds_in.resample({'time': '1ME'}).mean(skipna=True).compute()
    
    if os.path.exists(ofile): os.remove(ofile)
    ds_out.to_netcdf(ofile)
    
    del ds_in, ds_out
    return f'Finished processing {ofile}'

joblib.Parallel(n_jobs=96)(joblib.delayed(get_mon_from_hour)(var, year, month) for var in vars for year in range(2016, 2024) for month in range(1, 13))


# endregion

