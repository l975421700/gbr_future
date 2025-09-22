

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


# region get BARPA-C alltime hourly data

years = '2016'
yeare = '2023'
for var in ['rlut', 'rlutcs', 'pr', 'cll', 'clm', 'clh', 'clt', 'rsut', 'rsutcs', 'clwvi', 'clivi']:
    # var = 'cll'
    # ['clivi', 'clwvi', 'prw', 'cll', 'clh', 'clm', 'clt', 'pr', 'tas']
    print(f'#-------------------------------- {var}')
    
    fl = sorted(glob.glob(f'data/sim/um/barpa_c/{var}/{var}_hourly_*.nc'))
    barpa_c_hourly = xr.open_mfdataset(fl)[var].sel(time=slice(years, yeare))
    barpa_c_hourly_alltime = mon_sea_ann(
        var_monthly=barpa_c_hourly, lcopy=False, mm=True, sm=True, am=True)
    
    ofile = f'data/sim/um/barpa_c/barpa_c_hourly_alltime_{var}.pkl'
    if os.path.exists(ofile): os.remove(ofile)
    with open(ofile,'wb') as f:
        pickle.dump(barpa_c_hourly_alltime, f)
    
    del barpa_c_hourly, barpa_c_hourly_alltime



'''
#-------------------------------- check
var = 'pr'
with open(f'data/sim/um/barpa_c/barpa_c_hourly_alltime_{var}.pkl','rb') as f:
    barpa_c_hourly_alltime = pickle.load(f)

fl = sorted(glob.glob(f'scratch/data/sim/um/barpa_c/{var}/{var}_hourly_*.nc'))
ifile = -1
ds = xr.open_dataset(fl[ifile])

print((barpa_c_hourly_alltime['mon'][ifile] == ds[var].squeeze()).all().values)


'''
# endregion
