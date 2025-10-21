

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
import argparse
from metpy.calc import specific_humidity_from_dewpoint, relative_humidity_from_dewpoint, vertical_velocity_pressure, mixing_ratio_from_specific_humidity, relative_humidity_from_specific_humidity
from metpy.units import units
from metpy.constants import water_heat_vaporization, dry_air_spec_heat_press
import time
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
    get_inversion, get_inversion_numba,
    get_LCL,
    get_LTS,
    get_EIS, get_EIS_simplified,
    )

from namelist import cmip6_units, zerok, seconds_per_d

# endregion


# region get monthly BARRA-R2 inversionh, LCL, LTS, EIS
# 4 vars, 8 years, 12 months: NCPUs Used: 96; Memory Used: 1.0TB; Walltime Used: 00:11:29

vars = ['ECTEI'] # ['inversionh', 'LCL', 'LTS', 'EIS']

def get_mon_from_hour(var, year, month):
    # var = 'LTS'; year=2024; month=1
    print(f'#---------------- {var} {year} {month:02d}')
    
    ifile = f'data/sim/um/barra_r2/{var}/{var}_hourly_{year}{month:02d}.nc'
    ofile = f'data/sim/um/barra_r2/{var}/{var}_monthly_{year}{month:02d}.nc'
    
    ds_in = xr.open_dataset(ifile)[var]
    ds_out = ds_in.resample({'time': '1ME'}).mean(skipna=True).compute()
    
    if os.path.exists(ofile): os.remove(ofile)
    ds_out.to_netcdf(ofile)
    
    del ds_in, ds_out
    return f'Finished processing {ofile}'

joblib.Parallel(n_jobs=48)(joblib.delayed(get_mon_from_hour)(var, year, month) for var in vars for year in range(2016, 2024) for month in range(1, 13))


'''
#---- check
var = 'inversionh'
year = 2016
month = 6
ds_in = xr.open_dataset(f'data/sim/um/barra_r2/{var}/{var}_hourly_{year}{month:02d}.nc')[var]
ds_out = xr.open_dataset(f'data/sim/um/barra_r2/{var}/{var}_monthly_{year}{month:02d}.nc')[var]

ilat = 100
ilon = 100
print(np.nanmean(ds_in[:, ilat, ilon].values) - ds_out[0, ilat, ilon].values)

'''
# endregion
