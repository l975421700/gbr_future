

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


# region get BARRA-C2 alltime hourly data


for var in ['clivi', 'clwvi', 'prw']:
    # var = 'cll'
    # ['cll', 'clh', 'clm', 'clt', 'pr', 'tas']
    print(f'#-------------------------------- {var}')
    
    fl = sorted(glob.glob(f'scratch/data/sim/um/barra_c2/{var}/{var}_hourly_*.nc'))
    barra_c2_hourly = xr.open_mfdataset(fl)[var].sel(time=slice('1979', '2023'))
    barra_c2_hourly_alltime = mon_sea_ann(
        var_monthly=barra_c2_hourly, lcopy=False, mm=True, sm=True, am=True)
    
    ofile = f'data/sim/um/barra_c2/barra_c2_hourly_alltime_{var}.pkl'
    if os.path.exists(ofile): os.remove(ofile)
    with open(ofile,'wb') as f:
        pickle.dump(barra_c2_hourly_alltime, f)
    
    del barra_c2_hourly, barra_c2_hourly_alltime



'''
#-------------------------------- check
var = 'tas'
with open(f'data/sim/um/barra_c2/barra_c2_hourly_alltime_{var}.pkl','rb') as f:
    barra_c2_hourly_alltime = pickle.load(f)

fl = sorted(glob.glob(f'scratch/data/sim/um/barra_c2/{var}/{var}_hourly_*.nc'))
ifile = -1
ds = xr.open_dataset(fl[ifile])

print((barra_c2_hourly_alltime['mon'][ifile] == ds[var].squeeze()).all().values)
'''
# endregion
