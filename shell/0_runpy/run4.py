

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
import pickle

import sys  # print(sys.path)
sys.path.append('/home/563/qg8515/code/gbr_future/module')
from calculations import mon_sea_ann

# endregion


# region get alltime hourly count of each cloud type
# Memory Used: 221.51GB, Walltime Used: 04:01:46


min_lon, max_lon, min_lat, max_lat = [110.58, 157.34, -43.69, -7.01]
fl = sorted(glob.glob(f'data/obs/jaxa/clp/??????/cltype_hourly_count_??????.nc'))
cltype_hourly_count = xr.open_mfdataset(fl, parallel=True).cltype_hourly_count.sel(lon=slice(min_lon, max_lon), lat=slice(max_lat, min_lat))
cltype_hourly_count_alltime = mon_sea_ann(
    var_monthly=cltype_hourly_count, lcopy=False, mm=True, sm=True, am=True)

ofile='data/obs/jaxa/clp/cltype_hourly_count_alltime.pkl'
if os.path.exists(ofile): os.remove(ofile)
with open(ofile, 'wb') as f:
    pickle.dump(cltype_hourly_count_alltime, f)




'''
#-------------------------------- check
with open('data/obs/jaxa/clp/cltype_hourly_count_alltime.pkl', 'rb') as f:
    cltype_hourly_count_alltime = pickle.load(f)
fl = sorted(glob.glob(f'data/obs/jaxa/clp/??????/cltype_hourly_count_??????.nc'))
ifile = 10
ds = xr.open_dataset(fl[ifile]).cltype_hourly_count
print((cltype_hourly_count_alltime['mon'][ifile] == ds).all().values)


#-------------------------------- check 2
with open('data/obs/jaxa/clp/cltype_hourly_count_alltime.pkl', 'rb') as f:
    cltype_hourly_count_alltime = pickle.load(f)
with open('data/obs/jaxa/clp/cltype_count_alltime.pkl', 'rb') as f:
    cltype_count_alltime = pickle.load(f)

min_lon, max_lon, min_lat, max_lat = [110.58, 157.34, -43.69, -7.01]
ialltime='mon'
itype = 'Stratocumulus'
itime=2
print((cltype_count_alltime[ialltime][itime].loc[{'types': itype}].sel(lon=slice(min_lon, max_lon), lat=slice(max_lat, min_lat)) == cltype_hourly_count_alltime[ialltime][itime].loc[{'types': itype}].sum(dim='hour')).all().values)


#-------------------------------- check 3
with open('data/obs/jaxa/clp/cltype_hourly_count_alltime.pkl', 'rb') as f:
    cltype_hourly_count_alltime = pickle.load(f)

itime=-1
ihour=3
ilat = 100
ilon = 100

for ialltime in cltype_hourly_count_alltime.keys():
    # ialltime='ann'
    print(f'#-------------------------------- {ialltime}')
    
    print(cltype_hourly_count_alltime[ialltime][itime, ihour, 0, ilat, ilon].values == cltype_hourly_count_alltime[ialltime][itime, ihour, 1:, ilat, ilon].sum(dim='types').values)
    print((cltype_hourly_count_alltime[ialltime][itime, ihour, 0, ilat, ilon].values - cltype_hourly_count_alltime[ialltime][itime, ihour, 1:, ilat, ilon].sum(dim='types').values) / cltype_hourly_count_alltime[ialltime][itime, ihour, 0, ilat, ilon].values)


'''
# endregion
