

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


# region get alltime hourly frequency of each cloud type
# Memory Used: 220.64GB; Walltime Used: 11:06:50

with open('data/obs/jaxa/clp/cltype_hourly_count_alltime.pkl', 'rb') as f:
    cltype_hourly_count_alltime = pickle.load(f)

cltype_hourly_frequency_alltime = {}
for ialltime in ['mon', 'sea', 'ann', 'mm', 'sm', 'am']:
    # ialltime = 'mon'
    print(f'#-------------------------------- {ialltime}')
    
    cltype_hourly_frequency_alltime[ialltime] = xr.zeros_like(cltype_hourly_count_alltime[ialltime][:, :, 1:]).rename('cltype_hourly_frequency')
    
    for itype in cltype_hourly_frequency_alltime[ialltime].types.values:
        # itype='Stratocumulus'
        print(f'#---------------- {itype}')
        cltype_hourly_frequency_alltime[ialltime].loc[{'types': itype}][:] = (cltype_hourly_count_alltime[ialltime].loc[{'types': itype}] / cltype_hourly_count_alltime[ialltime].loc[{'types': 'finite'}] * 100).compute().astype(np.float32)

ofile='data/obs/jaxa/clp/cltype_hourly_frequency_alltime.pkl'
if os.path.exists(ofile): os.remove(ofile)
with open(ofile, 'wb') as f:
    pickle.dump(cltype_hourly_frequency_alltime, f)



'''
#-------------------------------- check 1
# hourly frequency has problem, to be fixed

with open('data/obs/jaxa/clp/cltype_hourly_count_alltime.pkl', 'rb') as f:
    cltype_hourly_count_alltime = pickle.load(f)
with open('data/obs/jaxa/clp/cltype_hourly_frequency_alltime.pkl', 'rb') as f:
    cltype_hourly_frequency_alltime = pickle.load(f)

for ialltime in list(cltype_hourly_count_alltime.keys()):
    print(ialltime)
    print(cltype_hourly_count_alltime['am'].loc[{'types': 'Unknown'}].sum().values)


ialltime = 'am'
itype = 'Stratocumulus'
itime=-1
ihour=3

data1 = cltype_hourly_frequency_alltime[ialltime].loc[{'types': itype}][itime, ihour].values
data2 = (cltype_hourly_count_alltime[ialltime].loc[{'types': itype}][itime, ihour] / cltype_hourly_count_alltime[ialltime].loc[{'types': 'finite'}][itime, ihour] * 100).compute().astype(np.float32).values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())

ialltime='am'
itype = 'Stratocumulus'
itime = -1
ihour = 3
ilat = 100
ilon = 100

print(cltype_hourly_frequency_alltime[ialltime].loc[{'types': itype}][itime, ihour, ilat, ilon].values == (cltype_hourly_count_alltime[ialltime].loc[{'types': itype}][itime, ihour, ilat, ilon] / cltype_hourly_count_alltime[ialltime].loc[{'types': 'finite'}][itime, ihour, ilat, ilon] * 100).compute().astype(np.float32).values)


#-------------------------------- check 2
with open('data/obs/jaxa/clp/cltype_frequency_alltime.pkl', 'rb') as f:
    cltype_frequency_alltime = pickle.load(f)
with open('data/obs/jaxa/clp/cltype_hourly_frequency_alltime.pkl', 'rb') as f:
    cltype_hourly_frequency_alltime = pickle.load(f)
min_lon, max_lon, min_lat, max_lat = [110.58, 157.34, -43.69, -7.01]

ialltime = 'am'
itype = 'Stratocumulus'
itime=-1

data1 = cltype_frequency_alltime[ialltime].loc[{'types': itype}][itime].sel(lon=slice(min_lon, max_lon), lat=slice(max_lat, min_lat)).values
data2 = cltype_hourly_frequency_alltime[ialltime].loc[{'types': itype}][itime].mean(dim='hour', skipna=True).values
print(np.max(np.abs(data1[np.isfinite(data2) & np.isfinite(data1)] - data2[np.isfinite(data2) & np.isfinite(data1)])))


#-------------------------------- check 3
with open('data/obs/jaxa/clp/cltype_hourly_frequency_alltime.pkl', 'rb') as f:
    cltype_hourly_frequency_alltime = pickle.load(f)
with open('data/obs/jaxa/clp/cltype_hourly_count_alltime.pkl', 'rb') as f:
    cltype_hourly_count_alltime = pickle.load(f)

for ialltime in list(cltype_hourly_frequency_alltime.keys())[1:]:
    # ialltime='ann'
    print(f'#-------------------------------- {ialltime}')
    itime = -1
    ihour = 3
    ilat = 100
    ilon = 100
    print(cltype_hourly_frequency_alltime[ialltime][itime, ihour, :, ilat, ilon].values)
    print(cltype_hourly_frequency_alltime[ialltime][itime, ihour, :, ilat, ilon].values.sum())


for ialltime in list(cltype_hourly_frequency_alltime.keys())[1:]:
    # ialltime='am'
    print(f'#-------------------------------- {ialltime}')
    itime = -1
    ihour = 3
    data = cltype_hourly_frequency_alltime[ialltime][itime, ihour, :].sum(dim='types').values
    print(np.min(data))
    print(np.max(data))
    print(np.mean(data))


# check nan values there
ialltime = 'ann'
itime = -1
ihour = 3
itypes, ilats, ilons = np.where(np.isnan(cltype_hourly_frequency_alltime[ialltime][itime, ihour]))
# when you sum it up, nan disappears
print(np.isnan(cltype_hourly_frequency_alltime[ialltime][itime, ihour].sum(dim='types', skipna=False)).sum())
for itype, ilat, ilon in zip(itypes, ilats, ilons):
    print(cltype_hourly_frequency_alltime[ialltime][itime, ihour, itype, ilat, ilon].values)


# check zero values there
ialltime = 'ann'
itime = -1
ihour = 3
ilats, ilons = np.where(cltype_hourly_frequency_alltime[ialltime][itime, ihour, :].sum(dim='types') == 0)

for ilat, ilon in zip(ilats, ilons):
    print(np.max(cltype_hourly_count_alltime[ialltime][itime, ihour, :, ilat, ilon]).values)

cltype_hourly_frequency_alltime[ialltime][itime, ihour, :].sum(dim='types')[ilats[0], ilons[0]]
cltype_hourly_count_alltime[ialltime][itime, ihour, :, ilats[0], ilons[0]]


# check how many nan is there
itime = -1
ihour = 3
for ialltime in list(cltype_hourly_frequency_alltime.keys())[1:]:
    # ialltime='am'
    print(f'#-------------------------------- {ialltime}')
    print((np.isnan(cltype_hourly_frequency_alltime[ialltime][itime, ihour])).sum().values)


'''
# endregion


