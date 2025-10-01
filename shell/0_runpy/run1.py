

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


# region get alltime count of each cloud type
# Memory Used: 526.97GB, Walltime Used: 02:33:21

fl = sorted(glob.glob('data/obs/jaxa/clp/??????/??/cltype_count_????????.nc'))
cltype_count = xr.open_mfdataset(fl).cltype_count

cltype_count_alltime = {}

cltype_count_alltime['daily'] = cltype_count

print('get mon')
cltype_count_alltime['mon'] = cltype_count.resample({'time': '1ME'}).sum().compute()

print('get sea')
cltype_count_alltime['sea'] = cltype_count_alltime['mon'].resample({'time': 'QE-FEB'}).sum()[1:-1].compute()

print('get ann')
cltype_count_alltime['ann'] = cltype_count_alltime['mon'].resample({'time': '1YE'}).sum()[1:].compute()

print('get mm')
cltype_count_alltime['mm'] = cltype_count_alltime['mon'].groupby('time.month').sum().compute()
cltype_count_alltime['mm'] = cltype_count_alltime['mm'].rename({'month': 'time'})

print('get sm')
cltype_count_alltime['sm'] = cltype_count_alltime['sea'].groupby('time.season').sum().compute()
cltype_count_alltime['sm'] = cltype_count_alltime['sm'].rename({'season': 'time'})

print('get am')
cltype_count_alltime['am'] = cltype_count_alltime['ann'].sum(dim='time').compute()
cltype_count_alltime['am'] = cltype_count_alltime['am'].expand_dims('time', axis=0)

print('output data')
ofile='data/obs/jaxa/clp/cltype_count_alltime.pkl'
if os.path.exists(ofile): os.remove(ofile)
with open(ofile, 'wb') as f:
    pickle.dump(cltype_count_alltime, f)


'''
#-------------------------------- check
with open('data/obs/jaxa/clp/cltype_count_alltime.pkl', 'rb') as f:
    cltype_count_alltime = pickle.load(f)

ifile = -1
itype = 'Stratocumulus'
ilat = 100
ilon = 100

fl = sorted(glob.glob('data/obs/jaxa/clp/??????/??/cltype_count_????????.nc'))
ds = xr.open_dataset(fl[ifile])

print((ds.cltype_count.loc[{'types': itype}][0].values == cltype_count_alltime['daily'][ifile].loc[{'types': itype}].values).all())

print((cltype_count_alltime['mon'].loc[{'types': itype}][0, ilat, ilon] == cltype_count_alltime['daily'].loc[{'types': itype}][:28, ilat, ilon].resample({'time': '1ME'}).sum()).all().values)

print((cltype_count_alltime['sea'].loc[{'types': itype}][:, ilat, ilon] == cltype_count_alltime['mon'].loc[{'types': itype}][:, ilat, ilon].resample({'time': 'QE-FEB'}).sum()[1:-1]).all().values)

print((cltype_count_alltime['ann'].loc[{'types': itype}][:, ilat, ilon] == cltype_count_alltime['mon'].loc[{'types': itype}][:, ilat, ilon].resample({'time': '1YE'}).sum()[1:].compute()).all().values)

'''
# endregion
