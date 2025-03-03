

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

# endregion


# region get alltime count of each cloud type

fl = sorted(glob.glob('/scratch/v46/qg8515/data/obs/jaxa/clp/??????/??/cltype_count_????????.nc'))
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
ofile='/scratch/v46/qg8515/data/obs/jaxa/clp/cltype_count_alltime.pkl'
if os.path.exists(ofile): os.remove(ofile)
with open(ofile, 'wb') as f:
    pickle.dump(cltype_count_alltime, f)

del cltype_count, cltype_count_alltime
'''
#-------------------------------- check




'''
# endregion


# region get alltime frequency of each cloud type

with open('/scratch/v46/qg8515/data/obs/jaxa/clp/cltype_count_alltime.pkl', 'rb') as f:
    cltype_count_alltime = pickle.load(f)

cltype_frequency_alltime = {}
for ialltime in ['mon', 'sea', 'ann', 'mm', 'sm', 'am']:
    # ialltime = 'mon'
    print(f'#-------------------------------- {ialltime}')
    
    cltype_frequency_alltime[ialltime] = xr.zeros_like(cltype_count_alltime[ialltime][:, 1:]).rename('cltype_frequency')
    
    for itype in cltype_frequency_alltime[ialltime].types.values:
        # itype='Stratocumulus'
        print(f'#---------------- {itype}')
        cltype_frequency_alltime[ialltime].loc[{'types': itype}][:] = (cltype_count_alltime[ialltime].loc[{'types': itype}] / cltype_count_alltime[ialltime].loc[{'types': 'finite'}] * 100).compute().astype(np.float32)

ofile='/scratch/v46/qg8515/data/obs/jaxa/clp/cltype_frequency_alltime.pkl'
if os.path.exists(ofile): os.remove(ofile)
with open(ofile, 'wb') as f:
    pickle.dump(cltype_frequency_alltime, f)


'''
#-------------------------------- check


'''
# endregion

