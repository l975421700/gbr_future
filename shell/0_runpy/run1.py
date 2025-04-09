

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

with open('/scratch/v46/qg8515/data/obs/jaxa/clp/cltype_hourly_count_alltime.pkl', 'rb') as f:
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

ofile='/scratch/v46/qg8515/data/obs/jaxa/clp/cltype_hourly_frequency_alltime.pkl'
if os.path.exists(ofile): os.remove(ofile)
with open(ofile, 'wb') as f:
    pickle.dump(cltype_hourly_frequency_alltime, f)



'''
#-------------------------------- check 1
with open('/scratch/v46/qg8515/data/obs/jaxa/clp/cltype_hourly_count_alltime.pkl', 'rb') as f:
    cltype_hourly_count_alltime = pickle.load(f)
with open('/scratch/v46/qg8515/data/obs/jaxa/clp/cltype_hourly_frequency_alltime.pkl', 'rb') as f:
    cltype_hourly_frequency_alltime = pickle.load(f)

ialltime = 'mon'
itype = 'Stratocumulus'

print((cltype_hourly_frequency_alltime[ialltime].loc[{'types': itype}] == (cltype_hourly_count_alltime[ialltime].loc[{'types': itype}] / cltype_hourly_count_alltime[ialltime].loc[{'types': 'finite'}] * 100).compute().astype(np.float32)).all().values)


#-------------------------------- check 2
with open('/scratch/v46/qg8515/data/obs/jaxa/clp/cltype_frequency_alltime.pkl', 'rb') as f:
    cltype_frequency_alltime = pickle.load(f)
with open('/scratch/v46/qg8515/data/obs/jaxa/clp/cltype_hourly_frequency_alltime.pkl', 'rb') as f:
    cltype_hourly_frequency_alltime = pickle.load(f)

ialltime = 'mon'
itype = 'Stratocumulus'
print(np.max(np.abs(cltype_frequency_alltime[ialltime].loc[{'types': itype}] - cltype_hourly_frequency_alltime[ialltime].loc[{'types': itype}].mean(axis='hour'))))

'''
# endregion
