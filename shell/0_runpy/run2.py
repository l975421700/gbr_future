

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


# region get alltime frequency of each cloud type
# Memory Used: 323.52GB, Walltime Used: 00:11:15

with open('data/obs/jaxa/clp/cltype_count_alltime.pkl', 'rb') as f:
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

ofile='data/obs/jaxa/clp/cltype_frequency_alltime.pkl'
if os.path.exists(ofile): os.remove(ofile)
with open(ofile, 'wb') as f:
    pickle.dump(cltype_frequency_alltime, f)


'''
#-------------------------------- check
with open('data/obs/jaxa/clp/cltype_count_alltime.pkl', 'rb') as f:
    cltype_count_alltime = pickle.load(f)
with open('data/obs/jaxa/clp/cltype_frequency_alltime.pkl', 'rb') as f:
    cltype_frequency_alltime = pickle.load(f)

ialltime = 'mon'
itype = 'Stratocumulus'
ilat = 100
ilon = 100

data1 = cltype_frequency_alltime[ialltime].loc[{'types': itype}][:, ilat, ilon]
data2 = (cltype_count_alltime[ialltime].loc[{'types': itype}][:, ilat, ilon] / cltype_count_alltime[ialltime].loc[{'types': 'finite'}][:, ilat, ilon] * 100).compute().astype(np.float32)
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all().values)

'''
# endregion

