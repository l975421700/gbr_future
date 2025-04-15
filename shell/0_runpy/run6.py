

# qsub -I -q express -l walltime=4:00:00,ncpus=1,mem=192GB,jobfs=100MB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22+gdata/py18


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


# region get BARPA-R mon data

for var in ['rlut']:
    # var = 'pr'
    # ['pr', 'clh', 'clm', 'cll', 'clt', 'evspsbl', 'hfls', 'hfss', 'psl', 'rlds', 'rldscs', 'rlus', 'rluscs', 'rlut', 'rlutcs', 'rsds', 'rsdscs', 'rsdt', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'sfcWind', 'tas', 'ts', 'evspsblpot', 'hurs', 'huss', 'uas', 'vas', 'clivi', 'clwvi']
    print(var)
    
    fl = sorted(glob.glob(f'/g/data/py18/BARPA/output/CMIP6/DD/AUS-15/BOM/ERA5/evaluation/r1i1p1f1/BARPA-R/v1-r1/mon/{var}/latest/*'))
    
    barpa_r_mon = xr.open_mfdataset(fl)[var].sel(time=slice('1979', '2020'))
    if var in ['pr', 'evspsbl', 'evspsblpot']:
        barpa_r_mon = barpa_r_mon * seconds_per_d
    elif var in ['tas', 'ts']:
        barpa_r_mon = barpa_r_mon - zerok
    elif var in ['rlus', 'rluscs', 'rlut', 'rlutcs', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'hfls', 'hfss']:
        barpa_r_mon = barpa_r_mon * (-1)
    elif var in ['psl']:
        barpa_r_mon = barpa_r_mon / 100
    elif var in ['huss']:
        barpa_r_mon = barpa_r_mon * 1000
    
    barpa_r_mon_alltime = mon_sea_ann(
        var_monthly=barpa_r_mon, lcopy=False, mm=True, sm=True, am=True,)
    
    ofile = f'data/sim/um/barpa_r/barpa_r_mon_alltime_{var}.pkl'
    if os.path.exists(ofile): os.remove(ofile)
    with open(ofile,'wb') as f:
        pickle.dump(barpa_r_mon_alltime, f)
    
    del barpa_r_mon, barpa_r_mon_alltime



# endregion


