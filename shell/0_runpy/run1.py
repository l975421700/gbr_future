

# qsub -I -q express -P nf33 -l walltime=3:00:00,ncpus=1,mem=192GB,jobfs=100MB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22


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
    )

from namelist import zerok, seconds_per_d

# endregion


# region get BARRA-C2 alltime hourly pl data

years = 1979
yeare = 2023

for var in ['hur']:
    print(f'#-------------------------------- {var}')
    
    fl = sorted([
        file for iyear in np.arange(years, yeare+1, 1)
        for file in glob.glob(f'scratch/data/sim/um/barra_c2/{var}/{var}_monthly_{iyear}??.nc')])
    
    barra_c2_pl_mon = xr.open_mfdataset(fl, parallel=True)[var]
    barra_c2_pl_mon_alltime = mon_sea_ann(
        var_monthly=barra_c2_pl_mon, lcopy=False,mm=True,sm=True,am=True)
    
    ofile = f'data/sim/um/barra_c2/barra_c2_pl_mon_alltime_{var}.pkl'
    if os.path.exists(ofile): os.remove(ofile)
    with open(ofile,'wb') as f:
        pickle.dump(barra_c2_pl_mon_alltime, f)
    
    del barra_c2_pl_mon, barra_c2_pl_mon_alltime




'''
#-------------------------------- check
years = 1979
yeare = 2023
itime = -1

for var in ['hur', 'wap']:
    # var = 'hur'
    print(f'#-------------------------------- {var}')
    
    fl = sorted([
        file for iyear in np.arange(years, yeare+1, 1)
        for file in glob.glob(f'scratch/data/sim/um/barra_r2/{var}/{var}_monthly_{iyear}??.nc')])
    
    with open(f'data/sim/um/barra_r2/barra_r2_pl_mon_alltime_{var}.pkl','rb') as f:
        barra_r2_pl_mon_alltime = pickle.load(f)
    
    ds = xr.open_dataset(fl[itime])[var].squeeze().values
    ds1 = barra_r2_pl_mon_alltime['mon'].isel(time=itime).values
    print((ds1[np.isfinite(ds1)] == ds[np.isfinite(ds)]).all())

'''
# endregion

