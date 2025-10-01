

# qsub -I -q normal -l walltime=10:00:00,ncpus=1,mem=192GB,storage=gdata/v46+gdata/rt52+gdata/ob53+gdata/zv2+gdata/gx60


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

# management
import os
import sys  # print(sys.path)
sys.path.append(os.getcwd() + '/code/gbr_future/module')
import glob
import pickle
import datetime
from calculations import (
    mon_sea_ann,
    )

from namelist import cmip6_units, zerok, seconds_per_d

# endregion


# region get agcd data

for var in ['pr']:
    # var = 'pr', 'tmax', 'tmin'
    print(f'#-------------------------------- {var}')
    
    if var=='pr':
        fl = sorted(glob.glob('/g/data/zv2/agcd/v2-0-2/precip/total/r001/01month/*'))
        agcd_data = xr.open_mfdataset(fl[-45:])['precip'].rename(var).sel(time=slice('1979', '2023')) / xr.open_mfdataset(fl[-45:])['precip'].rename(var).sel(time=slice('1979', '2023')).time.dt.days_in_month
        agcd_alltime = mon_sea_ann(
            var_monthly=agcd_data, lcopy=False, mm=True, sm=True, am=True)
    else:
        fl = sorted(glob.glob(f'/g/data/zv2/agcd/v1-0-2/{var}/mean/r005/01day/*'))
        agcd_data = xr.open_mfdataset(fl[-50:])[var].sel(time=slice('1979', '2023'))
        agcd_alltime = mon_sea_ann(
            var_daily=agcd_data, lcopy=False, mm=True, sm=True, am=True)
    
    ofile = f'data/obs/agcd/agcd_alltime_{var}.pkl'
    if os.path.exists(ofile): os.remove(ofile)
    with open(ofile,'wb') as f:
        pickle.dump(agcd_alltime, f)
    
    del agcd_data, agcd_alltime


'''
#--------------------------------
ifile = -10
itime=-4
agcd_alltime = {}
for var in ['pr', 'tmax', 'tmin']:
    # var = 'pr'
    print(f'#-------------------------------- {var}')
    
    with open(f'data/obs/agcd/agcd_alltime_{var}.pkl','rb') as f:
        agcd_alltime[var] = pickle.load(f)
    
    if var=='pr':
        fl = sorted(glob.glob('/g/data/zv2/agcd/v2-0-2/precip/total/r001/01month/*'))
        ds = (xr.open_dataset(fl[-45:][ifile])['precip'] / xr.open_dataset(fl[-45:][ifile])['precip'].time.dt.days_in_month).astype(np.float32)
        print((agcd_alltime[var]['mon'].sel(time=ds[itime].time).values == ds[itime].values).all())
    else:
        fl = sorted(glob.glob(f'/g/data/zv2/agcd/v1-0-2/{var}/mean/r005/01day/*'))
        ds = xr.open_dataset(fl[-45:][ifile])[var]
        print((agcd_alltime[var]['daily'].sel(time=ds[itime].time).values == ds[itime].values).all())




'''
# endregion


