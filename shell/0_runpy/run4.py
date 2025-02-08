

# qsub -I -q normal -l walltime=4:00:00,ncpus=1,mem=192GB,storage=gdata/v46+gdata/rt52+gdata/ob53


# region import packages

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()

# management
import os
import sys  # print(sys.path)
sys.path.append(os.getcwd() + '/code/gbr_future/module')
import glob
import pickle

from calculations import (
    mon_sea_ann,
    )

# endregion


# region get era5 sl mon data

for var in ['100v']:
    # var = '2d'
    # 'tp', 'msl', 'sst', 'hcc', 'mcc', 'lcc', 'tcc', '2t', 'wind', 'slhf', 'msnlwrf', 'msnswrf', 'sshf', 'mtdwswrf', 'mtnlwrf', 'mtnswrf', 'msdwlwrf', 'msdwswrf', 'msdwlwrfcs', 'msdwswrfcs', 'msnlwrfcs', 'msnswrfcs', 'mtnlwrfcs', 'mtnswrfcs', 'cbh', 'tciw', 'tclw', 'e', 'z', 'mslhf', 'msshf', 'tcw', 'tcwv', 'tcsw', 'tcrw', 'tcslw',
    # '10si', '2d', 'cp', 'lsp', 'deg0l', 'mper', 'pev', 'skt'
    print(var)
    
    fl = sorted([
        file for iyear in np.arange(1979, 2024, 1)
        for file in glob.glob(f'/g/data/rt52/era5/single-levels/monthly-averaged/{var}/{iyear}/*.nc')])
    
    era5_sl_mon = xr.open_mfdataset(fl, parallel=True)
    era5_sl_mon = era5_sl_mon.rename({'latitude': 'lat', 'longitude': 'lon'})
    if var == '2t': var='t2m'
    if var == '10si': var='si10'
    if var == '2d': var='d2m'
    if var == '10u': var='u10'
    if var == '10v': var='v10'
    if var == '100u': var='u100'
    if var == '100v': var='v100'
    era5_sl_mon_alltime = mon_sea_ann(
        var_monthly=era5_sl_mon[var], lcopy=False, mm=True, sm=True, am=True,)
    
    with open(f'data/obs/era5/mon/era5_sl_mon_alltime_{var}.pkl', 'wb') as f:
        pickle.dump(era5_sl_mon_alltime, f)
    
    del era5_sl_mon, era5_sl_mon_alltime



'''
# check

era5_sl_mon_alltime = {}
for var in ['tp', 'msl', 'sst', 'hcc', 'mcc', 'lcc', 'tcc', 't2m', 'wind', 'slhf', 'msnlwrf', 'msnswrf', 'sshf', 'mtdwswrf', 'mtnlwrf', 'mtnswrf', 'msdwlwrf', 'msdwswrf', 'msdwlwrfcs', 'msdwswrfcs', 'msnlwrfcs', 'msnswrfcs', 'mtnlwrfcs', 'mtnswrfcs', 'cbh', 'tciw', 'tclw', 'e', 'z', 'mslhf', 'msshf', 'tcw', 'tcwv', 'tcsw', 'tcrw', 'tcslw', ]:
    print(var)
    with open(f'data/obs/era5/mon/era5_sl_mon_alltime_{var}.pkl', 'rb') as f:
        era5_sl_mon_alltime[var] = pickle.load(f)
    
    print(era5_sl_mon_alltime[var]['mon'].shape)
    del era5_sl_mon_alltime[var]

# https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation

Mean surface latent heat flux: slhf
Mean surface net long-wave radiation flux: msnlwrf
Mean surface net short-wave radiation flux: msnswrf
Mean surface sensible heat flux: sshf
Mean surface downward long-wave radiation flux: msdwlwrf
Mean surface downward short-wave radiation flux: msdwswrf
Mean surface downward long-wave radiation flux, clear sky: msdwlwrfcs
Mean surface downward short-wave radiation flux, clear sky: msdwswrfcs
Mean surface net long-wave radiation flux, clear sky: msnlwrfcs
Mean surface net short-wave radiation flux, clear sky: msnswrfcs

Mean top downward short-wave radiation flux: mtdwswrf
Mean top net long-wave radiation flux: mtnlwrf
Mean top net short-wave radiation flux: mtnswrf
Mean top net long-wave radiation flux, clear sky: mtnlwrfcs
Mean top net short-wave radiation flux, clear sky: mtnswrfcs

Cloud base height: cbh
Total column cloud ice water: tciw
Total column cloud liquid water: tclw
tcw
tcwv

Evaporation: e
Geopotential: z

'''
# endregion


