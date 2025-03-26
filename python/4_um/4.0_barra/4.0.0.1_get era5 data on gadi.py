

# qsub -I -q express -l walltime=4:00:00,ncpus=1,mem=192GB,storage=gdata/v46+gdata/rt52+gdata/ob53


# region import packages

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
# from dask.diagnostics import ProgressBar
# pbar = ProgressBar()
# pbar.register()

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


# region get era5 sl mon data

for var in ['pev', 'mper']:
    # var = 'tp'
    # 'tp', 'e', 'cp', 'lsp', 'pev', 'msl', 'sst', '2t', '2d', 'skt', 'hcc', 'mcc', 'lcc', 'tcc', 'z', 'mper',
    print(var)
    
    fl = sorted([
        file for iyear in np.arange(1979, 2024, 1)
        for file in glob.glob(f'/g/data/rt52/era5/single-levels/monthly-averaged/{var}/{iyear}/*.nc')])
    if var == '2t': var='t2m'
    if var == '10si': var='si10'
    if var == '2d': var='d2m'
    if var == '10u': var='u10'
    if var == '10v': var='v10'
    if var == '100u': var='u100'
    if var == '100v': var='v100'
    era5_sl_mon = xr.open_mfdataset(fl, parallel=True).rename({'latitude': 'lat', 'longitude': 'lon'})[var]
    
    if var in ['tp', 'e', 'cp', 'lsp', 'pev']:
        era5_sl_mon = era5_sl_mon * 1000
    elif var in ['msl']:
        era5_sl_mon = era5_sl_mon / 100
    elif var in ['sst', 't2m', 'd2m', 'skt']:
        era5_sl_mon = era5_sl_mon - zerok
    elif var in ['hcc', 'mcc', 'lcc', 'tcc']:
        era5_sl_mon = era5_sl_mon * 100
    elif var in ['z']:
        era5_sl_mon = era5_sl_mon / 9.80665
    elif var in ['mper']:
        era5_sl_mon = era5_sl_mon * seconds_per_d
    
    if var in ['e', 'pev', 'mper']:
        era5_sl_mon = era5_sl_mon * (-1)
    
    era5_sl_mon_alltime = mon_sea_ann(
        var_monthly=era5_sl_mon, lcopy=False, mm=True, sm=True, am=True,)
    
    ofile = f'data/obs/era5/mon/era5_sl_mon_alltime_{var}.pkl'
    if os.path.exists(ofile): os.remove(ofile)
    with open(ofile, 'wb') as f:
        pickle.dump(era5_sl_mon_alltime, f)
    
    del era5_sl_mon, era5_sl_mon_alltime




#-------------------------------- check
era5_sl_mon_alltime = {}
for var in ['tciw', 'tclw', 'tcw', 'tcwv', 'tcsw', 'tcrw', 'tcslw']:
    print(f'#---------------- {var}')
    with open(f'data/obs/era5/mon/era5_sl_mon_alltime_{var}.pkl', 'rb') as f:
        era5_sl_mon_alltime[var] = pickle.load(f)

for var in ['tciw', 'tclw', 'tcw', 'tcwv', 'tcsw', 'tcrw', 'tcslw']:
    print(f'#---------------- {var}')
    print(era5_sl_mon_alltime[var]['am'].weighted(np.cos(np.deg2rad(era5_sl_mon_alltime[var]['am'].lat))).mean().values)

# tcw = tciw + tclw + tcwv + tcsw + tcrw





'''
#-------------------------------- check

itime=-1
era5_sl_mon_alltime = {}
for var in ['tp', 'msl', 'sst', 'hcc', 'mcc', 'lcc', 'tcc', '2t', 'msnlwrf', 'msnswrf', 'mtdwswrf', 'mtnlwrf', 'mtnswrf', 'msdwlwrf', 'msdwswrf', 'msdwlwrfcs', 'msdwswrfcs', 'msnlwrfcs', 'msnswrfcs', 'mtnlwrfcs', 'mtnswrfcs', 'cbh', 'tciw', 'tclw', 'e', 'z', 'mslhf', 'msshf', 'tcw', 'tcwv', 'tcsw', 'tcrw', 'tcslw', '10si', '2d', 'cp', 'lsp', 'deg0l', 'mper', 'pev', 'skt', '10u', '10v', '100u', '100v']:
    # var = 'msnlwrf'
    print(f'#---------------- {var}')
    
    fl = sorted([
        file for iyear in np.arange(1979, 2024, 1)
        for file in glob.glob(f'/g/data/rt52/era5/single-levels/monthly-averaged/{var}/{iyear}/*.nc')])
    if var == '2t': var='t2m'
    if var == '10si': var='si10'
    if var == '2d': var='d2m'
    if var == '10u': var='u10'
    if var == '10v': var='v10'
    if var == '100u': var='u100'
    if var == '100v': var='v100'
    
    ds = xr.open_dataset(fl[itime]).rename({'latitude': 'lat', 'longitude': 'lon'})[var].squeeze()
    if var in ['tp', 'e', 'cp', 'lsp', 'pev']:
        ds = ds * 1000
    elif var in ['msl']:
        ds = ds / 100
    elif var in ['sst', 't2m', 'd2m', 'skt']:
        ds = ds - zerok
    elif var in ['hcc', 'mcc', 'lcc', 'tcc']:
        ds = ds * 100
    elif var in ['z']:
        ds = ds / 9.80665
    elif var in ['mper']:
        ds = ds * seconds_per_d
    ds = ds.astype(np.float32)
    
    with open(f'data/obs/era5/mon/era5_sl_mon_alltime_{var}.pkl', 'rb') as f:
        era5_sl_mon_alltime[var] = pickle.load(f)
    
    ds2 = era5_sl_mon_alltime[var]['mon'].isel(time=itime)
    ds2 = ds2.astype(np.float32)
    print((ds.values[np.isfinite(ds.values)] == ds2.values[np.isfinite(ds2.values)]).all())
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


# region get era5 pl mon data

for var in ['q', 't', 'z']:
    # var = 'pv'
    print(var)
    
    fl = sorted([
        file for iyear in np.arange(1979, 2024, 1)
        for file in glob.glob(f'/g/data/rt52/era5/pressure-levels/monthly-averaged/{var}/{iyear}/*.nc')])
    # print(xr.open_dataset(fl[0])[var].units)
    
    era5_pl_mon = xr.open_mfdataset(fl, parallel=True).rename({'latitude': 'lat', 'longitude': 'lon'})[var]
    
    if var in ['q']:
        era5_pl_mon = era5_pl_mon * 1000
    elif var in ['t']:
        era5_pl_mon = era5_pl_mon - zerok
    elif var in ['z']:
        era5_pl_mon = era5_pl_mon / 9.80665
    
    era5_pl_mon_alltime = mon_sea_ann(
        var_monthly=era5_pl_mon, lcopy=False, mm=True, sm=True, am=True,)
    
    ofile = f'data/obs/era5/mon/era5_pl_mon_alltime_{var}.pkl'
    if os.path.exists(ofile): os.remove(ofile)
    with open(ofile, 'wb') as f:
        pickle.dump(era5_pl_mon_alltime, f)
    
    del era5_pl_mon, era5_pl_mon_alltime




'''
#-------------------------------- check
itime = -1
era5_pl_mon_alltime = {}
for var in ['pv', 'q', 'r', 't', 'u', 'v', 'w', 'z']:
    # var = 'pv'
    print(f'#-------------------------------- {var}')
    
    fl = sorted([
        file for iyear in np.arange(1979, 2024, 1)
        for file in glob.glob(f'/g/data/rt52/era5/pressure-levels/monthly-averaged/{var}/{iyear}/*.nc')])
    
    ds = xr.open_dataset(fl[itime]).rename({'latitude': 'lat', 'longitude': 'lon'})[var].squeeze()
    if var in ['q']:
        ds = ds * 1000
    elif var in ['t']:
        ds = ds - zerok
    elif var in ['z']:
        ds = ds / 9.80665
    ds = ds.astype(np.float32)
    
    with open(f'data/obs/era5/mon/era5_pl_mon_alltime_{var}.pkl', 'rb') as f:
        era5_pl_mon_alltime[var] = pickle.load(f)
    ds2 = era5_pl_mon_alltime[var]['mon'].isel(time=itime)
    ds2 = ds2.astype(np.float32)
    print((ds.values[np.isfinite(ds.values)] == ds2.values[np.isfinite(ds2.values)]).all())
    del era5_pl_mon_alltime[var]





'''
# endregion


# region derive era5 sl mon data


era5_sl_mon_alltime = {}
for var1, var2, var3 in zip(['msuwlwrf'], ['msnlwrf'], ['msdwlwrf']):
    # ['mtnlwrfcl', 'mtnswrfcl', 'mtuwswrfcl'], ['mtnlwrf', 'mtnswrf', 'mtuwswrf'], ['mtnlwrfcs', 'mtnswrfcs', 'mtuwswrfcs']
    # ['mtuwswrfcs'], ['mtnswrfcs'], ['mtdwswrf']
    # ['mtuwswrf'], ['mtnswrf'], ['mtdwswrf']
    # ['msnlwrfcl', 'msnswrfcl', 'msdwlwrfcl', 'msdwswrfcl', 'msuwlwrfcl', 'msuwswrfcl'], ['msnlwrf', 'msnswrf', 'msdwlwrf', 'msdwswrf', 'msuwlwrf', 'msuwswrf'], ['msnlwrfcs', 'msnswrfcs', 'msdwlwrfcs', 'msdwswrfcs', 'msuwlwrfcs', 'msuwswrfcs']
    # ['msuwswrfcs'], ['msnswrfcs'], ['msdwswrfcs']
    # ['msuwlwrfcs'], ['msnlwrfcs'], ['msdwlwrfcs']
    # ['msuwswrf'], ['msnswrf'], ['msdwswrf']
    # ['msuwlwrf'], ['msnlwrf'], ['msdwlwrf']
    print(f'Derive {var1} from {var2} and {var3}')
    print(str(datetime.datetime.now())[11:19])
    
    with open(f'data/obs/era5/mon/era5_sl_mon_alltime_{var2}.pkl', 'rb') as f:
        era5_sl_mon_alltime[var2] = pickle.load(f)
    with open(f'data/obs/era5/mon/era5_sl_mon_alltime_{var3}.pkl', 'rb') as f:
        era5_sl_mon_alltime[var3] = pickle.load(f)
    
    if var1 in ['msuwlwrf', 'msuwswrf', 'msuwlwrfcs', 'msuwswrfcs', 'msnlwrfcl', 'msnswrfcl', 'msdwlwrfcl', 'msdwswrfcl', 'msuwlwrfcl', 'msuwswrfcl', 'mtuwswrf', 'mtuwswrfcs', 'mtnlwrfcl', 'mtnswrfcl', 'mtuwswrfcl']:
        print('var2 - var3')
        era5_sl_mon = (era5_sl_mon_alltime[var2]['mon'] - era5_sl_mon_alltime[var3]['mon']).rename(var1)
    
    era5_sl_mon_alltime[var1] = mon_sea_ann(
        var_monthly=era5_sl_mon, lcopy=False, mm=True, sm=True, am=True)
    
    with open(f'data/obs/era5/mon/era5_sl_mon_alltime_{var1}.pkl', 'wb') as f:
        pickle.dump(era5_sl_mon_alltime[var1], f)
    
    del era5_sl_mon_alltime[var2], era5_sl_mon_alltime[var3], era5_sl_mon_alltime[var1], era5_sl_mon
    print(str(datetime.datetime.now())[11:19])




'''
#-------------------------------- check
itime=-1
era5_sl_mon_alltime = {}
for var1, var2, var3 in zip(
    ['msuwlwrf',  'msuwswrf',  'msuwlwrfcs',  'msuwswrfcs',  'msnlwrfcl', 'msnswrfcl', 'msdwlwrfcl', 'msdwswrfcl', 'msuwlwrfcl', 'msuwswrfcl',  'mtuwswrf',  'mtuwswrfcs',  'mtnlwrfcl', 'mtnswrfcl', 'mtuwswrfcl'],
    ['msnlwrf',  'msnswrf',  'msnlwrfcs',  'msnswrfcs',  'msnlwrf', 'msnswrf', 'msdwlwrf', 'msdwswrf', 'msuwlwrf', 'msuwswrf',  'mtnswrf',  'mtnswrfcs',  'mtnlwrf', 'mtnswrf', 'mtuwswrf'],
    ['msdwlwrf',  'msdwswrf',  'msdwlwrfcs',  'msdwswrfcs',  'msnlwrfcs', 'msnswrfcs', 'msdwlwrfcs', 'msdwswrfcs', 'msuwlwrfcs', 'msuwswrfcs',  'mtdwswrf',  'mtdwswrf',  'mtnlwrfcs', 'mtnswrfcs', 'mtuwswrfcs']):
    # var1='msuwlwrf'; var2='msnlwrf'; var3='msdwlwrf'
    print(f'Derive {var1} from {var2} and {var3}')
    
    with open(f'data/obs/era5/mon/era5_sl_mon_alltime_{var1}.pkl', 'rb') as f:
        era5_sl_mon_alltime[var1] = pickle.load(f)
    with open(f'data/obs/era5/mon/era5_sl_mon_alltime_{var2}.pkl', 'rb') as f:
        era5_sl_mon_alltime[var2] = pickle.load(f)
    with open(f'data/obs/era5/mon/era5_sl_mon_alltime_{var3}.pkl', 'rb') as f:
        era5_sl_mon_alltime[var3] = pickle.load(f)
    
    data1 = era5_sl_mon_alltime[var1]['mon'][itime].astype(np.float32)
    data2 = (era5_sl_mon_alltime[var2]['mon'][itime] - era5_sl_mon_alltime[var3]['mon'][itime]).astype(np.float32)
    print((data1.values[np.isfinite(data1.values)] == data2.values[np.isfinite(data2.values)]).all())
    del era5_sl_mon_alltime[var1], era5_sl_mon_alltime[var2], era5_sl_mon_alltime[var3]



# check
era5_sl_mon_alltime = {}
with open(f'data/obs/era5/mon/era5_sl_mon_alltime_mtuwswrfcl.pkl', 'rb') as f:
    era5_sl_mon_alltime['mtuwswrfcl'] = pickle.load(f)
with open(f'data/obs/era5/mon/era5_sl_mon_alltime_mtnswrfcl.pkl', 'rb') as f:
    era5_sl_mon_alltime['mtnswrfcl'] = pickle.load(f)

np.max(np.abs(era5_sl_mon_alltime['mtuwswrfcl']['am'].values - era5_sl_mon_alltime['mtnswrfcl']['am'].values))

'''
# endregion



