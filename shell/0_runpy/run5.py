

# region import packages

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
# from dask.diagnostics import ProgressBar
# pbar = ProgressBar()
# pbar.register()
import joblib

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


# region get era5 hourly data
# Memory Used: 165.03GB; Walltime Used: 00:10:35

var = 'mtdwswrf' # ['mtnswrf', 'mtdwswrf', 'mtnlwrf', 'tcwv', 'tclw', 'tciw', 'lcc', 'mcc', 'hcc', 'tcc', 'tp', '2t']
print(f'#-------------------------------- {var}')
odir = f'scratch/data/obs/era5/{var}'
os.makedirs(odir, exist_ok=True)

# year=2020; month=1
def process_year_month(year, month, var, odir):
    print(f'#---------------- {year} {month:02d}')
    
    ifile = glob.glob(f'/g/data/rt52/era5/single-levels/reanalysis/{var}/{year}/{var}_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}??.nc')[0]
    if var == '2t': var='t2m'
    if var == '10si': var='si10'
    if var == '2d': var='d2m'
    if var == '10u': var='u10'
    if var == '10v': var='v10'
    if var == '100u': var='u100'
    if var == '100v': var='v100'
    ofile = f'{odir}/{var}_hourly_{year}{month:02d}.nc'
    ds = xr.open_dataset(ifile, chunks={}).rename({'latitude': 'lat', 'longitude': 'lon'})[var]
    
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
    
    if var in ['e', 'pev', 'mper']:
        ds = ds * (-1)
    
    ds = ds.groupby('time.hour').mean().astype(np.float32).expand_dims(dim={'time': [ds.time[0].values]}).compute()
    
    if os.path.exists(ofile): os.remove(ofile)
    ds.to_netcdf(ofile)
    
    del ds
    return f'Finished processing {ofile}'


joblib.Parallel(n_jobs=48)(joblib.delayed(process_year_month)(year, month, var, odir) for year in range(1979, 2024) for month in range(1, 13))




'''
#-------------------------------- check

year=2023; month=12

for var in ['tcwv', 'tclw', 'tciw']:
    # var = 'lcc'
    print(f'#-------------------------------- {var}')
    odir = f'scratch/data/obs/era5/{var}'
    
    ifile = glob.glob(f'/g/data/rt52/era5/single-levels/reanalysis/{var}/{year}/{var}_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}??.nc')[0]
    if var == '2t': var='t2m'
    if var == '10si': var='si10'
    if var == '2d': var='d2m'
    if var == '10u': var='u10'
    if var == '10v': var='v10'
    if var == '100u': var='u100'
    if var == '100v': var='v100'
    ofile = f'{odir}/{var}_hourly_{year}{month:02d}.nc'
    
    ds = xr.open_dataset(ifile, chunks={}).rename({'latitude': 'lat', 'longitude': 'lon'})[var]
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
    
    if var in ['e', 'pev', 'mper']:
        ds = ds * (-1)
    
    ds = ds.groupby('time.hour').mean().astype(np.float32).expand_dims(dim={'time': [ds.time[0].values]}).compute()
    
    ds_out = xr.open_dataset(ofile)[var]
    
    print((ds.values == ds_out.values).all())

'''
# endregion

