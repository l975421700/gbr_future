

# region import packages

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
# from dask.diagnostics import ProgressBar
# pbar = ProgressBar()
# pbar.register()
import argparse
import calendar
import time

# management
import os
import sys  # print(sys.path)
sys.path.append(os.getcwd() + '/code/gbr_future/module')

# endregion


# region get MOL and ROL cll, clm, clt
# Memory Used: 46.43GB, Walltime Used: 00:02:09

import argparse
parser=argparse.ArgumentParser()
parser.add_argument('-y', '--year', type=int, required=True,)
parser.add_argument('-m', '--month', type=int, required=True,)
args = parser.parse_args()

year=args.year; month=args.month
# year = 2016; month = 10

# option
var_vars = {
    # 'cll_mol': ['lcc', 'mcc', 'hcc'],
    'cll_rol': ['lcc', 'mcc', 'hcc'],
}

# settings
start_time = time.perf_counter()

for var in var_vars.keys():
    print(f'#-------------------------------- {var}')
    odir = f'data/sim/era5/{var}'
    os.makedirs(odir, exist_ok=True)
    
    ds = {}
    for var2 in var_vars[var]:
        print(f'#---------------- {var2}')
        ds[var2] = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/{var2}/{year}/{var2}_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')[var2]
    
    if var=='cll_mol':
        # var='cll_mol'
        ds[var] = (ds['lcc'] - xr.apply_ufunc(np.maximum, ds['mcc'], ds['hcc'])).clip(min=0)
    elif var=='cll_rol':
        # var='cll_rol'
        ds[var] = ds['lcc'] * (1 - ds['mcc']) * (1 - ds['hcc'])
    
    print('get mm')
    ds_mm = ds[var].resample({'time': '1ME'}).mean().rename(var)
    ofile1 = f'{odir}/{var}_{year}{month:02d}.nc'
    if os.path.exists(ofile1): os.remove(ofile1)
    ds_mm.to_netcdf(ofile1)
    
    print('get mhm')
    ds_mhm = ds[var].resample(time='1ME').map(lambda x: x.groupby('time.hour').mean()).rename(var)
    ofile2 = f'{odir}/{var}_hourly_{year}{month:02d}.nc'
    if os.path.exists(ofile2): os.remove(ofile2)
    ds_mhm.to_netcdf(ofile2)

end_time = time.perf_counter()
print(f"Execution time: {end_time - start_time:.1f} seconds")
# 91.4 s



'''
#-------------------------------- check
year = 2024; month = 1
# var_vars = {'cll_mol': ['lcc', 'mcc', 'hcc']}
var_vars = {'cll_rol': ['lcc', 'mcc', 'hcc']}
ilat = 200; ilon = 200

for var in var_vars.keys():
    print(f'#-------------------------------- {var}')
    odir = f'data/sim/era5/{var}'
    ds = {}
    for var2 in var_vars[var]:
        print(f'#---------------- {var2}')
        ds[var2] = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/{var2}/{year}/{var2}_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')[var2]
    
    ds_mm = xr.open_dataset(f'{odir}/{var}_{year}{month:02d}.nc')[var]
    ds_mhm = xr.open_dataset(f'{odir}/{var}_hourly_{year}{month:02d}.nc')[var]
    
    if var=='cll_mol':
        data1 = (ds['lcc'][:, ilat, ilon] - np.maximum(ds['mcc'][:, ilat, ilon], ds['hcc'][:, ilat, ilon])).clip(min=0)
    elif var=='cll_rol':
        data1 = ds['lcc'][:, ilat, ilon] - ds['lcc'][:, ilat, ilon] * ds['mcc'][:, ilat, ilon] - ds['lcc'][:, ilat, ilon] * ds['hcc'][:, ilat, ilon] + ds['lcc'][:, ilat, ilon] * ds['mcc'][:, ilat, ilon] * ds['hcc'][:, ilat, ilon]
    print(np.mean(data1).values.astype('float32') == ds_mm[0, ilat, ilon].values.astype('float32'))
    print((data1.groupby('time.hour').mean().values.astype('float32') == ds_mhm[0, ilat, ilon, :].values.astype('float32')).all())



'''
# endregion

