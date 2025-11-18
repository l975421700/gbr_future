

# region import packages

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
import argparse
import time

# management
import os
import sys  # print(sys.path)
sys.path.append(os.getcwd() + '/code/gbr_future/module')

# endregion


# region get MOL and ROL cll, clm, clt

import argparse
parser=argparse.ArgumentParser()
parser.add_argument('-y', '--year', type=int, required=True,)
parser.add_argument('-m', '--month', type=int, required=True,)
args = parser.parse_args()

year=args.year; month=args.month
# year = 2024; month = 1

# option
var_vars = {
    'cll_mol': ['cll', 'clm', 'clh'],
}

# settings
min_lon, max_lon, min_lat, max_lat = [110.58, 157.34, -43.69, -7.01]
start_time = time.perf_counter()

for var in var_vars.keys():
    print(f'#-------------------------------- {var}')
    odir = f'data/sim/um/barra_c2/{var}'
    os.makedirs(odir, exist_ok=True)
    
    ds = {}
    for var2 in var_vars[var]:
        print(f'#---------------- {var2}')
        ds[var2] = xr.open_dataset(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/{var2}/latest/{var2}_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc')[var2].sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))
    
    if var=='cll_mol':
        # var='cll_mol'
        ds[var] = (ds['cll'] - xr.apply_ufunc(np.maximum, ds['clm'], ds['clh'])).clip(min=0)
    
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
var_vars = {'cll_mol': ['cll', 'clm', 'clh']}
min_lon, max_lon, min_lat, max_lat = [110.58, 157.34, -43.69, -7.01]
ilat = 100; ilon = 100

for var in var_vars.keys():
    print(f'#-------------------------------- {var}')
    odir = f'data/sim/um/barra_c2/{var}'
    ds = {}
    for var2 in var_vars[var]:
        print(f'#---------------- {var2}')
        ds[var2] = xr.open_dataset(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/{var2}/latest/{var2}_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc')[var2].sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))
    
    ds_mm = xr.open_dataset(f'{odir}/{var}_{year}{month:02d}.nc')['__xarray_dataarray_variable__']
    ds_mhm = xr.open_dataset(f'{odir}/{var}_hourly_{year}{month:02d}.nc')['__xarray_dataarray_variable__']
    
    if var=='cll_mol':
        data1 = (ds['cll'][:, ilat, ilon] - np.maximum(ds['clm'][:, ilat, ilon], ds['clh'][:, ilat, ilon])).clip(min=0)
    print(np.mean(data1).values == ds_mm[0, ilat, ilon].values)
    print((data1.groupby('time.hour').mean().values == ds_mhm[0, ilat, ilon, :].values).all())



'''
# endregion

