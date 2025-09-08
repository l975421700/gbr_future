

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
    get_inversion, get_inversion_numba,
    get_LCL,
    get_LTS,
    get_EIS, get_EIS_simplified,
    )

from namelist import zerok, seconds_per_d

# endregion


# region get BARRA-C2 inversionh, LCL, LTS, EIS
# get_inversion_numba:  Memory Used: 60.06GB, Walltime Used: 02:07:38
# get_LCL:              Memory Used: 188.22GB,Walltime Used: 03:42:41
# get_LTS:              Memory Used: 68.83GB, Walltime Used: 00:04:39
# get_EIS:


var = 'EIS' # ['inversionh', 'LCL', 'LTS', 'EIS']
print(f'#-------------------------------- {var}')
odir = f'data/sim/um/barra_c2/{var}'
os.makedirs(odir, exist_ok=True)


def std_func(ds_in, ivar):
    ds = ds_in.expand_dims(dim='pressure', axis=1)
    varname = [varname for varname in ds.data_vars if varname.startswith(ivar)][0]
    ds = ds.rename({varname: ivar})
    # ds = ds.chunk(chunks={'time': len(ds.time), 'pressure': 1, 'lat': len(ds.lat), 'lon': len(ds.lon)})
    ds = ds.astype('float32')
    return(ds)


parser=argparse.ArgumentParser()
parser.add_argument('-y', '--year', type=int, required=True,)
parser.add_argument('-m', '--month', type=int, required=True,)
args = parser.parse_args()

year=args.year
month=args.month
# year=2020; month=6
print(f'#---------------- {year} {month:02d}')


if var == 'inversionh':
    vars = ['ta', 'zg', 'orog']
elif var == 'LCL':
    vars = ['tas', 'ps', 'hurs']
elif var == 'LTS':
    vars = ['tas', 'ps', 'ta700']
elif var == 'EIS':
    vars = ['LCL', 'LTS', 'tas', 'ta700', 'zg700', 'orog']

dss = {}
for ivar in vars:
    print(f'#-------- {ivar}')
    if ivar in ['ta', 'zg']:
        dss[ivar] = xr.open_mfdataset(sorted(glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/{ivar}[0-9]*[!m]/latest/*{year}{month:02d}.nc')), parallel=True, preprocess=lambda ds: std_func(ds, ivar=ivar))[ivar].chunk({'pressure': -1})
    elif ivar in ['orog']:
        dss[ivar] = xr.open_dataset('/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/fx/orog/latest/orog_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1.nc')['orog']
    elif ivar in ['tas', 'ps', 'hurs', 'ta700', 'zg700']:
        dss[ivar] = xr.open_dataset(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/{ivar}/latest/{ivar}_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc')[ivar]
    elif ivar in ['LCL', 'LTS']:
        dss[ivar] = xr.open_dataset(f'data/sim/um/barra_c2/{ivar}/{ivar}_hourly_{year}{month:02d}.nc')[ivar]
    
    # if ivar == 'orog':
    #     dss[ivar] = dss[ivar].isel(lat=slice(0, 10), lon=slice(0, 10))
    # else:
    #     dss[ivar] = dss[ivar].isel(time=slice(0, 10), lat=slice(0, 10), lon=slice(0, 10))


if var == 'inversionh':
    dss[var] = xr.apply_ufunc(
        # get_inversion,
        get_inversion_numba,
        dss['ta'].sortby('pressure', ascending=False),
        dss['zg'].sortby('pressure', ascending=False),
        dss['orog'],
        input_core_dims=[['pressure'], ['pressure'], []],
        vectorize=True, dask='parallelized', output_dtypes=[float],
        ).compute().rename(var)
elif var == 'LCL':
    dss[var] = xr.apply_ufunc(
        get_LCL,
        dss['ps'], dss['tas'], dss['hurs'] / 100,
        vectorize=True, dask='parallelized').compute().rename(var)
elif var == 'LTS':
    dss[var] = get_LTS(dss['tas'], dss['ps'], dss['ta700']).compute().rename(var)
elif var == 'EIS':
    dss[var] = xr.apply_ufunc(
        # get_EIS,
        get_EIS_simplified,
        dss['LCL'], dss['LTS'],
        dss['tas'], dss['ta700'], dss['zg700'], dss['orog'],
        vectorize=True, dask='parallelized').compute().rename(var)


ofile = f'{odir}/{var}_hourly_{year}{month:02d}.nc'
if os.path.exists(ofile): os.remove(ofile)
dss[var].to_netcdf(ofile)




'''
#-------------------------------- check
def std_func(ds_in, ivar):
    ds = ds_in.expand_dims(dim='pressure', axis=1)
    varname = [varname for varname in ds.data_vars if varname.startswith(ivar)][0]
    return(ds.rename({varname: ivar}).astype('float32'))

year=2024; month=1
print(f'#-------------------------------- {year} {month:02d}')
vars = ['LTS', 'inversionh', 'LCL', 'EIS',#
        'ta', 'zg', 'orog', 'tas', 'ps', 'hurs', 'ta700', 'zg700']

dss = {}
for ivar in vars:
    print(f'#---------------- {ivar}')
    if ivar in ['ta', 'zg']:
        dss[ivar] = xr.open_mfdataset(sorted(glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/{ivar}[0-9]*[!m]/latest/*{year}{month:02d}.nc')), parallel=True, preprocess=lambda ds: std_func(ds, ivar=ivar))[ivar].chunk({'pressure': -1})
    elif ivar in ['orog']:
        dss[ivar] = xr.open_dataset('/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/fx/orog/latest/orog_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1.nc')['orog']
    elif ivar in ['tas', 'ps', 'hurs', 'ta700', 'zg700']:
        dss[ivar] = xr.open_dataset(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/{ivar}/latest/{ivar}_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc')[ivar]
    elif ivar in ['inversionh', 'LCL', 'LTS', 'EIS']:
        dss[ivar] = xr.open_dataset(f'data/sim/um/barra_c2/{ivar}/{ivar}_hourly_{year}{month:02d}.nc')[ivar]

itime = 100
ilat  = 100
ilon  = 100

for var in ['LTS', ]: #, 'EIS', 'inversionh', 'LCL'
    print(f'#---------------- {var}')
    print(dss[var][itime, ilat, ilon].values)
    if var == 'inversionh':
        # var = 'inversionh'
        print(get_inversion(
            dss['ta'][itime, ::-1, ilat, ilon].values,
            dss['zg'][itime, ::-1, ilat, ilon].values,
            dss['orog'][ilat, ilon].values
            ))
        print(get_inversion_numba(
            dss['ta'][itime, ::-1, ilat, ilon].values,
            dss['zg'][itime, ::-1, ilat, ilon].values,
            dss['orog'][ilat, ilon].values
            ))
    elif var == 'LCL':
        # var = 'LCL'
        print(get_LCL(
            dss['ps'][itime, ilat, ilon].values,
            dss['tas'][itime, ilat, ilon].values,
            dss['hurs'][itime, ilat, ilon].values / 100,
            ))
    elif var == 'LTS':
        # var = 'LTS'
        print(get_LTS(
            dss['tas'][itime, ilat, ilon].values,
            dss['ps'][itime, ilat, ilon].values,
            dss['ta700'][itime, ilat, ilon].values,
        ))
    elif var == 'EIS':
        # var = 'EIS'
        print(get_EIS(
            dss['tas'][itime, ilat, ilon].values,
            dss['ps'][itime, ilat, ilon].values,
            dss['ta700'][itime, ilat, ilon].values,
            dss['hurs'][itime, ilat, ilon].values / 100,
            dss['zg700'][itime, ilat, ilon].values,
        ))

# check two get_inversionh methods
ds1 = xr.open_dataset('data/sim/um/barra_c2/inversionh/inversionh_hourly_202401.nc')
ds2 = xr.open_dataset('data/sim/um/barra_c2/inversionh/inversionh_hourly_2024012.nc')
np.max(np.abs(ds1['inversionh'].values - ds2['inversionh'].values))

ds = xr.open_dataset('data/sim/um/barra_c2/LTS/LTS_hourly_202312.nc')['LTS']

for ivar in vars:
    print(f'#---------------- {ivar}')
    print(dss[ivar])
'''
# endregion
