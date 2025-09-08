

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
import argparse
import calendar
from metpy.calc import geopotential_to_height, relative_humidity_from_dewpoint
from metpy.units import units

# management
import os
import sys  # print(sys.path)
sys.path.append(os.getcwd() + '/code/gbr_future/module')
import glob
import pickle
import datetime

from calculations import (
    mon_sea_ann,
    get_inversion, get_inversion_numba,
    get_LCL,
    get_LTS,
    get_EIS, get_EIS_simplified,
    )

from namelist import cmip6_units, zerok, seconds_per_d, cmip6_era5_var

# endregion


# region get era5 inversionh, LCL, LTS, EIS
# get_inversion_numba:  Memory Used:
# get_LCL:              Memory Used:
# get_LTS:              Memory Used:
# get_EIS:


var = 'LTS' # ['inversionh', 'LCL', 'LTS', 'EIS']
print(f'#-------------------------------- {var}')
odir = f'data/obs/era5/hourly/{var}'
os.makedirs(odir, exist_ok=True)


parser=argparse.ArgumentParser()
parser.add_argument('-y', '--year', type=int, required=True,)
parser.add_argument('-m', '--month', type=int, required=True,)
args = parser.parse_args()

year=args.year
month=args.month
# year=2024; month=12
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
        # ivar = 'zg'
        dss[ivar] = xr.open_dataset(f'/g/data/rt52/era5/pressure-levels/reanalysis/{cmip6_era5_var[ivar]}/{year}/{cmip6_era5_var[ivar]}_era5_oper_pl_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc').rename({'level': 'pressure'})[cmip6_era5_var[ivar]].sortby('pressure', ascending=False)
        if ivar == 'zg': dss[ivar] = geopotential_to_height(dss[ivar])
    elif ivar in ['orog']:
        # ivar = 'orog'
        dss[ivar] = xr.open_dataset('/g/data/rt52/era5/single-levels/reanalysis/z/2020/z_era5_oper_sfc_20200601-20200630.nc')['z'][0]
        dss[ivar] = geopotential_to_height(dss[ivar])
    elif ivar in ['tas', 'ps', 'hurs', 'ta700', 'zg700']:
        # ivar = 'hurs'
        if ivar == 'tas':
            # ivar = 'tas'
            dss[ivar] = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/2t/{year}/2t_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')[cmip6_era5_var[ivar]]
        elif ivar == 'hurs':
            # ivar = 'hurs'
            era5_t2m = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/2t/{year}/2t_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['t2m']
            era5_d2m = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/2d/{year}/2d_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['d2m']
            dss[ivar] = relative_humidity_from_dewpoint(era5_t2m * units.K, era5_d2m * units.K)
        elif ivar == 'ta700':
            # ivar = 'ta700'
            dss[ivar] = xr.open_dataset(f'/g/data/rt52/era5/pressure-levels/reanalysis/t/{year}/t_era5_oper_pl_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['t'].sel(level=700)
        elif ivar == 'zg700':
            # ivar = 'zg700'
            dss[ivar] = xr.open_dataset(f'/g/data/rt52/era5/pressure-levels/reanalysis/z/{year}/z_era5_oper_pl_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['z'].sel(level=700)
            dss[ivar] = geopotential_to_height(dss[ivar])
        else:
            # ivar = 'ps'
            dss[ivar] = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/{cmip6_era5_var[ivar]}/{year}/{cmip6_era5_var[ivar]}_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')[cmip6_era5_var[ivar]]
    elif ivar in ['LCL', 'LTS']:
        dss[ivar] = xr.open_dataset(f'data/obs/era5/hourly/{ivar}/{ivar}_hourly_{year}{month:02d}.nc')[ivar]
    
    dss[ivar] = dss[ivar].rename({'latitude': 'lat', 'longitude': 'lon'})
    
    # if ivar == 'orog':
    #     dss[ivar] = dss[ivar].isel(lat=slice(0, 10), lon=slice(0, 10))
    # else:
    #     dss[ivar] = dss[ivar].isel(time=slice(0, 10), lat=slice(0, 10), lon=slice(0, 10))


if var == 'inversionh':
    dss[var] = xr.apply_ufunc(
        # get_inversion,
        get_inversion_numba,
        dss['ta'],
        dss['zg'],
        dss['orog'],
        input_core_dims=[['pressure'], ['pressure'], []],
        vectorize=True, dask='parallelized', output_dtypes=[float],
        ).compute().rename(var)
elif var == 'LCL':
    dss[var] = xr.apply_ufunc(
        get_LCL,
        dss['ps'], dss['tas'], dss['hurs'],
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

year=2024; month=12
print(f'#-------------------------------- {year} {month:02d}')
vars = ['LTS', 'inversionh', 'LCL', 'EIS',#
        'ta', 'zg', 'orog', 'tas', 'ps', 'hurs', 'ta700', 'zg700']

dss = {}
for ivar in vars:
    print(f'#---------------- {ivar}')
    if ivar in ['ta', 'zg']:
        # ivar = 'zg'
        dss[ivar] = xr.open_dataset(f'/g/data/rt52/era5/pressure-levels/reanalysis/{cmip6_era5_var[ivar]}/{year}/{cmip6_era5_var[ivar]}_era5_oper_pl_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc').rename({'level': 'pressure'})[cmip6_era5_var[ivar]].sortby('pressure', ascending=False)
        if ivar == 'zg': dss[ivar] = geopotential_to_height(dss[ivar])
    elif ivar in ['orog']:
        # ivar = 'orog'
        dss[ivar] = xr.open_dataset('/g/data/rt52/era5/single-levels/reanalysis/z/2020/z_era5_oper_sfc_20200601-20200630.nc')['z'][0]
        dss[ivar] = geopotential_to_height(dss[ivar])
    elif ivar in ['tas', 'ps', 'hurs', 'ta700', 'zg700']:
        # ivar = 'hurs'
        if ivar == 'tas':
            dss[ivar] = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/2t/{year}/2t_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')[cmip6_era5_var[ivar]]
        elif ivar == 'hurs':
            era5_t2m = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/2t/{year}/2t_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['t2m']
            era5_d2m = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/2d/{year}/2d_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['d2m']
            dss[ivar] = relative_humidity_from_dewpoint(era5_t2m * units.K, era5_d2m * units.K)
        elif ivar == 'ta700':
            dss[ivar] = xr.open_dataset(f'/g/data/rt52/era5/pressure-levels/reanalysis/t/{year}/t_era5_oper_pl_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['t'].sel(level=700)
        elif ivar == 'zg700':
            dss[ivar] = xr.open_dataset(f'/g/data/rt52/era5/pressure-levels/reanalysis/z/{year}/z_era5_oper_pl_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['z'].sel(level=700)
            dss[ivar] = geopotential_to_height(dss[ivar])
        else:
            dss[ivar] = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/{cmip6_era5_var[ivar]}/{year}/{cmip6_era5_var[ivar]}_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')[cmip6_era5_var[ivar]]
    elif ivar in ['LCL', 'LTS']:
        dss[ivar] = xr.open_dataset(f'data/obs/era5/hourly/{ivar}/{ivar}_hourly_{year}{month:02d}.nc')[ivar]
    
    dss[ivar] = dss[ivar].rename({'latitude': 'lat', 'longitude': 'lon'})

itime = 50
ilat  = 50
ilon  = 50

for var in ['LTS', 'EIS', 'inversionh', 'LCL', ]: #
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
            dss['orog'][ilat, ilon].values,
        ))
        print(get_EIS_simplified(
            dss['LCL'][itime, ilat, ilon].values,
            dss['LTS'][itime, ilat, ilon].values,
            dss['tas'][itime, ilat, ilon].values,
            dss['ta700'][itime, ilat, ilon].values,
            dss['zg700'][itime, ilat, ilon].values,
            dss['orog'][ilat, ilon].values,
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

