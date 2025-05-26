

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


# region derive BARRA-C2 hourly pl data
# wap:24min; hur:14min

time1 = time.perf_counter()
var = 'wap' # ['hur', 'wap']
print(f'#-------------------------------- {var}')
odir = f'scratch/data/sim/um/barra_c2/{var}'
os.makedirs(odir, exist_ok=True)


def std_func(ds_in, var):
    ds = ds_in.expand_dims(dim='pressure', axis=1)
    varname = [varname for varname in ds.data_vars if varname.startswith(var)][0]
    ds = ds.rename({varname: var})
    ds = ds.chunk(chunks={'time': len(ds.time), 'pressure': 1, 'lat': len(ds.lat), 'lon': len(ds.lon)})
    ds = ds.astype('float32')
    # if var == 'hus':
    #     ds = ds * 1000
    # elif var == 'ta':
    #     ds = ds - zerok
    return(ds)


parser=argparse.ArgumentParser()
parser.add_argument('-y', '--year', type=int, required=True,)
parser.add_argument('-m', '--month', type=int, required=True,)
args = parser.parse_args()

year=args.year
month=args.month
# year=2020; month=6
print(f'#---------------- {year} {month:02d}')

if var=='wap':
    vars = ['hus', 'wa', 'ta']
elif var=='hur':
    vars = ['hus', 'ta']

dss = {}

for ivar in vars:
    print(f'#-------- {ivar}')
    dss[ivar] = xr.open_mfdataset(sorted(glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/{ivar}[0-9]*[!m]/latest/{ivar}[0-9]*{year}{month:02d}-{year}{month:02d}.nc')), parallel=True, preprocess=lambda ds: std_func(ds, var=ivar))[ivar]

if var=='wap':
    dss['mixr'] = mixing_ratio_from_specific_humidity(dss['hus'].sel(pressure=dss['wa'].pressure) * units('kg/kg')).compute()
    dss['wap'] = vertical_velocity_pressure(
        dss['wa'] * units('m/s'),
        dss['wa'].pressure * units.hPa,
        dss['ta'].sel(pressure=dss['wa'].pressure) * units.K,
        dss['mixr']).metpy.dequantify().astype('float32').compute()
elif var=='hur':
    dss['hur'] = (relative_humidity_from_specific_humidity(
        dss['hus'].pressure * units.hPa,
        dss['ta'] * units.K,
        dss['hus'] * units('kg/kg')).metpy.dequantify().astype('float32') * 100).compute()

# ofile1 = f'{odir}/{var}_hourly_{year}{month:02d}.nc'
ofile2 = f'{odir}/{var}_monthly_{year}{month:02d}.nc'
# if os.path.exists(ofile1): os.remove(ofile1)
if os.path.exists(ofile2): os.remove(ofile2)

# dss[var].to_netcdf(ofile1)
dss[var].resample({'time': '1MS'}).mean().compute().rename(var).to_netcdf(ofile2)
time2 = time.perf_counter()
print(f'Execution time: {time2 - time1:.1f} s')



'''
#-------------------------------- check
def std_func(ds_in, var):
    ds = ds_in.expand_dims(dim='pressure', axis=1)
    varname = [varname for varname in ds.data_vars if varname.startswith(var)][0]
    ds = ds.rename({varname: var})
    ds = ds.chunk(chunks={'time': len(ds.time), 'pressure': 1, 'lat': len(ds.lat), 'lon': len(ds.lon)})
    ds = ds.astype('float32')
    return(ds)

year=2012; month=1
dss = {}
for ivar in ['hus', 'wa', 'ta']:
    print(f'#-------- {ivar}')
    dss[ivar] = xr.open_mfdataset(sorted(glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/{ivar}[0-9]*[!m]/latest/{ivar}[0-9]*{year}{month:02d}-{year}{month:02d}.nc')), parallel=True, preprocess=lambda ds: std_func(ds, var=ivar))[ivar]

dss['hur'] = xr.open_dataset(f'scratch/data/sim/um/barra_c2/hur/hur_monthly_{year}{month:02d}.nc')['hur']
dss['wap'] = xr.open_dataset(f'scratch/data/sim/um/barra_c2/wap/wap_monthly_{year}{month:02d}.nc')['wap']

ilat = 200
ilon = 200
iplev = 500

aaa = (relative_humidity_from_specific_humidity(
        iplev * units.hPa,
        dss['ta'].sel(pressure=iplev).isel(lon=ilat, lat=ilat) * units.K,
        dss['hus'].sel(pressure=iplev).isel(lon=ilat, lat=ilat) * units('kg/kg')).metpy.dequantify().astype('float32') * 100).compute()
print((aaa.mean().values - dss['hur'].sel(pressure=iplev).isel(lon=ilat, lat=ilat).squeeze().values) / dss['hur'].sel(pressure=iplev).isel(lon=ilat, lat=ilat).squeeze().values)

bbb = vertical_velocity_pressure(
    dss['wa'].sel(pressure=iplev).isel(lon=ilat, lat=ilat) * units('m/s'),
    iplev * units.hPa,
    dss['ta'].sel(pressure=iplev).isel(lon=ilat, lat=ilat) * units.K,
    mixing_ratio_from_specific_humidity(
        dss['hus'].sel(pressure=iplev).isel(lon=ilat, lat=ilat) * units('kg/kg')
        ).compute()
    ).metpy.dequantify().astype('float32').compute()

print((bbb.mean().values - dss['wap'].sel(pressure=iplev).isel(lon=ilat, lat=ilat).squeeze().values) / dss['wap'].sel(pressure=iplev).isel(lon=ilat, lat=ilat).squeeze().values)

'''
# endregion


