

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


# region get BARPA-C hourly data
# qsub -I -q normal -P gb02 -l walltime=00:30:00,ncpus=48,mem=192GB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22+gdata/py18+gdata/gx60
# Memory Used: 161.84GB; Walltime Used: 00:16:55

var = 'clivi' # ['rlut', 'rlutcs', 'pr', 'cll', 'clm', 'clh', 'clt', 'rsut', 'rsutcs', 'clwvi', 'clivi']
print(f'#-------------------------------- {var}')
odir = f'data/sim/um/barpa_c/{var}'
os.makedirs(odir, exist_ok=True)

def process_year_month(year, month, var, odir):
    print(f'#---------------- {year} {month:02d}')
    
    ifile = f'/g/data/py18/BARPA/output/CMIP6/DD/AUST-04/BOM/ERA5/evaluation/r1i1p1f1/BARPA-C/v1-r1/1hr/{var}/latest/{var}_AUST-04_ERA5_evaluation_r1i1p1f1_BOM_BARPA-C_v1-r1_1hr_{year}{month:02d}-{year}{month:02d}.nc'
    ofile = f'{odir}/{var}_hourly_{year}{month:02d}.nc'
    
    ds = xr.open_dataset(ifile, chunks={})[var]
    if var in ['pr', 'evspsbl', 'evspsblpot']:
        ds = ds * seconds_per_d
    elif var in ['tas', 'ts']:
        ds = ds - zerok
    elif var in ['rlus', 'rluscs', 'rlut', 'rlutcs', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'hfls', 'hfss']:
        ds = ds * (-1)
    elif var in ['psl']:
        ds = ds / 100
    elif var in ['huss']:
        ds = ds * 1000
    ds = ds.groupby('time.hour').mean().astype(np.float32).expand_dims(dim={'time': [ds.time[0].values]}).compute()
    
    if os.path.exists(ofile): os.remove(ofile)
    ds.to_netcdf(ofile)
    
    return f'Finished processing {ofile}'


joblib.Parallel(n_jobs=48)(joblib.delayed(process_year_month)(year, month, var, odir) for year in range(2016, 2022) for month in range(1, 13))




'''
#-------------------------------- check
var = 'rlut'
year = 2020
month = 1

ds1 = xr.open_dataset(f'data/sim/um/barpa_c/{var}/{var}_hourly_{year}{month:02d}.nc', chunks={})
ds2 = xr.open_dataset(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARPA-C/v1/1hr/{var}/latest/{var}_AUST-04_ERA5_historical_hres_BOM_BARPA-C_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc', chunks={})[var]
if var in ['pr', 'evspsbl', 'evspsblpot']:
    ds2 = ds2 * seconds_per_d
elif var in ['tas', 'ts']:
    ds2 = ds2 - zerok
elif var in ['rlus', 'rluscs', 'rlut', 'rlutcs', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'hfls', 'hfss']:
    ds2 = ds2 * (-1)
elif var in ['psl']:
    ds2 = ds2 / 100
elif var in ['huss']:
    ds2 = ds2 * 1000
ds2 = ds2.groupby('time.hour').mean().astype(np.float32).compute()

print((ds1[var].squeeze() == ds2).all().values)




# single job
process_year_month(2020, 1, var, odir)

for var in ['cll']:
    # var = 'cll'
    # ['clh', 'clm', 'cll', 'clt', 'pr', 'tas']
    print(f'#-------------------------------- {var}')
    
    fl = sorted(glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARPA-C/v1/1hr/{var}/latest/*'))[:540]
    
    def preprocess(ds, var=var):
        ds_out = ds[var].groupby('time.hour').mean().astype(np.float32).expand_dims(dim={'time': [ds.time[0].values]})
        return(ds_out)
    
    %timeit xr.open_mfdataset(fl[:6], preprocess=preprocess, parallel=True).compute()
    
    ds = xr.open_dataset(fl[3], chunks={})
    ds[var].groupby('time.hour').mean().astype(np.float32).expand_dims(dim={'time': [ds.time[0].values]})
    preprocess(ds)


    ds = xr.open_dataset(fl[3], chunks={})
    ds[var].groupby('time.hour').mean().astype(np.float32).expand_dims(dim={'time': [ds.time[0].values]})
    
    ds = xr.open_mfdataset(fl[:12], chunks={})[var]
    ds.groupby(['time.month', 'time.hour']).mean()
    # (lat: 1018, lon: 1298, month: 12, hour: 24)
    
    
    ds = xr.open_mfdataset(fl[:24], chunks={})[var]
    ds.groupby(['time.year', 'time.month', 'time.hour']).mean()


for var in ['cll']:
    # var = 'cll'
    # ['clh', 'clm', 'cll', 'clt', 'pr', 'tas']
    print(f'#-------------------------------- {var}')
    
    odir = f'data/sim/um/barpa_c/{var}'
    os.makedirs(odir, exist_ok=True)
    
    for year in np.arange(1979, 2024, 1):
        print(f'#---------------- {year}')
        for month in np.arange(1, 13, 1):
            print(f'#-------- {month}')
            # year=2020; month=1
            ifile = sorted(glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARPA-C/v1/1hr/{var}/latest/{var}_AUST-04_ERA5_historical_hres_BOM_BARPA-C_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc'))[0]
            ofile = f'{odir}/{var}_hourly_{year}{month:02d}.nc'
            
            ds = xr.open_dataset(ifile, chunks={})
            ds = ds[var].groupby('time.hour').mean().astype(np.float32).expand_dims(dim={'time': [ds.time[0].values]}).compute()
            
            if os.path.exists(ofile): os.remove(ofile)
            ds.to_netcdf(ofile)


# dask
client = Client(processes=True)

# 0
client.compute((dask.delayed(process_year_month)(2020, month, var, odir) for month in range(3, 5)))
# 1
tasks = [process_year_month(year, month, var, odir)
         for year in range(2024, 2025)
         for month in range(1, 7)]
# 2
tasks = []
for year in np.arange(2024, 2025, 1):
    for month in np.arange(1, 7, 1):
        print(f'#---------------- {year} {month:02d}')
        task = dask.delayed(process_year_month)(year, month, var, odir)
        tasks.append(task)

# 1
for task in tasks:
    future = client.compute(task)
    print(future.result())
# 2
futures = client.compute(tasks)
client.close()


'''
# endregion

