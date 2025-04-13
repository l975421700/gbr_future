

# qsub -I -q express -l walltime=2:00:00,ncpus=1,mem=192GB,jobfs=20GB,storage=gdata/v46+gdata/ob53+scratch/v46+gdata/rr1+gdata/rt52+gdata/oi10+gdata/hh5+gdata/fs38


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
import joblib
import argparse

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


# region get BARRA-R2 mon data
# mem=20GB, jobfs=20GB, lowclouds, 30 min

for var in ['prw']:
    # var = 'rsut'
    print(var)
    
    fl = sorted(glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/mon/{var}/latest/*')) #[:540]
    
    with tempfile.NamedTemporaryFile(suffix='.nc') as temp_output:
        cdo.mergetime(input=fl, output=temp_output.name)
        barra_r2_mon = xr.open_dataset(temp_output.name)[var].sel(time=slice('1979', '2023')).compute()
    
    if var in ['pr', 'evspsbl', 'evspsblpot']:
        barra_r2_mon = barra_r2_mon * seconds_per_d
    elif var in ['tas', 'ts']:
        barra_r2_mon = barra_r2_mon - zerok
    elif var in ['rlus', 'rluscs', 'rlut', 'rlutcs', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'hfls', 'hfss']:
        barra_r2_mon = barra_r2_mon * (-1)
    elif var in ['psl']:
        barra_r2_mon = barra_r2_mon / 100
    elif var in ['huss']:
        barra_r2_mon = barra_r2_mon * 1000
    
    barra_r2_mon_alltime = mon_sea_ann(
        var_monthly=barra_r2_mon, lcopy=True, mm=True, sm=True, am=True,)
    
    ofile = f'data/sim/um/barra_r2/barra_r2_mon_alltime_{var}.pkl'
    if os.path.exists(ofile): os.remove(ofile)
    with open(ofile,'wb') as f:
        pickle.dump(barra_r2_mon_alltime, f)
    
    del barra_r2_mon, barra_r2_mon_alltime


'''
#-------------------------------- check
ifile = -100

barra_r2_mon_alltime = {}
for var in ['clivi', 'clwvi', 'prw']:
    # ['pr', 'clh', 'clm', 'cll', 'clt', 'evspsbl', 'hfls', 'hfss', 'psl', 'rlds', 'rldscs', 'rlus', 'rluscs', 'rlut', 'rlutcs', 'rsds', 'rsdscs', 'rsdt', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'sfcWind', 'tas', 'ts', 'evspsblpot', 'hurs', 'huss', 'uas', 'vas']
    # var = 'huss'
    print(f'#-------- {var}')
    
    with open(f'data/sim/um/barra_r2/barra_r2_mon_alltime_{var}.pkl','rb') as f:
        barra_r2_mon_alltime[var] = pickle.load(f)
    
    fl = sorted(glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/mon/{var}/latest/*'))[:540]
    
    data1 = xr.open_dataset(fl[ifile])[var]
    data2 = barra_r2_mon_alltime[var]['mon'][ifile]
    if var in ['pr', 'evspsbl', 'evspsblpot']:
        data1 = data1 * seconds_per_d
    elif var in ['tas', 'ts']:
        data1 = data1 - zerok
    elif var in ['rlus', 'rluscs', 'rlut', 'rlutcs', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'hfls', 'hfss']:
        data1 = data1 * (-1)
    elif var in ['psl']:
        data1 = data1 / 100
    elif var in ['huss']:
        data1 = data1 * 1000
    
    print((data1.squeeze().values.astype(np.float32)[np.isfinite(data1.squeeze().values.astype(np.float32))] == data2.values[np.isfinite(data2.values)]).all())
    del barra_r2_mon_alltime[var]

'''
# endregion


# region get BARRA-R2 hourly data
# Memory Used: 2.78GB; Walltime Used: 00:07:57; NCPUs Used: 1

parser=argparse.ArgumentParser()
parser.add_argument('-y', '--year', type=int, required=True,)
parser.add_argument('-m', '--month', type=int, required=True,)
args = parser.parse_args()

year=args.year
month=args.month
# year=1988; month=4
print(f'#-------------------------------- {year} {month:02d}')

for var in ['rsdt', 'rsut', 'rlut']:
    # var = 'cll'
    print(f'#---------------- {var}')
    
    odir = f'scratch/data/sim/um/barra_r2/{var}'
    os.makedirs(odir, exist_ok=True)
    
    ifile = f'/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/{var}/latest/{var}_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc'
    
    ds = xr.open_dataset(ifile)[var]
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
    
    ofile = f'{odir}/{var}_hourly_{year}{month:02d}.nc'
    if os.path.exists(ofile): os.remove(ofile)
    ds.to_netcdf(ofile)
    del ds


'''
#-------------------------------- check
var = 'rlut' # ['rsdt', 'rsut', 'rlut', 'clivi', 'clwvi', 'prw', 'clh', 'clm', 'cll', 'clt', 'pr', 'tas']
year = 2020
month = 1
ds1 = xr.open_dataset(f'scratch/data/sim/um/barra_r2/{var}/{var}_hourly_{year}{month:02d}.nc', chunks={})[var]
ds2 = xr.open_dataset(f'/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/{var}/latest/{var}_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc', chunks={})[var]
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
print((ds1.squeeze() == ds2).all().values)
del ds1, ds2


#-------------------------------- original method
# qsub -I -q megamem -l walltime=01:00:00,ncpus=48,mem=1470GB,storage=gdata/v46+gdata/ob53+scratch/v46+gdata/rr1+gdata/rt52+gdata/oi10+gdata/hh5+gdata/fs38

var = 'cll' # ['clh', 'clm', 'cll', 'clt', 'pr', 'tas']
print(f'#-------------------------------- {var}')
odir = f'scratch/data/sim/um/barra_r2/{var}'
os.makedirs(odir, exist_ok=True)

def process_year_month(year, month, var, odir):
    print(f'#---------------- {year} {month:02d}')
    
    ifile = f'/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/{var}/latest/{var}_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc'
    ofile = f'{odir}/{var}_hourly_{year}{month:02d}.nc'
    
    ds = xr.open_dataset(ifile)[var]
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
    
    del ds
    return f'Finished processing {ofile}'


joblib.Parallel(n_jobs=12)(joblib.delayed(process_year_month)(year, month, var, odir) for year in range(2023, 2024) for month in range(1, 13))

'''
# endregion


# region get BARRA-R2 alltime hourly data


for var in ['rsdt', 'rsut', 'rlut']:
    # var = 'cll'
    print(f'#-------------------------------- {var}')
    
    fl = sorted(glob.glob(f'scratch/data/sim/um/barra_r2/{var}/{var}_hourly_*.nc'))
    barra_r2_hourly = xr.open_mfdataset(fl)[var].sel(time=slice('1979', '2023'))
    barra_r2_hourly_alltime = mon_sea_ann(
        var_monthly=barra_r2_hourly, lcopy=False, mm=True, sm=True, am=True)
    
    ofile = f'data/sim/um/barra_r2/barra_r2_hourly_alltime_{var}.pkl'
    if os.path.exists(ofile): os.remove(ofile)
    with open(ofile,'wb') as f:
        pickle.dump(barra_r2_hourly_alltime, f)
    
    del barra_r2_hourly, barra_r2_hourly_alltime



'''
#-------------------------------- check
var = 'prw' # ['clivi', 'clwvi', 'prw', 'cll', 'clh', 'clm', 'clt', 'pr', 'tas']
with open(f'data/sim/um/barra_r2/barra_r2_hourly_alltime_{var}.pkl','rb') as f:
    barra_r2_hourly_alltime = pickle.load(f)

fl = sorted(glob.glob(f'scratch/data/sim/um/barra_r2/{var}/{var}_hourly_*.nc'))
ifile = 3
ds = xr.open_dataset(fl[ifile])[var]

print((barra_r2_hourly_alltime['mon'][ifile] == ds.squeeze()).all().values)
del barra_r2_hourly_alltime
'''
# endregion

