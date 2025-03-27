

# qsub -I -q express -l walltime=2:00:00,ncpus=1,mem=192GB,jobfs=20GB,storage=gdata/v46+gdata/ob53+scratch/v46+gdata/rr1+gdata/rt52+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public


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


# region get BARPA-C mon data

for var in ['cll', 'rsut']:
    # var = 'cll'
    print(var)
    
    fl = sorted(glob.glob(f'/scratch/public/chs548/BARPA-C/mon/{var}/v20241201/*'))
    
    barpa_c_mon = xr.open_mfdataset(fl)[var].sel(time=slice('2013', '2021'))
    if var in ['pr', 'evspsbl', 'evspsblpot']:
        barpa_c_mon = barpa_c_mon * seconds_per_d
    elif var in ['tas', 'ts']:
        barpa_c_mon = barpa_c_mon - zerok
    elif var in ['rlus', 'rluscs', 'rlut', 'rlutcs', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'hfls', 'hfss']:
        barpa_c_mon = barpa_c_mon * (-1)
    elif var in ['psl']:
        barpa_c_mon = barpa_c_mon / 100
    elif var in ['huss']:
        barpa_c_mon = barpa_c_mon * 1000
    
    barpa_c_mon_alltime = mon_sea_ann(
        var_monthly=barpa_c_mon, lcopy=False, mm=True, sm=True, am=True,)
    
    ofile = f'data/sim/um/barpa_c/barpa_c_mon_alltime_{var}.pkl'
    if os.path.exists(ofile): os.remove(ofile)
    with open(ofile,'wb') as f:
        pickle.dump(barpa_c_mon_alltime, f)
    
    del barpa_c_mon, barpa_c_mon_alltime




'''
#-------------------------------- check
ifile = -1
barpa_c_mon_alltime = {}
for var in ['cll', 'rsut']:
    print(f'#-------- {var}')
    
    with open(f'data/sim/um/barpa_c/barpa_c_mon_alltime_{var}.pkl','rb') as f:
        barpa_c_mon_alltime[var] = pickle.load(f)
    
    fl = sorted(glob.glob(f'/scratch/public/chs548/BARPA-C/mon/{var}/v20241201/*'))
    
    data1 = xr.open_dataset(fl[ifile])[var]
    data2 = barpa_c_mon_alltime[var]['mon'][ifile]
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
    del barpa_c_mon_alltime[var]

'''
# endregion


# region get BARPA-C hourly data
# qsub -I -q normal -l walltime=00:30:00,ncpus=48,mem=192GB,storage=gdata/v46+gdata/ob53+scratch/v46+gdata/rr1+gdata/rt52+gdata/oi10+gdata/hh5+gdata/fs38

var = 'cll'
print(f'#-------------------------------- {var}')
odir = f'scratch/data/sim/um/barpa_c/{var}'
os.makedirs(odir, exist_ok=True)

# year=2021; month=12
def process_year_month(year, month, var, odir):
    print(f'#---------------- {year} {month:02d}')
    
    ifile = f'/scratch/public/chs548/BARPA-C/1hr/{var}/v20241201/{var}_AUST-04_ERA5_evaluation_r1i1p1f1_BOM_BARPA-C_v1-r1_1hr_{year}{month:02d}-{year}{month:02d}.nc'
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

joblib.Parallel(n_jobs=12)(joblib.delayed(process_year_month)(year, month, var, odir) for year in range(2021, 2022) for month in range(1, 13))


'''
#-------------------------------- check
var = 'cll'
year = 2021
month = 1

ds1 = xr.open_dataset(f'scratch/data/sim/um/barpa_c/{var}/{var}_hourly_{year}{month:02d}.nc')[var]
ds2 = xr.open_dataset(f'/scratch/public/chs548/BARPA-C/1hr/{var}/v20241201/{var}_AUST-04_ERA5_evaluation_r1i1p1f1_BOM_BARPA-C_v1-r1_1hr_{year}{month:02d}-{year}{month:02d}.nc')[var]
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

'''
# endregion


# region get BARPA-C alltime hourly data


for var in ['cll']:
    # var = 'cll'
    print(f'#-------------------------------- {var}')
    
    fl = sorted(glob.glob(f'scratch/data/sim/um/barpa_c/{var}/{var}_hourly_*.nc'))
    barpa_c_hourly = xr.open_mfdataset(fl)[var].sel(time=slice('2013', '2021'))
    barpa_c_hourly_alltime = mon_sea_ann(
        var_monthly=barpa_c_hourly, lcopy=False, mm=True, sm=True, am=True)
    
    ofile = f'data/sim/um/barpa_c/barpa_c_hourly_alltime_{var}.pkl'
    if os.path.exists(ofile): os.remove(ofile)
    with open(ofile,'wb') as f:
        pickle.dump(barpa_c_hourly_alltime, f)
    
    del barpa_c_hourly, barpa_c_hourly_alltime



'''
#-------------------------------- check
var = 'cll'
with open(f'data/sim/um/barpa_c/barpa_c_hourly_alltime_{var}.pkl','rb') as f:
    barpa_c_hourly_alltime = pickle.load(f)

fl = sorted(glob.glob(f'scratch/data/sim/um/barpa_c/{var}/{var}_hourly_*.nc'))
ifile = -1
ds = xr.open_dataset(fl[ifile])

print((barpa_c_hourly_alltime['mon'][ifile] == ds[var].squeeze()).all().values)
'''
# endregion

