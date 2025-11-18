

# qsub -I -q normal -P v46 -l walltime=3:00:00,ncpus=1,mem=96GB,jobfs=100MB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22+gdata/py18+gdata/gx60


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


# region get BARPA-C mon data

years = '2016'
yeare = '2023'
for var in ['cll_mol']:
    # var = 'cll'
    # 'cll', 'clm', 'clh', 'clt', 'rsut', 'rsutcs', 'clwvi', 'clivi', 'rlut', 'rlutcs', 'pr', 'rsdt', 'hfls', 'hfss'
    print(var)
    
    # fl = sorted(glob.glob(f'/g/data/py18/BARPA/output/CMIP6/DD/AUST-04/BOM/ERA5/evaluation/r1i1p1f1/BARPA-C/v1-r1/mon/{var}/latest/*.nc'))
    # fl = sorted(glob.glob(f'data/sim/um/barpa_c/{var}/{var}_monthly_*.nc'))
    fl = sorted(glob.glob(f'data/sim/um/barpa_c/{var}/{var}_??????.nc'))
    
    barpa_c_mon = xr.open_mfdataset(fl)[var].sel(time=slice(years, yeare))
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
for var in ['rlut', 'rlutcs', 'pr', 'cll', 'clm', 'clh', 'clt', 'rsut', 'rsutcs', 'clwvi', 'clivi']:
    # var = 'clwvi'
    # ['pr', 'clh', 'clm', 'cll', 'clt', 'evspsbl', 'hfls', 'hfss', 'psl', 'rlds', 'rldscs', 'rlus', 'rluscs', 'rlut', 'rlutcs', 'rsds', 'rsdscs', 'rsdt', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'sfcWind', 'tas', 'ts', 'evspsblpot', 'hurs', 'huss', 'uas', 'vas', 'clivi', 'clwvi']
    print(f'#-------- {var}')
    
    with open(f'data/sim/um/barpa_c/barpa_c_mon_alltime_{var}.pkl','rb') as f:
        barpa_c_mon_alltime[var] = pickle.load(f)
    
    fl = sorted(glob.glob(f'/g/data/py18/BARPA/output/CMIP6/DD/AUST-04/BOM/ERA5/evaluation/r1i1p1f1/BARPA-C/v1-r1/mon/{var}/latest/*'))
    # fl = sorted(glob.glob(f'data/sim/um/barpa_c/{var}/{var}_monthly_*.nc'))[:96]
    
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
# qsub -I -q normal -P gb02 -l walltime=00:30:00,ncpus=48,mem=192GB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22+gdata/py18+gdata/gx60
# Memory Used: 161.84GB; Walltime Used: 00:16:55

var = 'cll' # ['rlut', 'rlutcs', 'pr', 'cll', 'clm', 'clh', 'clt', 'rsut', 'rsutcs', 'clwvi', 'clivi']
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

year = 2020
month = 1

for var in ['rlut', 'rlutcs', 'pr', 'cll', 'clm', 'clh', 'clt', 'rsut', 'rsutcs', 'clwvi', 'clivi']:
    # var = 'cll'
    print(var)
    
    ds1 = xr.open_dataset(f'data/sim/um/barpa_c/{var}/{var}_hourly_{year}{month:02d}.nc', chunks={})
    ds2 = xr.open_dataset(f'/g/data/py18/BARPA/output/CMIP6/DD/AUST-04/BOM/ERA5/evaluation/r1i1p1f1/BARPA-C/v1-r1/1hr/{var}/latest/{var}_AUST-04_ERA5_evaluation_r1i1p1f1_BOM_BARPA-C_v1-r1_1hr_{year}{month:02d}-{year}{month:02d}.nc', chunks={})[var]
    
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


'''
# endregion


# region get BARPA-C alltime hourly data
# Memory Used: 47.5GB, Walltime Used: 00:42:54

years = '2016'
yeare = '2023'
for var in ['rlut', 'rlutcs', 'pr', 'cll', 'clm', 'clh', 'clt', 'rsut', 'rsutcs', 'clwvi', 'clivi']:
    # var = 'cll'
    # ['clivi', 'clwvi', 'prw', 'cll', 'clh', 'clm', 'clt', 'pr', 'tas']
    print(f'#-------------------------------- {var}')
    
    fl = sorted(glob.glob(f'data/sim/um/barpa_c/{var}/{var}_hourly_*.nc'))
    barpa_c_hourly = xr.open_mfdataset(fl)[var].sel(time=slice(years, yeare))
    barpa_c_hourly_alltime = mon_sea_ann(
        var_monthly=barpa_c_hourly, lcopy=False, mm=True, sm=True, am=True)
    
    ofile = f'data/sim/um/barpa_c/barpa_c_hourly_alltime_{var}.pkl'
    if os.path.exists(ofile): os.remove(ofile)
    with open(ofile,'wb') as f:
        pickle.dump(barpa_c_hourly_alltime, f)
    
    del barpa_c_hourly, barpa_c_hourly_alltime




'''
#-------------------------------- check
ifile = -1
for var in ['rlut', 'rlutcs', 'pr', 'cll', 'clm', 'clh', 'clt', 'rsut', 'rsutcs', 'clwvi', 'clivi']:
    print(f'#-------------------------------- {var}')
    
    with open(f'data/sim/um/barpa_c/barpa_c_hourly_alltime_{var}.pkl','rb') as f:
        barpa_c_hourly_alltime = pickle.load(f)
    
    fl = sorted(glob.glob(f'data/sim/um/barpa_c/{var}/{var}_hourly_*.nc'))
    ds = xr.open_dataset(fl[ifile])[var]
    print((barpa_c_hourly_alltime['mon'][ifile] == ds.squeeze()).all().values)
    
    del ds, barpa_c_hourly_alltime




'''
# endregion


# region derive BARPA-C mon data


barpa_c_mon_alltime = {}
for var1, var2, var3 in zip(['rlutcl', 'rsutcl'], ['rlut', 'rsut'], ['rlutcs', 'rsutcs']):
    # ['rlutcl', 'rsntcl', 'rsutcl'], ['rlut', 'rsnt', 'rsut'], ['rlutcs', 'rsntcs', 'rsutcs']
    # ['rsntcs'], ['rsdt'], ['rsutcs']
    # ['rsnt'], ['rsdt'], ['rsut']
    # ['rluscl', 'rsuscl'], ['rlus', 'rsus'], ['rluscs', 'rsuscs']
    # ['rlnscl', 'rsnscl', 'rldscl', 'rsdscl'], ['rlns', 'rsns', 'rlds', 'rsds'], ['rlnscs', 'rsnscs', 'rldscs', 'rsdscs']
    # ['rlnscs', 'rsnscs'], ['rldscs', 'rsdscs'], ['rluscs', 'rsuscs']
    # ['rsns'], ['rsds'], ['rsus']
    # ['rlns'], ['rlds'], ['rlus']
    print(f'Derive {var1} from {var2} and {var3}')
    print(str(datetime.datetime.now())[11:19])
    
    with open(f'data/sim/um/barpa_c/barpa_c_mon_alltime_{var2}.pkl','rb') as f:
        barpa_c_mon_alltime[var2] = pickle.load(f)
    with open(f'data/sim/um/barpa_c/barpa_c_mon_alltime_{var3}.pkl','rb') as f:
        barpa_c_mon_alltime[var3] = pickle.load(f)
    
    if var1 in ['rlnscl', 'rsnscl', 'rldscl', 'rsdscl', 'rluscl', 'rsuscl', 'rlutcl', 'rsntcl', 'rsutcl']:
        # print('var2 - var3')
        barpa_c_mon = (barpa_c_mon_alltime[var2]['mon'] - barpa_c_mon_alltime[var3]['mon']).rename(var1)
    elif var1 in ['rlns', 'rsns', 'rlnscs', 'rsnscs', 'rsnt', 'rsntcs']:
        # print('var2 + var3')
        barpa_c_mon = (barpa_c_mon_alltime[var2]['mon'] + barpa_c_mon_alltime[var3]['mon']).rename(var1)
    
    barpa_c_mon_alltime[var1] = mon_sea_ann(
        var_monthly=barpa_c_mon, lcopy=False, mm=True, sm=True, am=True,)
    
    ofile = f'data/sim/um/barpa_c/barpa_c_mon_alltime_{var1}.pkl'
    if os.path.exists(ofile): os.remove(ofile)
    with open(ofile,'wb') as f:
        pickle.dump(barpa_c_mon_alltime[var1], f)
    
    del barpa_c_mon_alltime[var2], barpa_c_mon_alltime[var3], barpa_c_mon_alltime[var1], barpa_c_mon
    print(str(datetime.datetime.now())[11:19])




'''

#-------------------------------- check
itime = -1
barpa_c_mon_alltime = {}
for var1, var2, var3 in zip(
    ['rlutcl', 'rsutcl'], ['rlut', 'rsut'], ['rlutcs', 'rsutcs']):
    # var1 = 'rluscl'; var2 = 'rlus'; var3 = 'rluscs'
    # ['rlns',  'rsns',  'rlnscs', 'rsnscs',  'rlnscl', 'rsnscl', 'rldscl', 'rsdscl',  'rluscl', 'rsuscl',  'rsnt',  'rsntcs',  'rlutcl', 'rsntcl', 'rsutcl'],
    # ['rlds',  'rsds',  'rldscs', 'rsdscs',  'rlns', 'rsns', 'rlds', 'rsds',  'rlus', 'rsus',  'rsdt',  'rsdt',  'rlut', 'rsnt', 'rsut'],
    # ['rlus',  'rsus',  'rluscs', 'rsuscs',  'rlnscs', 'rsnscs', 'rldscs', 'rsdscs',  'rluscs', 'rsuscs',  'rsut',  'rsutcs',  'rlutcs', 'rsntcs', 'rsutcs']
    print(f'Derive {var1} from {var2} and {var3}')
    
    with open(f'data/sim/um/barpa_c/barpa_c_mon_alltime_{var1}.pkl','rb') as f:
        barpa_c_mon_alltime[var1] = pickle.load(f)
    with open(f'data/sim/um/barpa_c/barpa_c_mon_alltime_{var2}.pkl','rb') as f:
        barpa_c_mon_alltime[var2] = pickle.load(f)
    with open(f'data/sim/um/barpa_c/barpa_c_mon_alltime_{var3}.pkl','rb') as f:
        barpa_c_mon_alltime[var3] = pickle.load(f)
    
    data1 = barpa_c_mon_alltime[var1]['mon'][itime]
    if var1 in ['rlnscl', 'rsnscl', 'rldscl', 'rsdscl', 'rluscl', 'rsuscl', 'rlutcl', 'rsntcl', 'rsutcl']:
        # print('var2 - var3')
        data2 = barpa_c_mon_alltime[var2]['mon'][itime] - barpa_c_mon_alltime[var3]['mon'][itime]
    elif var1 in ['rlns', 'rsns', 'rlnscs', 'rsnscs', 'rsnt', 'rsntcs']:
        # print('var2 + var3')
        data2 = barpa_c_mon_alltime[var2]['mon'][itime] + barpa_c_mon_alltime[var3]['mon'][itime]
    
    print((data1.values[np.isfinite(data1.values)] == data2.values[np.isfinite(data2.values)]).all())
    
    del barpa_c_mon_alltime[var1], barpa_c_mon_alltime[var2], barpa_c_mon_alltime[var3]






# check
barpa_c_mon_alltime = {}
with open(f'data/sim/um/barpa_c/barpa_c_mon_alltime_rsntcl.pkl','rb') as f:
    barpa_c_mon_alltime['rsntcl'] = pickle.load(f)
with open(f'data/sim/um/barpa_c/barpa_c_mon_alltime_rsutcl.pkl','rb') as f:
    barpa_c_mon_alltime['rsutcl'] = pickle.load(f)

print(np.max(np.abs(barpa_c_mon_alltime['rsntcl']['am'].values + barpa_c_mon_alltime['rsutcl']['am'].values)))

del barpa_c_mon_alltime['rsntcl'], barpa_c_mon_alltime['rsutcl']

'''
# endregion




# region get MOL and ROL cll, clm, clt
# Memory Used: 46.43GB, Walltime Used: 00:02:09

import argparse
parser=argparse.ArgumentParser()
parser.add_argument('-y', '--year', type=int, required=True,)
parser.add_argument('-m', '--month', type=int, required=True,)
args = parser.parse_args()

year=args.year; month=args.month
# year = 2013; month = 1

# option
var_vars = {
    'cll_mol': ['cll', 'clm', 'clh'],
}

# settings
min_lon, max_lon, min_lat, max_lat = [110.58, 157.34, -43.69, -7.01]
start_time = time.perf_counter()

for var in var_vars.keys():
    print(f'#-------------------------------- {var}')
    odir = f'data/sim/um/barpa_c/{var}'
    os.makedirs(odir, exist_ok=True)
    
    ds = {}
    for var2 in var_vars[var]:
        print(f'#---------------- {var2}')
        ds[var2] = xr.open_dataset(f'/g/data/py18/BARPA/output/CMIP6/DD/AUST-04/BOM/ERA5/evaluation/r1i1p1f1/BARPA-C/v1-r1/1hr/{var2}/latest/{var2}_AUST-04_ERA5_evaluation_r1i1p1f1_BOM_BARPA-C_v1-r1_1hr_{year}{month:02d}-{year}{month:02d}.nc')[var2].sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))
    
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
year = 2020; month = 6
var_vars = {'cll_mol': ['cll', 'clm', 'clh']}
min_lon, max_lon, min_lat, max_lat = [110.58, 157.34, -43.69, -7.01]
ilat = 200; ilon = 200

for var in var_vars.keys():
    print(f'#-------------------------------- {var}')
    odir = f'data/sim/um/barpa_c/{var}'
    ds = {}
    for var2 in var_vars[var]:
        print(f'#---------------- {var2}')
        ds[var2] = xr.open_dataset(f'/g/data/py18/BARPA/output/CMIP6/DD/AUST-04/BOM/ERA5/evaluation/r1i1p1f1/BARPA-C/v1-r1/1hr/{var2}/latest/{var2}_AUST-04_ERA5_evaluation_r1i1p1f1_BOM_BARPA-C_v1-r1_1hr_{year}{month:02d}-{year}{month:02d}.nc')[var2].sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))
    
    ds_mm = xr.open_dataset(f'{odir}/{var}_{year}{month:02d}.nc')[var]
    ds_mhm = xr.open_dataset(f'{odir}/{var}_hourly_{year}{month:02d}.nc')[var]
    
    if var=='cll_mol':
        data1 = (ds['cll'][:, ilat, ilon] - np.maximum(ds['clm'][:, ilat, ilon], ds['clh'][:, ilat, ilon])).clip(min=0)
    print(np.mean(data1).values == ds_mm[0, ilat, ilon].values)
    print((data1.groupby('time.hour').mean().values == ds_mhm[0, ilat, ilon, :].values).all())



'''
# endregion




# Not finished because hourly data not available
# region get hourly BARPA-C LCL, LTS, EIS
# get_LCL:              Memory Used: 188.22GB,Walltime Used: 03:42:41
# get_LTS:              Memory Used: 68.83GB, Walltime Used: 00:04:39
# get_EIS:              Memory Used: 267.02GB, Walltime Used: 02:47:24


var = 'LTS' # ['LCL', 'LTS', 'EIS']
print(f'#-------------------------------- {var}')
odir = f'data/sim/um/barpa_c/{var}'
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
        dss[ivar] = xr.open_mfdataset(sorted(glob.glob(f'/g/data/py18/BARPA/output/CMIP6/DD/AUST-04/BOM/ERA5/evaluation/r1i1p1f1/BARPA-C/v1-r1/3hr/{ivar}[0-9]*[!m]/latest/*{year}{month:02d}.nc')), parallel=True, preprocess=lambda ds: std_func(ds, ivar=ivar))[ivar].chunk({'pressure': -1})
    elif ivar in ['orog']:
        dss[ivar] = xr.open_dataset('/g/data/py18/BARPA/output/CMIP6/DD/AUST-04/BOM/ERA5/evaluation/r1i1p1f1/BARPA-C/v1-r1/fx/orog/latest/orog_AUST-04_ERA5_evaluation_r1i1p1f1_BOM_BARPA-C_v1-r1_fx.nc')['orog']
    elif ivar in ['tas', 'ps', 'hurs', 'ta700', 'zg700']:
        dss[ivar] = xr.open_dataset(f'/g/data/py18/BARPA/output/CMIP6/DD/AUST-04/BOM/ERA5/evaluation/r1i1p1f1/BARPA-C/v1-r1/3hr/{ivar}/latest/{ivar}_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc')[ivar]
    elif ivar in ['LCL', 'LTS']:
        dss[ivar] = xr.open_dataset(f'data/sim/um/barpa_c/{ivar}/{ivar}_hourly_{year}{month:02d}.nc')[ivar]
    
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

year=2018; month=1
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
        dss[ivar] = xr.open_dataset(f'data/sim/um/barpa_c/{ivar}/{ivar}_hourly_{year}{month:02d}.nc')[ivar]

itime = 30
ilat  = 5
ilon  = 5

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
ds1 = xr.open_dataset('data/sim/um/barpa_c/inversionh/inversionh_hourly_202401.nc')
ds2 = xr.open_dataset('data/sim/um/barpa_c/inversionh/inversionh_hourly_2024012.nc')
np.max(np.abs(ds1['inversionh'].values - ds2['inversionh'].values))

ds = xr.open_dataset('data/sim/um/barpa_c/LTS/LTS_hourly_202312.nc')['LTS']

for ivar in vars:
    print(f'#---------------- {ivar}')
    print(dss[ivar])
'''
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

