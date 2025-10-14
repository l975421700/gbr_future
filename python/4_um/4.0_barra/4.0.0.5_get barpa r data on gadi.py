

# qsub -I -q normal -l walltime=1:00:00,ncpus=1,mem=20GB,jobfs=100MB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22+gdata/py18+gdata/gx60


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


# region get BARPA-R mon data
# Memory Used: 4.63GB; Walltime Used: 00:02:31, for all variables

years = '2016'
yeare = '2023'
for var in ['rlut', 'rsdt']:
    # var = 'pr'
    # ['pr', 'clh', 'clm', 'cll', 'clt', 'evspsbl', 'hfls', 'hfss', 'psl', 'rlds', 'rldscs', 'rlus', 'rluscs', 'rlut', 'rlutcs', 'rsds', 'rsdscs', 'rsdt', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'sfcWind', 'tas', 'ts', 'evspsblpot', 'hurs', 'huss', 'uas', 'vas', 'clivi', 'clwvi', 'zmla']
    print(var)
    
    fl = sorted(glob.glob(f'/g/data/py18/BARPA/output/CMIP6/DD/AUS-15/BOM/ERA5/evaluation/r1i1p1f1/BARPA-R/v1-r1/mon/{var}/latest/*.nc'))
    
    barpa_r_mon = xr.open_mfdataset(fl, drop_variables=["crs"])[var].sel(time=slice(years, yeare))
    if var in ['pr', 'evspsbl', 'evspsblpot']:
        barpa_r_mon = barpa_r_mon * seconds_per_d
    elif var in ['tas', 'ts']:
        barpa_r_mon = barpa_r_mon - zerok
    elif var in ['rlus', 'rluscs', 'rlut', 'rlutcs', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'hfls', 'hfss']:
        barpa_r_mon = barpa_r_mon * (-1)
    elif var in ['psl']:
        barpa_r_mon = barpa_r_mon / 100
    elif var in ['huss']:
        barpa_r_mon = barpa_r_mon * 1000
    
    barpa_r_mon_alltime = mon_sea_ann(
        var_monthly=barpa_r_mon, lcopy=False, mm=True, sm=True, am=True,)
    
    ofile = f'data/sim/um/barpa_r/barpa_r_mon_alltime_{var}.pkl'
    if os.path.exists(ofile): os.remove(ofile)
    with open(ofile,'wb') as f:
        pickle.dump(barpa_r_mon_alltime, f)
    
    del barpa_r_mon, barpa_r_mon_alltime



'''
#-------------------------------- check
ifile = -1

barpa_r_mon_alltime = {}
for var in ['cll', 'clm', 'clh', 'clt', 'rsut', 'rsutcs', 'clwvi', 'clivi', 'rlut', 'rlutcs', 'pr', 'hfls', 'hfss', 'hurs', 'huss']:
    # var = 'cll'
    # ['pr', 'clh', 'clm', 'cll', 'clt', 'evspsbl', 'hfls', 'hfss', 'psl', 'rlds', 'rldscs', 'rlus', 'rluscs', 'rlut', 'rlutcs', 'rsds', 'rsdscs', 'rsdt', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'sfcWind', 'tas', 'ts', 'evspsblpot', 'hurs', 'huss', 'uas', 'vas', 'clivi', 'clwvi']
    print(f'#-------- {var}')
    
    with open(f'data/sim/um/barpa_r/barpa_r_mon_alltime_{var}.pkl','rb') as f:
        barpa_r_mon_alltime[var] = pickle.load(f)
    
    fl = sorted(glob.glob(f'/g/data/py18/BARPA/output/CMIP6/DD/AUS-15/BOM/ERA5/evaluation/r1i1p1f1/BARPA-R/v1-r1/mon/{var}/latest/*'))
    # fl = sorted(glob.glob(f'data/sim/um/barpa_r/{var}/{var}_monthly_*.nc'))[:96]
    
    data1 = xr.open_dataset(fl[ifile])[var][-1]
    data2 = barpa_r_mon_alltime[var]['mon'][ifile]
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
    del barpa_r_mon_alltime[var]
'''
# endregion


# region get BARPA-R hourly data
# qsub -I -q normal -P v46 -l walltime=00:30:00,ncpus=48,mem=192GB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22+gdata/py18+gdata/gx60
# Memory Used: 60.32GB; Walltime Used: 00:01:41

var = 'cll' # ['cll', 'clm', 'clh', 'clt', 'rsut', 'rsutcs', 'clwvi', 'clivi', 'rlut', 'rlutcs', 'pr', 'hfls', 'hfss', 'hurs', 'huss']
print(f'#-------------------------------- {var}')
odir = f'data/sim/um/barpa_r/{var}'
os.makedirs(odir, exist_ok=True)

def process_year_month(year, month, var, odir):
    # year=2020;month=6
    print(f'#---------------- {year} {month:02d}')
    
    ifile = f'/g/data/py18/BARPA/output/CMIP6/DD/AUS-15/BOM/ERA5/evaluation/r1i1p1f1/BARPA-R/v1-r1/1hr/{var}/latest/{var}_AUS-15_ERA5_evaluation_r1i1p1f1_BOM_BARPA-R_v1-r1_1hr_{year}01-{year}12.nc'
    ofile = f'{odir}/{var}_hourly_{year}{month:02d}.nc'
    
    ds = xr.open_dataset(ifile, chunks={})[var].sel(time=slice(f'{year}-{month:02d}', f'{year}-{month:02d}'))
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


joblib.Parallel(n_jobs=48)(joblib.delayed(process_year_month)(year, month, var, odir) for year in range(2016, 2023) for month in range(1, 13))




'''
#-------------------------------- check

year = 2020
month = 6

for var in ['cll', 'clm', 'clh', 'rsut', 'rsutcs', 'clwvi', 'clivi', 'rlut', 'rlutcs', 'pr', 'hfls', 'hfss', 'hurs', 'huss']:
    # var = 'cll'
    print(var)
    
    ds1 = xr.open_dataset(f'data/sim/um/barpa_r/{var}/{var}_hourly_{year}{month:02d}.nc', chunks={})
    ds2 = xr.open_dataset(f'/g/data/py18/BARPA/output/CMIP6/DD/AUS-15/BOM/ERA5/evaluation/r1i1p1f1/BARPA-R/v1-r1/1hr/{var}/latest/{var}_AUS-15_ERA5_evaluation_r1i1p1f1_BOM_BARPA-R_v1-r1_1hr_{year}01-{year}12.nc', chunks={})[var].sel(time=slice(f'{year}-{month:02d}', f'{year}-{month:02d}'))
    
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



'''
# endregion


# region get BARPA-R alltime hourly data
# Memory Used: 8.03GB, Walltime Used: 00:01:46

years = '2016'
yeare = '2023'
for var in ['cll']:
    # var = 'cll'
    # 'cll', 'clm', 'clh', 'clt', 'rsut', 'rsutcs', 'clwvi', 'clivi', 'rlut', 'rlutcs', 'pr', 'hfls', 'hfss', 'hurs', 'huss'
    print(f'#-------------------------------- {var}')
    
    fl = sorted(glob.glob(f'data/sim/um/barpa_r/{var}/{var}_hourly_*.nc'))
    barpa_r_hourly = xr.open_mfdataset(fl)[var].sel(time=slice(years, yeare))
    barpa_r_hourly_alltime = mon_sea_ann(
        var_monthly=barpa_r_hourly, lcopy=False, mm=True, sm=True, am=True)
    
    ofile = f'data/sim/um/barpa_r/barpa_r_hourly_alltime_{var}.pkl'
    if os.path.exists(ofile): os.remove(ofile)
    with open(ofile,'wb') as f:
        pickle.dump(barpa_r_hourly_alltime, f)
    
    del barpa_r_hourly, barpa_r_hourly_alltime




'''
#-------------------------------- check
ifile = -1
for var in ['cll', 'clm', 'clh', 'clt', 'rsut', 'rsutcs', 'clwvi', 'clivi', 'rlut', 'rlutcs', 'pr', 'hfls', 'hfss', 'hurs', 'huss']:
    # var = 'rlut'
    print(f'#-------------------------------- {var}')
    
    with open(f'data/sim/um/barpa_r/barpa_r_hourly_alltime_{var}.pkl','rb') as f:
        barpa_r_hourly_alltime = pickle.load(f)
    
    fl = sorted(glob.glob(f'data/sim/um/barpa_r/{var}/{var}_hourly_*.nc'))
    ds = xr.open_dataset(fl[ifile])[var]
    print((barpa_r_hourly_alltime['mon'][ifile] == ds.squeeze()).all().values)
    
    del ds, barpa_r_hourly_alltime




'''
# endregion


# region derive BARPA-R mon data


barpa_r_mon_alltime = {}
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
    
    with open(f'data/sim/um/barpa_r/barpa_r_mon_alltime_{var2}.pkl','rb') as f:
        barpa_r_mon_alltime[var2] = pickle.load(f)
    with open(f'data/sim/um/barpa_r/barpa_r_mon_alltime_{var3}.pkl','rb') as f:
        barpa_r_mon_alltime[var3] = pickle.load(f)
    
    if var1 in ['rlnscl', 'rsnscl', 'rldscl', 'rsdscl', 'rluscl', 'rsuscl', 'rlutcl', 'rsntcl', 'rsutcl']:
        # print('var2 - var3')
        barpa_r_mon = (barpa_r_mon_alltime[var2]['mon'] - barpa_r_mon_alltime[var3]['mon']).rename(var1)
    elif var1 in ['rlns', 'rsns', 'rlnscs', 'rsnscs', 'rsnt', 'rsntcs']:
        # print('var2 + var3')
        barpa_r_mon = (barpa_r_mon_alltime[var2]['mon'] + barpa_r_mon_alltime[var3]['mon']).rename(var1)
    
    barpa_r_mon_alltime[var1] = mon_sea_ann(
        var_monthly=barpa_r_mon, lcopy=False, mm=True, sm=True, am=True,)
    
    ofile = f'data/sim/um/barpa_r/barpa_r_mon_alltime_{var1}.pkl'
    if os.path.exists(ofile): os.remove(ofile)
    with open(ofile,'wb') as f:
        pickle.dump(barpa_r_mon_alltime[var1], f)
    
    del barpa_r_mon_alltime[var2], barpa_r_mon_alltime[var3], barpa_r_mon_alltime[var1], barpa_r_mon
    print(str(datetime.datetime.now())[11:19])




'''

#-------------------------------- check
itime = -1
barpa_r_mon_alltime = {}
for var1, var2, var3 in zip(
    ['rlutcl', 'rsutcl'], ['rlut', 'rsut'], ['rlutcs', 'rsutcs']):
    # var1 = 'rluscl'; var2 = 'rlus'; var3 = 'rluscs'
    # ['rlns',  'rsns',  'rlnscs', 'rsnscs',  'rlnscl', 'rsnscl', 'rldscl', 'rsdscl',  'rluscl', 'rsuscl',  'rsnt',  'rsntcs',  'rlutcl', 'rsntcl', 'rsutcl'],
    # ['rlds',  'rsds',  'rldscs', 'rsdscs',  'rlns', 'rsns', 'rlds', 'rsds',  'rlus', 'rsus',  'rsdt',  'rsdt',  'rlut', 'rsnt', 'rsut'],
    # ['rlus',  'rsus',  'rluscs', 'rsuscs',  'rlnscs', 'rsnscs', 'rldscs', 'rsdscs',  'rluscs', 'rsuscs',  'rsut',  'rsutcs',  'rlutcs', 'rsntcs', 'rsutcs']
    print(f'Derive {var1} from {var2} and {var3}')
    
    with open(f'data/sim/um/barpa_r/barpa_r_mon_alltime_{var1}.pkl','rb') as f:
        barpa_r_mon_alltime[var1] = pickle.load(f)
    with open(f'data/sim/um/barpa_r/barpa_r_mon_alltime_{var2}.pkl','rb') as f:
        barpa_r_mon_alltime[var2] = pickle.load(f)
    with open(f'data/sim/um/barpa_r/barpa_r_mon_alltime_{var3}.pkl','rb') as f:
        barpa_r_mon_alltime[var3] = pickle.load(f)
    
    data1 = barpa_r_mon_alltime[var1]['mon'][itime]
    if var1 in ['rlnscl', 'rsnscl', 'rldscl', 'rsdscl', 'rluscl', 'rsuscl', 'rlutcl', 'rsntcl', 'rsutcl']:
        # print('var2 - var3')
        data2 = barpa_r_mon_alltime[var2]['mon'][itime] - barpa_r_mon_alltime[var3]['mon'][itime]
    elif var1 in ['rlns', 'rsns', 'rlnscs', 'rsnscs', 'rsnt', 'rsntcs']:
        # print('var2 + var3')
        data2 = barpa_r_mon_alltime[var2]['mon'][itime] + barpa_r_mon_alltime[var3]['mon'][itime]
    
    print((data1.values[np.isfinite(data1.values)] == data2.values[np.isfinite(data2.values)]).all())
    
    del barpa_r_mon_alltime[var1], barpa_r_mon_alltime[var2], barpa_r_mon_alltime[var3]






# check
barpa_r_mon_alltime = {}
with open(f'data/sim/um/barpa_r/barpa_r_mon_alltime_rsntcl.pkl','rb') as f:
    barpa_r_mon_alltime['rsntcl'] = pickle.load(f)
with open(f'data/sim/um/barpa_r/barpa_r_mon_alltime_rsutcl.pkl','rb') as f:
    barpa_r_mon_alltime['rsutcl'] = pickle.load(f)

print(np.max(np.abs(barpa_r_mon_alltime['rsntcl']['am'].values + barpa_r_mon_alltime['rsutcl']['am'].values)))

del barpa_r_mon_alltime['rsntcl'], barpa_r_mon_alltime['rsutcl']

'''
# endregion

