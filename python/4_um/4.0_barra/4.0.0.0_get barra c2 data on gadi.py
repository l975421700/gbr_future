

# qsub -I -q express -l walltime=4:00:00,ncpus=1,mem=20GB,jobfs=20GB,storage=gdata/v46+gdata/ob53+scratch/v46+gdata/rr1+gdata/rt52+gdata/oi10+gdata/hh5+gdata/fs38


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


# region get BARRA-C2 mon data

for var in ['prw']:
    # var = 'pr'
    print(var)
    
    fl = sorted(glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/mon/{var}/latest/*')) #[:540]
    
    barra_c2_mon = xr.open_mfdataset(fl)[var].sel(time=slice('1979', '2023'))
    if var in ['pr', 'evspsbl', 'evspsblpot']:
        barra_c2_mon = barra_c2_mon * seconds_per_d
    elif var in ['tas', 'ts']:
        barra_c2_mon = barra_c2_mon - zerok
    elif var in ['rlus', 'rluscs', 'rlut', 'rlutcs', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'hfls', 'hfss']:
        barra_c2_mon = barra_c2_mon * (-1)
    elif var in ['psl']:
        barra_c2_mon = barra_c2_mon / 100
    elif var in ['huss']:
        barra_c2_mon = barra_c2_mon * 1000
    
    barra_c2_mon_alltime = mon_sea_ann(
        var_monthly=barra_c2_mon, lcopy=False, mm=True, sm=True, am=True,)
    
    ofile = f'data/sim/um/barra_c2/barra_c2_mon_alltime_{var}.pkl'
    if os.path.exists(ofile): os.remove(ofile)
    with open(ofile,'wb') as f:
        pickle.dump(barra_c2_mon_alltime, f)
    
    del barra_c2_mon, barra_c2_mon_alltime




'''
parameter description
https://opus.nci.org.au/spaces/NDP/pages/338002591/BARRA2+Parameter+Descriptions
#-------------------------------- check
ifile = -1

barra_c2_mon_alltime = {}
for var in ['prw']:
    # var = 'clwvi'
    # ['pr', 'clh', 'clm', 'cll', 'clt', 'evspsbl', 'hfls', 'hfss', 'psl', 'rlds', 'rldscs', 'rlus', 'rluscs', 'rlut', 'rlutcs', 'rsds', 'rsdscs', 'rsdt', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'sfcWind', 'tas', 'ts', 'evspsblpot', 'hurs', 'huss', 'uas', 'vas', 'clivi', 'clwvi']
    print(f'#-------- {var}')
    
    with open(f'data/sim/um/barra_c2/barra_c2_mon_alltime_{var}.pkl','rb') as f:
        barra_c2_mon_alltime[var] = pickle.load(f)
    
    fl = sorted(glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/mon/{var}/latest/*'))[:540]
    
    data1 = xr.open_dataset(fl[ifile])[var]
    data2 = barra_c2_mon_alltime[var]['mon'][ifile]
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
    del barra_c2_mon_alltime[var]

#-------------------------------- check
barra_c2_mon_alltime = {}
for var in ['clivi', 'clwvi']:
    # var = 'evspsblpot'
    print(f'#-------- {var}')
    
    with open(f'data/sim/um/barra_c2/barra_c2_mon_alltime_{var}.pkl','rb') as f:
        barra_c2_mon_alltime[var] = pickle.load(f)


(barra_c2_mon_alltime['clh']['am'] + barra_c2_mon_alltime['clm']['am'] + barra_c2_mon_alltime['cll']['am']).to_netcdf('data/others/test/test.nc')
(barra_c2_mon_alltime['clh']['am'] + barra_c2_mon_alltime['clm']['am'] + barra_c2_mon_alltime['cll']['am'] - barra_c2_mon_alltime['clt']['am']).to_netcdf('data/others/test/test1.nc')



Precipitation: pr
High Level Cloud Fraction: clh
Mid Level Cloud Fraction: clm
Low Level Cloud Fraction: cll
Total Cloud Cover Percentage: clt
Evaporation Including Sublimation and Transpiration: evspsbl
Surface Upward Latent Heat Flux: hfls
Surface Upward Sensible Heat Flux: hfss

sea level pressure: psl
Surface downwelling LW radiation: rlds
Surface Downwelling Clear-Sky Longwave Radiation: rldscs
Surface Upwelling Longwave Radiation: rlus
Surface Upwelling Clear-Sky Longwave Radiation: rluscs
TOA Outgoing Longwave Radiation: rlut
TOA Outgoing Clear-Sky Longwave Radiation: rlutcs

Surface downwelling SW radiation: rsds
Surface Downwelling Clear-Sky Shortwave Radiation: rsdscs
TOA Incident Shortwave Radiation: rsdt
Surface Upwelling Shortwave Radiation: rsus
Surface Upwelling Clear-Sky Shortwave Radiation: rsuscs
TOA Outgoing Shortwave Radiation: rsut
TOA Outgoing Clear-Sky Shortwave Radiation: rsutcs
near surface wind speed: sfcWind
Near surface air temperature: tas
Surface temperature: ts



???
Ice Water Path: clivi
Condensed Water Path: clwvi

'''
# endregion


# region get BARRA-C2 mon pl data

for var in ['hus', 'ta']:
    # var = 'zg'
    # ['hus', 'ta', 'ua', 'va', 'wa', 'wap', 'zg']
    print(var)
    
    fl = sorted([
        file for iyear in np.arange(1979, 2024, 1)
        for file in glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/mon/{var}[0-9]*[!m]/latest/*BARRA-C2_v1_mon_{iyear}*')])
    
    def std_func(ds, var=var):
        ds = ds.expand_dims(dim='pressure', axis=1)
        varname = [varname for varname in ds.data_vars if varname.startswith(var)][0]
        ds = ds.rename({varname: var})
        return(ds)
    
    barra_c2_pl_mon = xr.open_mfdataset(fl, parallel=True, preprocess=std_func)[var]
    if var == 'hus':
        barra_c2_pl_mon = barra_c2_pl_mon * 1000
    elif var == 'ta':
        barra_c2_pl_mon = barra_c2_pl_mon - zerok
    
    barra_c2_pl_mon_alltime = mon_sea_ann(
        var_monthly=barra_c2_pl_mon, lcopy=False,mm=True,sm=True,am=True)
    
    ofile = f'data/sim/um/barra_c2/barra_c2_pl_mon_alltime_{var}.pkl'
    if os.path.exists(ofile): os.remove(ofile)
    with open(ofile,'wb') as f:
        pickle.dump(barra_c2_pl_mon_alltime, f)
    
    del barra_c2_pl_mon, barra_c2_pl_mon_alltime





'''
#-------------------------------- check
ipressure = 500
itime = -1

barra_c2_pl_mon_alltime = {}
for var in ['hus', 'ta', 'ua', 'va', 'wa', 'wap', 'zg']:
    # var = 'hus'
    print(f'#-------------------------------- {var}')
    
    with open(f'data/sim/um/barra_c2/barra_c2_pl_mon_alltime_{var}.pkl','rb') as f:
        barra_c2_pl_mon_alltime[var] = pickle.load(f)
    
    fl = sorted([
        file for iyear in np.arange(1979, 2024, 1)
        for file in glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/mon/{var}{str(ipressure)}/latest/*BARRA-C2_v1_mon_{iyear}*')])
    
    ds = xr.open_dataset(fl[itime])[f'{var}{str(ipressure)}'].squeeze()
    if var == 'hus':
        ds = ds * 1000
    elif var == 'ta':
        ds = ds - zerok
    
    ds2 = barra_c2_pl_mon_alltime[var]['mon'][itime].sel(pressure=ipressure)
    
    print((ds.values[np.isfinite(ds.values)].astype(np.float32) == ds2.values[np.isfinite(ds2.values)]).all())
    del barra_c2_pl_mon_alltime[var]




aaa = xr.open_dataset(fl[0])
aaa.expand_dims(dim='pressure', axis=1)
aaa['ta10'].expand_dims(dim='pressure', axis=1)

bbb = xr.open_dataset(fl[-1])
bbb.expand_dims(dim='pressure', axis=1)


import intake
data_catalog = intake.open_esm_datastore("/g/data/dk92/catalog/v2/esm/barra2-ob53/catalog.json")

for icol in data_catalog.df.columns:
    print(f'#-------------------------------- {icol}')
    print(data_catalog.df[icol].unique())


'''
# endregion


# region derive BARRA-C2 mon data


barra_c2_mon_alltime = {}
for var1, var2, var3 in zip(['rluscl', 'rsuscl', 'rlutcl', 'rsutcl'], ['rlus', 'rsus', 'rlut', 'rsut'], ['rluscs', 'rsuscs', 'rlutcs', 'rsutcs']):
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
    
    with open(f'data/sim/um/barra_c2/barra_c2_mon_alltime_{var2}.pkl','rb') as f:
        barra_c2_mon_alltime[var2] = pickle.load(f)
    with open(f'data/sim/um/barra_c2/barra_c2_mon_alltime_{var3}.pkl','rb') as f:
        barra_c2_mon_alltime[var3] = pickle.load(f)
    
    if var1 in ['rlnscl', 'rsnscl', 'rldscl', 'rsdscl', 'rluscl', 'rsuscl', 'rlutcl', 'rsntcl', 'rsutcl']:
        # print('var2 - var3')
        barra_c2_mon = (barra_c2_mon_alltime[var2]['mon'] - barra_c2_mon_alltime[var3]['mon']).rename(var1)
    elif var1 in ['rlns', 'rsns', 'rlnscs', 'rsnscs', 'rsnt', 'rsntcs']:
        # print('var2 + var3')
        barra_c2_mon = (barra_c2_mon_alltime[var2]['mon'] + barra_c2_mon_alltime[var3]['mon']).rename(var1)
    
    barra_c2_mon_alltime[var1] = mon_sea_ann(
        var_monthly=barra_c2_mon, lcopy=False, mm=True, sm=True, am=True,)
    
    ofile = f'data/sim/um/barra_c2/barra_c2_mon_alltime_{var1}.pkl'
    if os.path.exists(ofile): os.remove(ofile)
    with open(ofile,'wb') as f:
        pickle.dump(barra_c2_mon_alltime[var1], f)
    
    del barra_c2_mon_alltime[var2], barra_c2_mon_alltime[var3], barra_c2_mon_alltime[var1], barra_c2_mon
    print(str(datetime.datetime.now())[11:19])




'''

#-------------------------------- check
itime = -1
barra_c2_mon_alltime = {}
for var1, var2, var3 in zip(
    ['rluscl', 'rsuscl', 'rlutcl', 'rsutcl'], ['rlus', 'rsus', 'rlut', 'rsut'], ['rluscs', 'rsuscs', 'rlutcs', 'rsutcs']):
    # var1 = 'rluscl'; var2 = 'rlus'; var3 = 'rluscs'
    # ['rlns',  'rsns',  'rlnscs', 'rsnscs',  'rlnscl', 'rsnscl', 'rldscl', 'rsdscl',  'rluscl', 'rsuscl',  'rsnt',  'rsntcs',  'rlutcl', 'rsntcl', 'rsutcl'],
    # ['rlds',  'rsds',  'rldscs', 'rsdscs',  'rlns', 'rsns', 'rlds', 'rsds',  'rlus', 'rsus',  'rsdt',  'rsdt',  'rlut', 'rsnt', 'rsut'],
    # ['rlus',  'rsus',  'rluscs', 'rsuscs',  'rlnscs', 'rsnscs', 'rldscs', 'rsdscs',  'rluscs', 'rsuscs',  'rsut',  'rsutcs',  'rlutcs', 'rsntcs', 'rsutcs']
    print(f'Derive {var1} from {var2} and {var3}')
    
    with open(f'data/sim/um/barra_c2/barra_c2_mon_alltime_{var1}.pkl','rb') as f:
        barra_c2_mon_alltime[var1] = pickle.load(f)
    with open(f'data/sim/um/barra_c2/barra_c2_mon_alltime_{var2}.pkl','rb') as f:
        barra_c2_mon_alltime[var2] = pickle.load(f)
    with open(f'data/sim/um/barra_c2/barra_c2_mon_alltime_{var3}.pkl','rb') as f:
        barra_c2_mon_alltime[var3] = pickle.load(f)
    
    data1 = barra_c2_mon_alltime[var1]['mon'][itime]
    if var1 in ['rlnscl', 'rsnscl', 'rldscl', 'rsdscl', 'rluscl', 'rsuscl', 'rlutcl', 'rsntcl', 'rsutcl']:
        # print('var2 - var3')
        data2 = barra_c2_mon_alltime[var2]['mon'][itime] - barra_c2_mon_alltime[var3]['mon'][itime]
    elif var1 in ['rlns', 'rsns', 'rlnscs', 'rsnscs', 'rsnt', 'rsntcs']:
        # print('var2 + var3')
        data2 = barra_c2_mon_alltime[var2]['mon'][itime] + barra_c2_mon_alltime[var3]['mon'][itime]
    
    print((data1.values[np.isfinite(data1.values)] == data2.values[np.isfinite(data2.values)]).all())
    
    del barra_c2_mon_alltime[var1], barra_c2_mon_alltime[var2], barra_c2_mon_alltime[var3]






# check
barra_c2_mon_alltime = {}
with open(f'data/sim/um/barra_c2/barra_c2_mon_alltime_rsntcl.pkl','rb') as f:
    barra_c2_mon_alltime['rsntcl'] = pickle.load(f)
with open(f'data/sim/um/barra_c2/barra_c2_mon_alltime_rsutcl.pkl','rb') as f:
    barra_c2_mon_alltime['rsutcl'] = pickle.load(f)

print(np.max(np.abs(barra_c2_mon_alltime['rsntcl']['am'].values + barra_c2_mon_alltime['rsutcl']['am'].values)))

del barra_c2_mon_alltime['rsntcl'], barra_c2_mon_alltime['rsutcl']

'''
# endregion


# region get BARRA-C2 hourly data
# qsub -I -q normal -l walltime=00:30:00,ncpus=48,mem=192GB,storage=gdata/v46+gdata/ob53+scratch/v46+gdata/rr1+gdata/rt52+gdata/oi10+gdata/hh5+gdata/fs38


var = 'cll' # ['clh', 'clm', 'cll', 'clt', 'pr', 'tas']
print(f'#-------------------------------- {var}')
odir = f'scratch/data/sim/um/barra_c2/{var}'
os.makedirs(odir, exist_ok=True)

def process_year_month(year, month, var, odir):
    print(f'#---------------- {year} {month:02d}')
    
    ifile = f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/{var}/latest/{var}_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc'
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


joblib.Parallel(n_jobs=48)(joblib.delayed(process_year_month)(year, month, var, odir) for year in range(1979, 2024) for month in range(1, 13))




'''
#-------------------------------- check
var = 'pr'
year = 2020
month = 1

ds1 = xr.open_dataset(f'scratch/data/sim/um/barra_c2/{var}/{var}_hourly_{year}{month:02d}.nc', chunks={})
ds2 = xr.open_dataset(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/{var}/latest/{var}_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc', chunks={})[var]
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
    
    fl = sorted(glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/{var}/latest/*'))[:540]
    
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
    
    odir = f'scratch/data/sim/um/barra_c2/{var}'
    os.makedirs(odir, exist_ok=True)
    
    for year in np.arange(1979, 2024, 1):
        print(f'#---------------- {year}')
        for month in np.arange(1, 13, 1):
            print(f'#-------- {month}')
            # year=2020; month=1
            ifile = sorted(glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/{var}/latest/{var}_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc'))[0]
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


# region get BARRA-C2 alltime hourly data


for var in ['cll', 'clh', 'clm', 'clt', 'pr', 'tas']:
    # var = 'cll'
    print(f'#-------------------------------- {var}')
    
    fl = sorted(glob.glob(f'scratch/data/sim/um/barra_c2/{var}/{var}_hourly_*.nc'))
    barra_c2_hourly = xr.open_mfdataset(fl)[var].sel(time=slice('1979', '2023'))
    barra_c2_hourly_alltime = mon_sea_ann(
        var_monthly=barra_c2_hourly, lcopy=False, mm=True, sm=True, am=True)
    
    ofile = f'data/sim/um/barra_c2/barra_c2_hourly_alltime_{var}.pkl'
    if os.path.exists(ofile): os.remove(ofile)
    with open(ofile,'wb') as f:
        pickle.dump(barra_c2_hourly_alltime, f)
    
    del barra_c2_hourly, barra_c2_hourly_alltime



'''
#-------------------------------- check
var = 'tas'
with open(f'data/sim/um/barra_c2/barra_c2_hourly_alltime_{var}.pkl','rb') as f:
    barra_c2_hourly_alltime = pickle.load(f)

fl = sorted(glob.glob(f'scratch/data/sim/um/barra_c2/{var}/{var}_hourly_*.nc'))
ifile = -1
ds = xr.open_dataset(fl[ifile])

print((barra_c2_hourly_alltime['mon'][ifile] == ds[var].squeeze()).all().values)
'''
# endregion


