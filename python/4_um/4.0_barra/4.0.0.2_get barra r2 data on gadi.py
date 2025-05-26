

# qsub -I -q normal -P nf33 -l walltime=3:00:00,ncpus=1,mem=10GB,jobfs=100MB,storage=gdata/v46+gdata/ob53+scratch/v46+gdata/rr1+gdata/rt52+gdata/oi10+gdata/hh5+gdata/fs38


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


# region get BARRA-R2 mon pl data
# Memory Used: 68.42GB; Walltime Used: 01:06:43

for var in ['hus']:
    # var = 'ua'
    # ['hus', 'ta', 'ua', 'va', 'wa', 'zg']
    print(var)
    
    fl = sorted([
        file for iyear in np.arange(1979, 2024, 1)
        for file in glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/mon/{var}[0-9]*[!m]/latest/*BARRA-R2_v1_mon_{iyear}*')])
    
    def std_func(ds_in, var=var):
        ds = ds_in.expand_dims(dim='pressure', axis=1)
        varname = [varname for varname in ds.data_vars if varname.startswith(var)][0]
        ds = ds.rename({varname: var})
        ds = ds.chunk(chunks={'time': 1, 'pressure': 1, 'lat': len(ds.lat), 'lon': len(ds.lon)})
        ds = ds.astype('float32')
        if var == 'hus':
            ds = ds * 1000
        elif var == 'ta':
            ds = ds - zerok
        return(ds)
    
    barra_r2_pl_mon = xr.open_mfdataset(fl, parallel=True, preprocess=std_func)[var]
    barra_r2_pl_mon_alltime = mon_sea_ann(
        var_monthly=barra_r2_pl_mon, lcopy=False,mm=True,sm=True,am=True)
    
    ofile = f'data/sim/um/barra_r2/barra_r2_pl_mon_alltime_{var}.pkl'
    if os.path.exists(ofile): os.remove(ofile)
    with open(ofile,'wb') as f:
        pickle.dump(barra_r2_pl_mon_alltime, f)
    
    del barra_r2_pl_mon, barra_r2_pl_mon_alltime





'''
#-------------------------------- check
ipressure = 850
itime = -10

barra_r2_pl_mon_alltime = {}
for var in ['ta', 'ua', 'va', 'wa', 'zg']:
    # var = 'hus'
    print(f'#-------------------------------- {var}')
    
    with open(f'data/sim/um/barra_r2/barra_r2_pl_mon_alltime_{var}.pkl','rb') as f:
        barra_r2_pl_mon_alltime[var] = pickle.load(f)
    
    fl = sorted([
        file for iyear in np.arange(1979, 2024, 1)
        for file in glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/mon/{var}{str(ipressure)}/latest/*BARRA-R2_v1_mon_{iyear}*')])
    
    ds = xr.open_dataset(fl[itime])[f'{var}{str(ipressure)}'].squeeze()
    ds = ds.astype('float32')
    if var == 'hus':
        ds = ds * 1000
    elif var == 'ta':
        ds = ds - zerok
    
    ds2 = barra_r2_pl_mon_alltime[var]['mon'][itime].sel(pressure=ipressure)
    
    print((ds.values[np.isfinite(ds.values)].astype(np.float32) == ds2.values[np.isfinite(ds2.values)]).all())
    del barra_r2_pl_mon_alltime[var]




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


# region derive BARRA-R2 hourly pl data
# wap:24min; hur:14min

time1 = time.perf_counter()
var = 'wap' # ['hur', 'wap']
print(f'#-------------------------------- {var}')
odir = f'scratch/data/sim/um/barra_r2/{var}'
os.makedirs(odir, exist_ok=True)


def std_func(ds_in, var):
    ds = ds_in.expand_dims(dim='pressure', axis=1)
    varname = [varname for varname in ds.data_vars if varname.startswith(var)][0]
    ds = ds.rename({varname: var})
    ds = ds.chunk(chunks={'time': len(ds.time), 'pressure': 1, 'lat': len(ds.lat), 'lon': len(ds.lon)})
    ds = ds.astype('float32')
    # with tempfile.NamedTemporaryFile(suffix='.nc') as temp_output:
    #     ds.to_netcdf(temp_output.name)
    #     ds = xr.open_dataset(temp_output.name)[var]
    # if var == 'hus':
    #     ds = ds * 1000
    # elif var == 'ta':
    #     ds = ds - zerok
    return(ds)


# parser=argparse.ArgumentParser()
# parser.add_argument('-y', '--year', type=int, required=True,)
# parser.add_argument('-m', '--month', type=int, required=True,)
# args = parser.parse_args()

# year=args.year
# month=args.month
year=2007; month=8
print(f'#---------------- {year} {month:02d}')

if var=='wap':
    vars = ['hus', 'wa', 'ta']
elif var=='hur':
    vars = ['hus', 'ta']

dss = {}

for ivar in vars:
    print(f'#-------- {ivar}')
    dss[ivar] = xr.open_mfdataset(sorted(glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/{ivar}[0-9]*[!m]/latest/{ivar}[0-9]*{year}{month:02d}-{year}{month:02d}.nc')), parallel=True, preprocess=lambda ds: std_func(ds, var=ivar))[ivar]

if var=='wap':
    dss['mixr'] = mixing_ratio_from_specific_humidity(dss['hus'].sel(pressure=dss['wa'].pressure) * units('kg/kg'))
    dss['wap'] = vertical_velocity_pressure(
        dss['wa'] * units('m/s'),
        dss['wa'].pressure * units.hPa,
        dss['ta'].sel(pressure=dss['wa'].pressure) * units.K,
        dss['mixr']).metpy.dequantify().astype('float32')
elif var=='hur':
    dss['hur'] = (relative_humidity_from_specific_humidity(
        dss['hus'].pressure * units.hPa,
        dss['ta'] * units.K,
        dss['hus'] * units('kg/kg')).metpy.dequantify().astype('float32') * 100)

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

year=2022; month=6
dss = {}
for ivar in ['hus', 'wa', 'ta']:
    print(f'#-------- {ivar}')
    dss[ivar] = xr.open_mfdataset(sorted(glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/{ivar}[0-9]*[!m]/latest/{ivar}[0-9]*{year}{month:02d}-{year}{month:02d}.nc')), parallel=True, preprocess=lambda ds: std_func(ds, var=ivar))[ivar]

dss['hur'] = xr.open_dataset(f'scratch/data/sim/um/barra_r2/hur/hur_monthly_{year}{month:02d}.nc')['hur']
dss['wap'] = xr.open_dataset(f'scratch/data/sim/um/barra_r2/wap/wap_monthly_{year}{month:02d}.nc')['wap']

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


# region get BARRA-R2 alltime hourly pl data

years = 2008
yeare = 2024

for var in ['hur', 'wap']:
    print(f'#-------------------------------- {var}')
    
    fl = sorted([
        file for iyear in np.arange(years, yeare, 1)
        for file in glob.glob(f'scratch/data/sim/um/barra_r2/{var}/{var}_monthly_{iyear}??.nc')])
    
    barra_r2_pl_mon = xr.open_mfdataset(fl, parallel=True)[var]
    barra_r2_pl_mon_alltime = mon_sea_ann(
        var_monthly=barra_r2_pl_mon, lcopy=False,mm=True,sm=True,am=True)
    
    ofile = f'data/sim/um/barra_r2/barra_r2_pl_mon_alltime_{var}.pkl'
    if os.path.exists(ofile): os.remove(ofile)
    with open(ofile,'wb') as f:
        pickle.dump(barra_r2_pl_mon_alltime, f)
    
    del barra_r2_pl_mon, barra_r2_pl_mon_alltime




'''
#-------------------------------- check
years = 2008
yeare = 2024
itime = -1

for var in ['hur', 'wap']:
    # var = 'hur'
    print(f'#-------------------------------- {var}')
    
    fl = sorted([
        file for iyear in np.arange(years, yeare, 1)
        for file in glob.glob(f'scratch/data/sim/um/barra_r2/{var}/{var}_monthly_{iyear}??.nc')])
    
    with open(f'data/sim/um/barra_r2/barra_r2_pl_mon_alltime_{var}.pkl','rb') as f:
        barra_r2_pl_mon_alltime = pickle.load(f)
    
    ds = xr.open_dataset(fl[itime])[var].squeeze().values
    ds1 = barra_r2_pl_mon_alltime['mon'].isel(time=itime).values
    print((ds1[np.isfinite(ds1)] == ds[np.isfinite(ds)]).all())

'''
# endregion


# region derive BARRA-R2 mon data


barra_r2_mon_alltime = {}
for var1, var2, var3 in zip(['rlutcl', 'rsntcl', 'rsutcl'], ['rlut', 'rsnt', 'rsut'], ['rlutcs', 'rsntcs', 'rsutcs']):
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
    
    with open(f'data/sim/um/barra_r2/barra_r2_mon_alltime_{var2}.pkl','rb') as f:
        barra_r2_mon_alltime[var2] = pickle.load(f)
    with open(f'data/sim/um/barra_r2/barra_r2_mon_alltime_{var3}.pkl','rb') as f:
        barra_r2_mon_alltime[var3] = pickle.load(f)
    
    if var1 in ['rlnscl', 'rsnscl', 'rldscl', 'rsdscl', 'rluscl', 'rsuscl', 'rlutcl', 'rsntcl', 'rsutcl']:
        # print('var2 - var3')
        barra_r2_mon = (barra_r2_mon_alltime[var2]['mon'] - barra_r2_mon_alltime[var3]['mon']).rename(var1)
    elif var1 in ['rlns', 'rsns', 'rlnscs', 'rsnscs', 'rsnt', 'rsntcs']:
        # print('var2 + var3')
        barra_r2_mon = (barra_r2_mon_alltime[var2]['mon'] + barra_r2_mon_alltime[var3]['mon']).rename(var1)
    
    barra_r2_mon_alltime[var1] = mon_sea_ann(
        var_monthly=barra_r2_mon, lcopy=False, mm=True, sm=True, am=True,)
    
    ofile = f'data/sim/um/barra_r2/barra_r2_mon_alltime_{var1}.pkl'
    if os.path.exists(ofile): os.remove(ofile)
    with open(ofile,'wb') as f:
        pickle.dump(barra_r2_mon_alltime[var1], f)
    
    del barra_r2_mon_alltime[var2], barra_r2_mon_alltime[var3], barra_r2_mon_alltime[var1], barra_r2_mon
    print(str(datetime.datetime.now())[11:19])




'''

#-------------------------------- check
itime = -1
barra_r2_mon_alltime = {}
for var1, var2, var3 in zip(
    ['rluscl', 'rsuscl', 'rlutcl', 'rsutcl'], ['rlus', 'rsus', 'rlut', 'rsut'], ['rluscs', 'rsuscs', 'rlutcs', 'rsutcs']):
    # var1 = 'rluscl'; var2 = 'rlus'; var3 = 'rluscs'
    # ['rlns',  'rsns',  'rlnscs', 'rsnscs',  'rlnscl', 'rsnscl', 'rldscl', 'rsdscl',  'rluscl', 'rsuscl',  'rsnt',  'rsntcs',  'rlutcl', 'rsntcl', 'rsutcl'],
    # ['rlds',  'rsds',  'rldscs', 'rsdscs',  'rlns', 'rsns', 'rlds', 'rsds',  'rlus', 'rsus',  'rsdt',  'rsdt',  'rlut', 'rsnt', 'rsut'],
    # ['rlus',  'rsus',  'rluscs', 'rsuscs',  'rlnscs', 'rsnscs', 'rldscs', 'rsdscs',  'rluscs', 'rsuscs',  'rsut',  'rsutcs',  'rlutcs', 'rsntcs', 'rsutcs']
    print(f'Derive {var1} from {var2} and {var3}')
    
    with open(f'data/sim/um/barra_r2/barra_r2_mon_alltime_{var1}.pkl','rb') as f:
        barra_r2_mon_alltime[var1] = pickle.load(f)
    with open(f'data/sim/um/barra_r2/barra_r2_mon_alltime_{var2}.pkl','rb') as f:
        barra_r2_mon_alltime[var2] = pickle.load(f)
    with open(f'data/sim/um/barra_r2/barra_r2_mon_alltime_{var3}.pkl','rb') as f:
        barra_r2_mon_alltime[var3] = pickle.load(f)
    
    data1 = barra_r2_mon_alltime[var1]['mon'][itime]
    if var1 in ['rlnscl', 'rsnscl', 'rldscl', 'rsdscl', 'rluscl', 'rsuscl', 'rlutcl', 'rsntcl', 'rsutcl']:
        # print('var2 - var3')
        data2 = barra_r2_mon_alltime[var2]['mon'][itime] - barra_r2_mon_alltime[var3]['mon'][itime]
    elif var1 in ['rlns', 'rsns', 'rlnscs', 'rsnscs', 'rsnt', 'rsntcs']:
        # print('var2 + var3')
        data2 = barra_r2_mon_alltime[var2]['mon'][itime] + barra_r2_mon_alltime[var3]['mon'][itime]
    
    print((data1.values[np.isfinite(data1.values)] == data2.values[np.isfinite(data2.values)]).all())
    
    del barra_r2_mon_alltime[var1], barra_r2_mon_alltime[var2], barra_r2_mon_alltime[var3]






# check
barra_r2_mon_alltime = {}
with open(f'data/sim/um/barra_r2/barra_r2_mon_alltime_rsntcl.pkl','rb') as f:
    barra_r2_mon_alltime['rsntcl'] = pickle.load(f)
with open(f'data/sim/um/barra_r2/barra_r2_mon_alltime_rsutcl.pkl','rb') as f:
    barra_r2_mon_alltime['rsutcl'] = pickle.load(f)

print(np.max(np.abs(barra_r2_mon_alltime['rsntcl']['am'].values + barra_r2_mon_alltime['rsutcl']['am'].values)))

del barra_r2_mon_alltime['rsntcl'], barra_r2_mon_alltime['rsutcl']

'''
# endregion


# region derive BARRA-R2 mon data 2

for var1, vars in zip(['toa_albedocl'], [['rsutcl', 'rsdt']]):
    # var1 = 'toa_albedo'; vars = ['rsut', 'rsdt']
    # ['toa_albedo', 'toa_albedocs'], [['rsut', 'rsdt'], ['rsutcs', 'rsdt']]
    # ['rns'], [['rsus', 'rlus', 'rsds', 'rlds', 'hfls', 'hfss']]
    print(f'#-------------------------------- Derive {var1} from {vars}')
    
    barra_r2_mon_alltime = {}
    for var2 in vars:
        print(f'#---------------- {var2}')
        with open(f'data/sim/um/barra_r2/barra_r2_mon_alltime_{var2}.pkl','rb') as f:
            barra_r2_mon_alltime[var2] = pickle.load(f)
    
    if var1 in ['rns']:
        barra_r2_mon = sum(barra_r2_mon_alltime[var2]['mon'] for var2 in vars).rename(var1).compute()
        barra_r2_mon_alltime[var1] = mon_sea_ann(
            var_monthly=barra_r2_mon, lcopy=True, mm=True, sm=True, am=True)
    elif var1 in ['toa_albedo', 'toa_albedocs', 'toa_albedocl']:
        barra_r2_mon_alltime[var1] = {}
        for ialltime in barra_r2_mon_alltime[vars[0]].keys():
            print(f'#-------- {ialltime}')
            barra_r2_mon_alltime[var1][ialltime] = (barra_r2_mon_alltime[vars[0]][ialltime] / barra_r2_mon_alltime[vars[1]][ialltime] * (-1)).compute()
    
    ofile = f'data/sim/um/barra_r2/barra_r2_mon_alltime_{var1}.pkl'
    if os.path.exists(ofile): os.remove(ofile)
    with open(ofile,'wb') as f:
        pickle.dump(barra_r2_mon_alltime[var1], f)
    
    del barra_r2_mon_alltime



'''
#-------------------------------- check ['toa_albedo', 'toa_albedocs', 'toa_albedocl']
ilat = 100
ilon = 100
for var1, vars in zip(['toa_albedocl'], [['rsutcl', 'rsdt']]):
    # var1 = 'toa_albedo'; vars = ['rsut', 'rsdt']
    # ['rns'], [['rsus', 'rlus', 'rsds', 'rlds', 'hfls', 'hfss']]
    print(f'#-------------------------------- {var1} {vars}')
    
    barra_r2_mon_alltime = {}
    for var2 in [var1] + vars:
        print(f'#---------------- {var2}')
        with open(f'data/sim/um/barra_r2/barra_r2_mon_alltime_{var2}.pkl','rb') as f:
            barra_r2_mon_alltime[var2] = pickle.load(f)
    
    for ialltime in list(barra_r2_mon_alltime[var1].keys()):
        print(f'#-------- {ialltime}')
        data1 = barra_r2_mon_alltime[var1][ialltime][:, ilat, ilon].values
        data2 = barra_r2_mon_alltime[vars[0]][ialltime][:, ilat, ilon].values / barra_r2_mon_alltime[vars[1]][ialltime][:, ilat, ilon].values * (-1)
        print((data1==data2).all())




#-------------------------------- check ['rns']
barra_r2_mon_alltime = {}
for var in ['rns', 'rsus', 'rlus', 'rsds', 'rlds', 'hfls', 'hfss']:
    print(f'#-------- {var}')
    with open(f'data/sim/um/barra_r2/barra_r2_mon_alltime_{var}.pkl','rb') as f:
        barra_r2_mon_alltime[var] = pickle.load(f)

itime = -20
data1 = barra_r2_mon_alltime['rns']['mon'][itime]
data2 = sum(barra_r2_mon_alltime[var]['mon'][itime] for var in ['rsus', 'rlus', 'rsds', 'rlds', 'hfls', 'hfss']).compute().astype('float32')
print((data1 == data2).all().values)

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

