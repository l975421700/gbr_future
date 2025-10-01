

# qsub -I -q normal -l walltime=3:00:00,ncpus=1,mem=192GB,storage=gdata/v46+gdata/ob53+scratch/v46+gdata/rr1+gdata/rt52+gdata/oi10+gdata/hh5+gdata/fs38


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


# region get era5 sl mon data
# Memory Used: 14.87GB, Walltime Used: 00:15:42

for var in ['tp', 'hcc', 'mcc', 'lcc', 'tcc', 'tciw', 'tclw', 'mtnswrf', 'mtdwswrf']:
    # var = 'tp'
    # 'tp', 'e', 'cp', 'lsp', 'pev', 'msl', 'sst', '2t', '2d', 'skt', 'hcc', 'mcc', 'lcc', 'tcc', 'z', 'mper',
    print(var)
    
    fl = sorted([
        file for iyear in np.arange(1979, 2024, 1)
        for file in glob.glob(f'/g/data/rt52/era5/single-levels/monthly-averaged/{var}/{iyear}/*.nc')])
    if var == '2t': var='t2m'
    if var == '10si': var='si10'
    if var == '2d': var='d2m'
    if var == '10u': var='u10'
    if var == '10v': var='v10'
    if var == '100u': var='u100'
    if var == '100v': var='v100'
    era5_sl_mon = xr.open_mfdataset(fl, parallel=True).rename({'latitude': 'lat', 'longitude': 'lon'})[var]
    
    # fl = sorted(glob.glob(f'data/sim/era5/hourly/{var}/{var}_monthly_*.nc'))
    # era5_sl_mon = xr.open_mfdataset(fl)[var].sel(time=slice('2016', '2023'))
    
    if var in ['tp', 'e', 'cp', 'lsp', 'pev']:
        era5_sl_mon = era5_sl_mon * 1000
    elif var in ['msl']:
        era5_sl_mon = era5_sl_mon / 100
    elif var in ['sst', 't2m', 'd2m', 'skt']:
        era5_sl_mon = era5_sl_mon - zerok
    elif var in ['hcc', 'mcc', 'lcc', 'tcc']:
        era5_sl_mon = era5_sl_mon * 100
    elif var in ['z']:
        era5_sl_mon = era5_sl_mon / 9.80665
    elif var in ['mper']:
        era5_sl_mon = era5_sl_mon * seconds_per_d
    
    if var in ['e', 'pev', 'mper']:
        era5_sl_mon = era5_sl_mon * (-1)
    
    era5_sl_mon_alltime = mon_sea_ann(
        var_monthly=era5_sl_mon, lcopy=False, mm=True, sm=True, am=True,)
    
    ofile = f'data/sim/era5/mon/era5_sl_mon_alltime_{var}.pkl'
    if os.path.exists(ofile): os.remove(ofile)
    with open(ofile, 'wb') as f:
        pickle.dump(era5_sl_mon_alltime, f)
    
    del era5_sl_mon, era5_sl_mon_alltime




'''
#-------------------------------- check
era5_sl_mon_alltime = {}
for var in ['tciw', 'tclw', 'tcw', 'tcwv', 'tcsw', 'tcrw', 'tcslw']:
    print(f'#---------------- {var}')
    with open(f'data/sim/era5/mon/era5_sl_mon_alltime_{var}.pkl', 'rb') as f:
        era5_sl_mon_alltime[var] = pickle.load(f)

for var in ['tciw', 'tclw', 'tcw', 'tcwv', 'tcsw', 'tcrw', 'tcslw']:
    print(f'#---------------- {var}')
    print(era5_sl_mon_alltime[var]['am'].weighted(np.cos(np.deg2rad(era5_sl_mon_alltime[var]['am'].lat))).mean().values)

# tcw = tciw + tclw + tcwv + tcsw + tcrw





#-------------------------------- check

itime=-1
era5_sl_mon_alltime = {}
for var in ['inversionh', 'LCL', 'LTS', 'EIS']:
    # var = 'msnlwrf'
    # 'tp', 'msl', 'sst', 'hcc', 'mcc', 'lcc', 'tcc', '2t', 'msnlwrf', 'msnswrf', 'mtdwswrf', 'mtnlwrf', 'mtnswrf', 'msdwlwrf', 'msdwswrf', 'msdwlwrfcs', 'msdwswrfcs', 'msnlwrfcs', 'msnswrfcs', 'mtnlwrfcs', 'mtnswrfcs', 'cbh', 'tciw', 'tclw', 'e', 'z', 'mslhf', 'msshf', 'tcw', 'tcwv', 'tcsw', 'tcrw', 'tcslw', '10si', '2d', 'cp', 'lsp', 'deg0l', 'mper', 'pev', 'skt', '10u', '10v', '100u', '100v'
    print(f'#---------------- {var}')
    
    # fl = sorted([
    #     file for iyear in np.arange(1979, 2024, 1)
    #     for file in glob.glob(f'/g/data/rt52/era5/single-levels/monthly-averaged/{var}/{iyear}/*.nc')])
    # if var == '2t': var='t2m'
    # if var == '10si': var='si10'
    # if var == '2d': var='d2m'
    # if var == '10u': var='u10'
    # if var == '10v': var='v10'
    # if var == '100u': var='u100'
    # if var == '100v': var='v100'
    # ds = xr.open_dataset(fl[itime]).rename({'latitude': 'lat', 'longitude': 'lon'})[var].squeeze()
    
    fl = sorted(glob.glob(f'data/sim/era5/hourly/{var}/{var}_monthly_*.nc'))[:96]
    ds = xr.open_dataset(fl[itime])[var].squeeze()
    
    if var in ['tp', 'e', 'cp', 'lsp', 'pev']:
        ds = ds * 1000
    elif var in ['msl']:
        ds = ds / 100
    elif var in ['sst', 't2m', 'd2m', 'skt']:
        ds = ds - zerok
    elif var in ['hcc', 'mcc', 'lcc', 'tcc']:
        ds = ds * 100
    elif var in ['z']:
        ds = ds / 9.80665
    elif var in ['mper']:
        ds = ds * seconds_per_d
    ds = ds.astype(np.float32)
    
    with open(f'data/sim/era5/mon/era5_sl_mon_alltime_{var}.pkl', 'rb') as f:
        era5_sl_mon_alltime[var] = pickle.load(f)
    
    ds2 = era5_sl_mon_alltime[var]['mon'].isel(time=itime)
    ds2 = ds2.astype(np.float32)
    print((ds.values[np.isfinite(ds.values)] == ds2.values[np.isfinite(ds2.values)]).all())
    del era5_sl_mon_alltime[var]




# https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation

Mean surface latent heat flux: slhf
Mean surface net long-wave radiation flux: msnlwrf
Mean surface net short-wave radiation flux: msnswrf
Mean surface sensible heat flux: sshf
Mean surface downward long-wave radiation flux: msdwlwrf
Mean surface downward short-wave radiation flux: msdwswrf
Mean surface downward long-wave radiation flux, clear sky: msdwlwrfcs
Mean surface downward short-wave radiation flux, clear sky: msdwswrfcs
Mean surface net long-wave radiation flux, clear sky: msnlwrfcs
Mean surface net short-wave radiation flux, clear sky: msnswrfcs

Mean top downward short-wave radiation flux: mtdwswrf
Mean top net long-wave radiation flux: mtnlwrf
Mean top net short-wave radiation flux: mtnswrf
Mean top net long-wave radiation flux, clear sky: mtnlwrfcs
Mean top net short-wave radiation flux, clear sky: mtnswrfcs

Cloud base height: cbh
Total column cloud ice water: tciw
Total column cloud liquid water: tclw
tcw
tcwv

Evaporation: e
Geopotential: z


'''
# endregion


# region get era5 pl mon data

for var in ['q', 't', 'z']:
    # var = 'pv'
    print(var)
    
    fl = sorted([
        file for iyear in np.arange(1979, 2024, 1)
        for file in glob.glob(f'/g/data/rt52/era5/pressure-levels/monthly-averaged/{var}/{iyear}/*.nc')])
    # print(xr.open_dataset(fl[0])[var].units)
    
    era5_pl_mon = xr.open_mfdataset(fl, parallel=True).rename({'latitude': 'lat', 'longitude': 'lon'})[var]
    
    if var in ['q']:
        era5_pl_mon = era5_pl_mon * 1000
    elif var in ['t']:
        era5_pl_mon = era5_pl_mon - zerok
    elif var in ['z']:
        era5_pl_mon = era5_pl_mon / 9.80665
    
    era5_pl_mon_alltime = mon_sea_ann(
        var_monthly=era5_pl_mon, lcopy=False, mm=True, sm=True, am=True,)
    
    ofile = f'data/sim/era5/mon/era5_pl_mon_alltime_{var}.pkl'
    if os.path.exists(ofile): os.remove(ofile)
    with open(ofile, 'wb') as f:
        pickle.dump(era5_pl_mon_alltime, f)
    
    del era5_pl_mon, era5_pl_mon_alltime




'''
#-------------------------------- check
itime = -1
era5_pl_mon_alltime = {}
for var in ['pv', 'q', 'r', 't', 'u', 'v', 'w', 'z']:
    # var = 'pv'
    print(f'#-------------------------------- {var}')
    
    fl = sorted([
        file for iyear in np.arange(1979, 2024, 1)
        for file in glob.glob(f'/g/data/rt52/era5/pressure-levels/monthly-averaged/{var}/{iyear}/*.nc')])
    
    ds = xr.open_dataset(fl[itime]).rename({'latitude': 'lat', 'longitude': 'lon'})[var].squeeze()
    if var in ['q']:
        ds = ds * 1000
    elif var in ['t']:
        ds = ds - zerok
    elif var in ['z']:
        ds = ds / 9.80665
    ds = ds.astype(np.float32)
    
    with open(f'data/sim/era5/mon/era5_pl_mon_alltime_{var}.pkl', 'rb') as f:
        era5_pl_mon_alltime[var] = pickle.load(f)
    ds2 = era5_pl_mon_alltime[var]['mon'].isel(time=itime)
    ds2 = ds2.astype(np.float32)
    print((ds.values[np.isfinite(ds.values)] == ds2.values[np.isfinite(ds2.values)]).all())
    del era5_pl_mon_alltime[var]





'''
# endregion


# region derive era5 sl mon data


era5_sl_mon_alltime = {}
for var1, var2, var3 in zip(['mtuwswrf'], ['mtnswrf'], ['mtdwswrf']):
    # ['mtnlwrfcl', 'mtnswrfcl', 'mtuwswrfcl'], ['mtnlwrf', 'mtnswrf', 'mtuwswrf'], ['mtnlwrfcs', 'mtnswrfcs', 'mtuwswrfcs']
    # ['mtuwswrfcs'], ['mtnswrfcs'], ['mtdwswrf']
    # ['mtuwswrf'], ['mtnswrf'], ['mtdwswrf']
    # ['msnlwrfcl', 'msnswrfcl', 'msdwlwrfcl', 'msdwswrfcl', 'msuwlwrfcl', 'msuwswrfcl'], ['msnlwrf', 'msnswrf', 'msdwlwrf', 'msdwswrf', 'msuwlwrf', 'msuwswrf'], ['msnlwrfcs', 'msnswrfcs', 'msdwlwrfcs', 'msdwswrfcs', 'msuwlwrfcs', 'msuwswrfcs']
    # ['msuwswrfcs'], ['msnswrfcs'], ['msdwswrfcs']
    # ['msuwlwrfcs'], ['msnlwrfcs'], ['msdwlwrfcs']
    # ['msuwswrf'], ['msnswrf'], ['msdwswrf']
    # ['msuwlwrf'], ['msnlwrf'], ['msdwlwrf']
    print(f'Derive {var1} from {var2} and {var3}')
    print(str(datetime.datetime.now())[11:19])
    
    with open(f'data/sim/era5/mon/era5_sl_mon_alltime_{var2}.pkl', 'rb') as f:
        era5_sl_mon_alltime[var2] = pickle.load(f)
    with open(f'data/sim/era5/mon/era5_sl_mon_alltime_{var3}.pkl', 'rb') as f:
        era5_sl_mon_alltime[var3] = pickle.load(f)
    
    if var1 in ['msuwlwrf', 'msuwswrf', 'msuwlwrfcs', 'msuwswrfcs', 'msnlwrfcl', 'msnswrfcl', 'msdwlwrfcl', 'msdwswrfcl', 'msuwlwrfcl', 'msuwswrfcl', 'mtuwswrf', 'mtuwswrfcs', 'mtnlwrfcl', 'mtnswrfcl', 'mtuwswrfcl']:
        print('var2 - var3')
        era5_sl_mon = (era5_sl_mon_alltime[var2]['mon'] - era5_sl_mon_alltime[var3]['mon']).rename(var1)
    
    era5_sl_mon_alltime[var1] = mon_sea_ann(
        var_monthly=era5_sl_mon, lcopy=False, mm=True, sm=True, am=True)
    
    with open(f'data/sim/era5/mon/era5_sl_mon_alltime_{var1}.pkl', 'wb') as f:
        pickle.dump(era5_sl_mon_alltime[var1], f)
    
    del era5_sl_mon_alltime[var2], era5_sl_mon_alltime[var3], era5_sl_mon_alltime[var1], era5_sl_mon
    print(str(datetime.datetime.now())[11:19])




'''
#-------------------------------- check
itime=-1
era5_sl_mon_alltime = {}
for var1, var2, var3 in zip(
    ['msuwlwrf',  'msuwswrf',  'msuwlwrfcs',  'msuwswrfcs',  'msnlwrfcl', 'msnswrfcl', 'msdwlwrfcl', 'msdwswrfcl', 'msuwlwrfcl', 'msuwswrfcl',  'mtuwswrf',  'mtuwswrfcs',  'mtnlwrfcl', 'mtnswrfcl', 'mtuwswrfcl'],
    ['msnlwrf',  'msnswrf',  'msnlwrfcs',  'msnswrfcs',  'msnlwrf', 'msnswrf', 'msdwlwrf', 'msdwswrf', 'msuwlwrf', 'msuwswrf',  'mtnswrf',  'mtnswrfcs',  'mtnlwrf', 'mtnswrf', 'mtuwswrf'],
    ['msdwlwrf',  'msdwswrf',  'msdwlwrfcs',  'msdwswrfcs',  'msnlwrfcs', 'msnswrfcs', 'msdwlwrfcs', 'msdwswrfcs', 'msuwlwrfcs', 'msuwswrfcs',  'mtdwswrf',  'mtdwswrf',  'mtnlwrfcs', 'mtnswrfcs', 'mtuwswrfcs']):
    # var1='msuwlwrf'; var2='msnlwrf'; var3='msdwlwrf'
    print(f'Derive {var1} from {var2} and {var3}')
    
    with open(f'data/sim/era5/mon/era5_sl_mon_alltime_{var1}.pkl', 'rb') as f:
        era5_sl_mon_alltime[var1] = pickle.load(f)
    with open(f'data/sim/era5/mon/era5_sl_mon_alltime_{var2}.pkl', 'rb') as f:
        era5_sl_mon_alltime[var2] = pickle.load(f)
    with open(f'data/sim/era5/mon/era5_sl_mon_alltime_{var3}.pkl', 'rb') as f:
        era5_sl_mon_alltime[var3] = pickle.load(f)
    
    data1 = era5_sl_mon_alltime[var1]['mon'][itime].astype(np.float32)
    data2 = (era5_sl_mon_alltime[var2]['mon'][itime] - era5_sl_mon_alltime[var3]['mon'][itime]).astype(np.float32)
    print((data1.values[np.isfinite(data1.values)] == data2.values[np.isfinite(data2.values)]).all())
    del era5_sl_mon_alltime[var1], era5_sl_mon_alltime[var2], era5_sl_mon_alltime[var3]



# check
era5_sl_mon_alltime = {}
with open(f'data/sim/era5/mon/era5_sl_mon_alltime_mtuwswrfcl.pkl', 'rb') as f:
    era5_sl_mon_alltime['mtuwswrfcl'] = pickle.load(f)
with open(f'data/sim/era5/mon/era5_sl_mon_alltime_mtnswrfcl.pkl', 'rb') as f:
    era5_sl_mon_alltime['mtnswrfcl'] = pickle.load(f)

np.max(np.abs(era5_sl_mon_alltime['mtuwswrfcl']['am'].values - era5_sl_mon_alltime['mtnswrfcl']['am'].values))

with open(f'data/sim/era5/mon/era5_sl_mon_alltime_mtuwswrf.pkl', 'rb') as f:
    era5_sl_mon_alltime['mtuwswrf'] = pickle.load(f)

'''
# endregion


# region derive era5 sl mon data 2

for var1, vars in zip(['toa_albedo', 'toa_albedocs', 'toa_albedocl'], [['mtuwswrf', 'mtdwswrf'], ['mtuwswrfcs', 'mtdwswrf'], ['mtuwswrfcl', 'mtdwswrf']]):
    # var1='toa_albedo'; vars=['mtuwswrf', 'mtdwswrf']
    # ['msnrf'], [['msdwlwrf', 'msdwswrf', 'msuwlwrf', 'msuwswrf', 'mslhf', 'msshf']]
    print(f'#-------------------------------- Derive {var1} from {vars}')
    
    era5_sl_mon_alltime = {}
    for var2 in vars:
        print(f'#---------------- {var2}')
        with open(f'data/sim/era5/mon/era5_sl_mon_alltime_{var2}.pkl','rb') as f:
            era5_sl_mon_alltime[var2] = pickle.load(f)
    
    if var1 in ['msnrf']:
        era5_sl_mon = sum(era5_sl_mon_alltime[var2]['mon'] for var2 in vars).rename(var1).compute()
        era5_sl_mon_alltime[var1] = mon_sea_ann(
            var_monthly=era5_sl_mon, lcopy=True, mm=True, sm=True, am=True)
    elif var1 in ['toa_albedo', 'toa_albedocs', 'toa_albedocl']:
        era5_sl_mon_alltime[var1] = {}
        for ialltime in era5_sl_mon_alltime[vars[0]].keys():
            print(f'#-------- {ialltime}')
            era5_sl_mon_alltime[var1][ialltime] = (era5_sl_mon_alltime[vars[0]][ialltime] / era5_sl_mon_alltime[vars[1]][ialltime] * (-1)).compute()
    
    ofile = f'data/sim/era5/mon/era5_sl_mon_alltime_{var1}.pkl'
    if os.path.exists(ofile): os.remove(ofile)
    with open(ofile,'wb') as f:
        pickle.dump(era5_sl_mon_alltime[var1], f)
    del era5_sl_mon_alltime


'''
#-------------------------------- check ['toa_albedo', 'toa_albedocs', 'toa_albedocl']
ilat = 100
ilon = 100
for var1, vars in zip(['toa_albedo', 'toa_albedocs', 'toa_albedocl'], [['mtuwswrf', 'mtdwswrf'], ['mtuwswrfcs', 'mtdwswrf'], ['mtuwswrfcl', 'mtdwswrf']]):
    # var1='toa_albedo'; vars=['mtuwswrf', 'mtdwswrf']
    # ['msnrf'], [['msdwlwrf', 'msdwswrf', 'msuwlwrf', 'msuwswrf', 'mslhf', 'msshf']]
    print(f'#-------------------------------- {var1} {vars}')
    
    era5_sl_mon_alltime = {}
    for var2 in [var1] + vars:
        print(f'#---------------- {var2}')
        with open(f'data/sim/era5/mon/era5_sl_mon_alltime_{var2}.pkl','rb') as f:
            era5_sl_mon_alltime[var2] = pickle.load(f)
    
    for ialltime in era5_sl_mon_alltime[var1].keys():
        print(f'#-------- {ialltime}')
        data1 = era5_sl_mon_alltime[var1][ialltime][:, ilat, ilon].values
        data2 = (era5_sl_mon_alltime[vars[0]][ialltime][:, ilat, ilon] / era5_sl_mon_alltime[vars[1]][ialltime][:, ilat, ilon] * (-1)).compute().values
        print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())




#-------------------------------- check msnrf
era5_sl_mon_alltime = {}
for var2 in ['msnrf', 'msdwlwrf', 'msdwswrf', 'msuwlwrf', 'msuwswrf', 'mslhf', 'msshf']:
    print(f'#---------------- {var2}')
    with open(f'data/sim/era5/mon/era5_sl_mon_alltime_{var2}.pkl','rb') as f:
        era5_sl_mon_alltime[var2] = pickle.load(f)

itime=-1
data1 = era5_sl_mon_alltime['msnrf']['mon'][itime].values
data2 = sum(era5_sl_mon_alltime[var]['mon'][itime] for var in ['msdwlwrf', 'msdwswrf', 'msuwlwrf', 'msuwswrf', 'mslhf', 'msshf']).values.astype('float32')
print((data1 == data2).all())

'''
# endregion


# region get era5 hourly data
# Memory Used: 165.03GB; Walltime Used: 00:10:35

var = 'lcc' # ['mtnswrf', 'mtdwswrf', 'mtnlwrf', 'tcwv', 'tclw', 'tciw', 'lcc', 'mcc', 'hcc', 'tcc', 'tp', '2t']
print(f'#-------------------------------- {var}')
odir = f'scratch/data/sim/era5/{var}'
os.makedirs(odir, exist_ok=True)

# year=2020; month=1
def process_year_month(year, month, var, odir):
    print(f'#---------------- {year} {month:02d}')
    
    ifile = glob.glob(f'/g/data/rt52/era5/single-levels/reanalysis/{var}/{year}/{var}_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}??.nc')[0]
    if var == '2t': var='t2m'
    if var == '10si': var='si10'
    if var == '2d': var='d2m'
    if var == '10u': var='u10'
    if var == '10v': var='v10'
    if var == '100u': var='u100'
    if var == '100v': var='v100'
    ofile = f'{odir}/{var}_hourly_{year}{month:02d}.nc'
    ds = xr.open_dataset(ifile, chunks={}).rename({'latitude': 'lat', 'longitude': 'lon'})[var]
    
    if var in ['tp', 'e', 'cp', 'lsp', 'pev']:
        ds = ds * 1000
    elif var in ['msl']:
        ds = ds / 100
    elif var in ['sst', 't2m', 'd2m', 'skt']:
        ds = ds - zerok
    elif var in ['hcc', 'mcc', 'lcc', 'tcc']:
        ds = ds * 100
    elif var in ['z']:
        ds = ds / 9.80665
    elif var in ['mper']:
        ds = ds * seconds_per_d
    
    if var in ['e', 'pev', 'mper']:
        ds = ds * (-1)
    
    ds = ds.groupby('time.hour').mean().astype(np.float32).expand_dims(dim={'time': [ds.time[0].values]}).compute()
    
    if os.path.exists(ofile): os.remove(ofile)
    ds.to_netcdf(ofile)
    
    del ds
    return f'Finished processing {ofile}'


joblib.Parallel(n_jobs=48)(joblib.delayed(process_year_month)(year, month, var, odir) for year in range(1979, 2024) for month in range(1, 13))




'''
#-------------------------------- check

year=2020; month=6

for var in ['mtnswrf', 'mtdwswrf', 'mtnlwrf']:
    # var = 'lcc'
    print(f'#-------------------------------- {var}')
    odir = f'data/sim/era5/{var}'
    
    ifile = glob.glob(f'/g/data/rt52/era5/single-levels/reanalysis/{var}/{year}/{var}_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}??.nc')[0]
    if var == '2t': var='t2m'
    if var == '10si': var='si10'
    if var == '2d': var='d2m'
    if var == '10u': var='u10'
    if var == '10v': var='v10'
    if var == '100u': var='u100'
    if var == '100v': var='v100'
    ofile = f'{odir}/{var}_hourly_{year}{month:02d}.nc'
    
    ds = xr.open_dataset(ifile, chunks={}).rename({'latitude': 'lat', 'longitude': 'lon'})[var]
    if var in ['tp', 'e', 'cp', 'lsp', 'pev']:
        ds = ds * 1000
    elif var in ['msl']:
        ds = ds / 100
    elif var in ['sst', 't2m', 'd2m', 'skt']:
        ds = ds - zerok
    elif var in ['hcc', 'mcc', 'lcc', 'tcc']:
        ds = ds * 100
    elif var in ['z']:
        ds = ds / 9.80665
    elif var in ['mper']:
        ds = ds * seconds_per_d
    
    if var in ['e', 'pev', 'mper']:
        ds = ds * (-1)
    
    ds = ds.groupby('time.hour').mean().astype(np.float32).expand_dims(dim={'time': [ds.time[0].values]}).compute()
    
    ds_out = xr.open_dataset(ofile)[var]
    
    print((ds.values == ds_out.values).all())

'''
# endregion


# region get era5 alltime hourly data


for var in ['mtnswrf', 'mtdwswrf', 'mtnlwrf']:
    # var = 'lcc'
    # ['tcwv', 'tclw', 'tciw', 'lcc', 'mcc', 'hcc', 'tcc', 'tp', '2t']
    print(f'#-------------------------------- {var}')
    odir = f'data/sim/era5/{var}'
    
    if var == '2t': var='t2m'
    if var == '10si': var='si10'
    if var == '2d': var='d2m'
    if var == '10u': var='u10'
    if var == '10v': var='v10'
    if var == '100u': var='u100'
    if var == '100v': var='v100'
    
    fl = sorted(glob.glob(f'{odir}/{var}_hourly_*.nc'))
    era5_hourly = xr.open_mfdataset(fl)[var].sel(time=slice('1979', '2023'))
    era5_hourly_alltime = mon_sea_ann(
        var_monthly=era5_hourly, lcopy=False, mm=True, sm=True, am=True)
    
    ofile = f'data/sim/era5/hourly/era5_hourly_alltime_{var}.pkl'
    if os.path.exists(ofile): os.remove(ofile)
    with open(ofile,'wb') as f:
        pickle.dump(era5_hourly_alltime, f)
    
    del era5_hourly, era5_hourly_alltime




'''
#-------------------------------- check
ifile = 10
for var in ['tcwv', 'tclw', 'tciw']:
    # var = 'lcc'
    # ['lcc', 'mcc', 'hcc', 'tcc', 'tp', '2t']
    print(f'#-------------------------------- {var}')
    odir = f'data/sim/era5/{var}'
    
    if var == '2t': var='t2m'
    if var == '10si': var='si10'
    if var == '2d': var='d2m'
    if var == '10u': var='u10'
    if var == '10v': var='v10'
    if var == '100u': var='u100'
    if var == '100v': var='v100'
    
    fl = sorted(glob.glob(f'{odir}/{var}_hourly_*.nc'))
    ds = xr.open_dataset(fl[ifile])
    
    with open(f'data/sim/era5/hourly/era5_hourly_alltime_{var}.pkl','rb') as f:
        era5_hourly_alltime = pickle.load(f)
    
    print((era5_hourly_alltime['mon'][ifile] == ds[var].squeeze()).all().values)
    del era5_hourly_alltime


'''
# endregion


# region derive era5 alltime hourly data


for var1, vars in zip(['mtuwswrf'], [['mtnswrf', 'mtdwswrf']]):
    # var1='mtuwswrf'; vars=['mtnswrf', 'mtdwswrf']
    print(f'#-------------------------------- Derive {var1} from {vars}')
    
    era5_hourly_alltime = {}
    for var2 in vars:
        print(f'#---------------- {var2}')
        with open(f'data/sim/era5/hourly/era5_hourly_alltime_{var2}.pkl','rb') as f:
            era5_hourly_alltime[var2] = pickle.load(f)
    
    era5_hourly_alltime[var1] = {}
    for ialltime in era5_hourly_alltime[vars[0]].keys():
        # ialltime = 'am'
        print(f'#-------- {ialltime}')
        
        if var1 in ['mtuwswrf']:
            print(f'{var1} = {vars[0]} - {vars[1]}')
            era5_hourly_alltime[var1][ialltime] = (era5_hourly_alltime[vars[0]][ialltime] - era5_hourly_alltime[vars[1]][ialltime]).rename(var1).compute()
    
    ofile = f'data/sim/era5/hourly/era5_hourly_alltime_{var1}.pkl'
    if os.path.exists(ofile): os.remove(ofile)
    with open(ofile,'wb') as f:
        pickle.dump(era5_hourly_alltime[var1], f)
    del era5_hourly_alltime




'''
#-------------------------------- check
ilat = 100
ilon = 100
for var1, vars in zip(['mtuwswrf'], [['mtnswrf', 'mtdwswrf']]):
    # var1='mtuwswrf'; vars=['mtnswrf', 'mtdwswrf']
    print(f'#-------------------------------- {var1} and {vars}')
    
    era5_hourly_alltime = {}
    for var2 in [var1]+vars:
        print(f'#---------------- {var2}')
        with open(f'data/sim/era5/hourly/era5_hourly_alltime_{var2}.pkl','rb') as f:
            era5_hourly_alltime[var2] = pickle.load(f)
    
    for ialltime in era5_hourly_alltime[vars[0]].keys():
        # ialltime = 'am'
        print(f'#-------- {ialltime}')
        
        if var1 in ['mtuwswrf']:
            print(f'{var1} = {vars[0]} - {vars[1]}')
            print((era5_hourly_alltime[var1][ialltime][:, :, ilat, ilon].values == (era5_hourly_alltime[vars[0]][ialltime][:, :, ilat, ilon] - era5_hourly_alltime[vars[1]][ialltime][:, :, ilat, ilon]).values).all())

'''
# endregion




# region get era5 inversionh, LCL, LTS, EIS
# get_inversion_numba:  Memory Used: 799.59GB,  Walltime Used: 01:26:26
# get_LCL:              Memory Used: 150.88GB,  Walltime Used: 02:12:26
# get_LTS:              Memory Used:
# get_EIS:              Memory Used: 208.88GB,  Walltime Used: 02:21:16


var = 'inversionh' # ['inversionh', 'LCL', 'LTS', 'EIS']
print(f'#-------------------------------- {var}')
odir = f'data/sim/era5/hourly/{var}'
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
        dss[ivar] = xr.open_dataset(f'data/sim/era5/hourly/{ivar}/{ivar}_hourly_{year}{month:02d}.nc')[ivar]
    
    if not ivar in ['LCL', 'LTS']:
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

year=2018; month=1
print(f'#-------------------------------- {year} {month:02d}')
vars = ['LTS', 'LCL', 'inversionh', 'EIS', #
        'ta', 'zg', 'orog', 'zg700', 'tas', 'ps', 'hurs', 'ta700', #
        ]

dss = {}
for ivar in vars:
    print(f'#---------------- {ivar}')
    if ivar in ['ta', 'zg']:
        # ivar = 'zg'
        dss[ivar] = xr.open_dataset(f'/g/data/rt52/era5/pressure-levels/reanalysis/{cmip6_era5_var[ivar]}/{year}/{cmip6_era5_var[ivar]}_era5_oper_pl_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc').rename({'level': 'pressure'})[cmip6_era5_var[ivar]].sortby('pressure', ascending=False)
        # if ivar == 'zg': dss[ivar] = geopotential_to_height(dss[ivar])
    elif ivar in ['orog']:
        # ivar = 'orog'
        dss[ivar] = xr.open_dataset('/g/data/rt52/era5/single-levels/reanalysis/z/2020/z_era5_oper_sfc_20200601-20200630.nc')['z'][0]
        # dss[ivar] = geopotential_to_height(dss[ivar])
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
            # dss[ivar] = geopotential_to_height(dss[ivar])
        else:
            dss[ivar] = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/{cmip6_era5_var[ivar]}/{year}/{cmip6_era5_var[ivar]}_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')[cmip6_era5_var[ivar]]
    elif ivar in ['LCL', 'LTS', 'inversionh', 'EIS']:
        dss[ivar] = xr.open_dataset(f'data/sim/era5/hourly/{ivar}/{ivar}_hourly_{year}{month:02d}.nc')[ivar]
    
    if not ivar in ['LCL', 'LTS', 'inversionh', 'EIS']:
        dss[ivar] = dss[ivar].rename({'latitude': 'lat', 'longitude': 'lon'})

itime = 20
ilat  = 40
ilon  = 40

for var in ['LTS', 'LCL', 'inversionh', 'EIS', ]: #
    print(f'#---------------- {var}')
    print(dss[var][itime, ilat, ilon].values)
    if var == 'inversionh':
        # var = 'inversionh'
        print(get_inversion(
            dss['ta'][itime, :, ilat, ilon].values,
            geopotential_to_height(dss['zg'][itime, :, ilat, ilon].values * units("m2/s2")).m,
            geopotential_to_height(dss['orog'][ilat, ilon].values * units("m2/s2")).m
            ))
        print(get_inversion_numba(
            dss['ta'][itime, :, ilat, ilon].values,
            geopotential_to_height(dss['zg'][itime, :, ilat, ilon].values * units("m2/s2")).m,
            geopotential_to_height(dss['orog'][ilat, ilon].values * units("m2/s2")).m
            ))
    elif var == 'LCL':
        # var = 'LCL'
        print(get_LCL(
            dss['ps'][itime, ilat, ilon].values,
            dss['tas'][itime, ilat, ilon].values,
            dss['hurs'][itime, ilat, ilon].values,
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
            dss['hurs'][itime, ilat, ilon].values,
            geopotential_to_height(dss['zg700'][itime, ilat, ilon].values * units("m2/s2")).m,
            geopotential_to_height(dss['orog'][ilat, ilon].values * units("m2/s2")).m,
        ))
        print(get_EIS_simplified(
            dss['LCL'][itime, ilat, ilon].values,
            dss['LTS'][itime, ilat, ilon].values,
            dss['tas'][itime, ilat, ilon].values,
            dss['ta700'][itime, ilat, ilon].values,
            geopotential_to_height(dss['zg700'][itime, ilat, ilon].values * units("m2/s2")).m,
            geopotential_to_height(dss['orog'][ilat, ilon].values * units("m2/s2")).m,
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


# region get monthly era5 inversionh, LCL, LTS, EIS
# 4 vars, 8 years, 12 months: NCPUs Used: 96; Memory Used: 1.17TB; Walltime Used: 00:20:50

vars = ['inversionh', 'LCL', 'LTS', 'EIS']

def get_mon_from_hour(var, year, month):
    # var = 'LTS'; year=2024; month=1
    print(f'#---------------- {var} {year} {month:02d}')
    
    ifile = f'data/sim/era5/hourly/{var}/{var}_hourly_{year}{month:02d}.nc'
    ofile = f'data/sim/era5/hourly/{var}/{var}_monthly_{year}{month:02d}.nc'
    
    ds_in = xr.open_dataset(ifile)[var]
    ds_out = ds_in.resample({'time': '1ME'}).mean(skipna=True).compute()
    
    if os.path.exists(ofile): os.remove(ofile)
    ds_out.to_netcdf(ofile)
    
    del ds_in, ds_out
    return f'Finished processing {ofile}'

joblib.Parallel(n_jobs=4)(joblib.delayed(get_mon_from_hour)(var, year, month) for var in vars for year in range(2024, 2025) for month in range(1, 2))


'''
#---- check
var = 'EIS'
year = 2016
month = 6
ds_in = xr.open_dataset(f'data/sim/era5/hourly/{var}/{var}_hourly_{year}{month:02d}.nc')[var]
ds_out = xr.open_dataset(f'data/sim/era5/hourly/{var}/{var}_monthly_{year}{month:02d}.nc')[var]

ilat = 100
ilon = 100
print(np.nanmean(ds_in[:, ilat, ilon].values) - ds_out[0, ilat, ilon].values)

'''
# endregion

