

# qsub -I -q normal -l walltime=1:00:00,ncpus=1,mem=192GB,jobfs=100MB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22+gdata/gx60


# region get command line options

import argparse
parser=argparse.ArgumentParser()
parser.add_argument('-y', '--year', type=int, required=True,)
parser.add_argument('-m', '--month', type=int, required=True,)
args = parser.parse_args()

year=args.year
month=args.month

# endregion


# region import packages

import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
import pandas as pd
import glob
from datetime import datetime
import os
import calendar
import pickle

import sys  # print(sys.path)
sys.path.append('/home/563/qg8515/code/gbr_future/module')
from calculations import mon_sea_ann

# endregion


# region get daily count of each cloud type
# walltime=10:00:00, mem=20GB (It took around 20 min at the fastest speed)

ISCCP_types = {'Clear': 0,
               'Cirrus': 1, 'Cirrostratus': 2, 'Deep convection': 3,
               'Altocumulus': 4, 'Altostratus': 5, 'Nimbostratus':6,
               'Cumulus':7, 'Stratocumulus': 8, 'Stratus': 9,
            #    'Unknown':10,
               }

for year in np.arange(year, year+1, 1):
    # year=2015
    print(f'#-------------------------------- {year}')
    for month in np.arange(month, month+1, 1):
        # month=7
        print(f'#---------------- {month:02d}')
        ymfolder = f'data/obs/jaxa/clp/{year}{month:02d}'
        if os.path.isdir(ymfolder):
            for day in np.arange(1, calendar.monthrange(year, month)[1]+1, 1):
                # day=4
                print(f'#-------- {day:02d}')
                dfolder = f'{ymfolder}/{day:02d}'
                if os.path.isdir(dfolder):
                    fl = sorted(glob.glob(f'{dfolder}/??/CLTYPE_*.nc'))
                    ds = xr.open_mfdataset(fl)
                    cltype_count = xr.DataArray(
                        name='cltype_count',
                        data=np.zeros((1, 11, len(ds.latitude), len(ds.longitude))),
                        dims=['time', 'types', 'lat', 'lon'],
                        coords={
                            'time': [datetime.strptime(f'{year}-{month:02d}-{day:02d}', '%Y-%m-%d')],
                            'types': ['finite'] + list(ISCCP_types.keys()),
                            'lat': ds.latitude.values,
                            'lon': ds.longitude.values
                        }
                    )
                    cltype_values = ds.CLTYPE.values
                    cltype_count.loc[{'types': 'finite'}][0] = np.isfinite(cltype_values).sum(axis=0)
                    for itype in list(ISCCP_types.keys()):
                        print(f'#---- {itype}')
                        cltype_count.loc[{'types': itype}][0] = (cltype_values == ISCCP_types[itype]).sum(axis=0)
                    check = (cltype_count[0, 0] == cltype_count[0, 1:].sum(axis=0)).all().values
                    if not check:
                        print('Warning 3: Sum of all cloud types != finite counts')
                    ofile = f'{dfolder}/cltype_count_{year}{month:02d}{day:02d}.nc'
                    if os.path.exists(ofile): os.remove(ofile)
                    cltype_count.to_netcdf(ofile)
                    del ds, cltype_values, cltype_count
                else:
                    print(f'Warning 2: No folder for year {year} month {month:02d} day {day:02d}')
        else:
            print(f'Warning 1: No folder for year {year} month {month:02d}')


'''
#-------------------------------- check
ISCCP_types = {'Clear': 0,
               'Cirrus': 1, 'Cirrostratus': 2, 'Deep convection': 3,
               'Altocumulus': 4, 'Altostratus': 5, 'Nimbostratus':6,
               'Cumulus':7, 'Stratocumulus': 8, 'Stratus': 9,
            #    'Unknown':10,
               }
year=2023
month=12
day=31
icloud='Stratocumulus'

ymfolder = f'data/obs/jaxa/clp/{year}{month:02d}'
dfolder = f'{ymfolder}/{day:02d}'
fl = sorted(glob.glob(f'{dfolder}/??/CLTYPE_{year}{month:02d}{day:02d}????.nc'))
ds = xr.open_mfdataset(fl)
cltype_values = ds.CLTYPE.values
cltype_count = xr.open_dataset(f'{dfolder}/cltype_count_{year}{month:02d}{day:02d}.nc')['cltype_count']
print((cltype_count[0, 0] == cltype_count[0, 1:].sum(axis=0)).all().values)
print((np.isfinite(cltype_values).sum(axis=0) == cltype_count[0, 0].values).all())
print(((cltype_values == ISCCP_types[icloud]).sum(axis=0) == cltype_count.loc[{'types': icloud}][0].values).all())


'''
# endregion


# region get alltime count of each cloud type
# Memory Used: 367.07GB, Walltime Used: 03:49:17

fl = sorted(glob.glob('data/obs/jaxa/clp/??????/??/cltype_count_????????.nc'))
cltype_count = xr.open_mfdataset(fl).cltype_count

cltype_count_alltime = {}

cltype_count_alltime['daily'] = cltype_count

print('get mon')
cltype_count_alltime['mon'] = cltype_count.resample({'time': '1ME'}).sum().compute()

print('get sea')
cltype_count_alltime['sea'] = cltype_count_alltime['mon'].resample({'time': 'QE-FEB'}).sum()[1:-1].compute()

print('get ann')
cltype_count_alltime['ann'] = cltype_count_alltime['mon'].resample({'time': '1YE'}).sum().compute()

print('get mm')
cltype_count_alltime['mm'] = cltype_count_alltime['mon'].groupby('time.month').sum().compute()
cltype_count_alltime['mm'] = cltype_count_alltime['mm'].rename({'month': 'time'})

print('get sm')
cltype_count_alltime['sm'] = cltype_count_alltime['sea'].groupby('time.season').sum().compute()
cltype_count_alltime['sm'] = cltype_count_alltime['sm'].rename({'season': 'time'})

print('get am')
cltype_count_alltime['am'] = cltype_count_alltime['ann'].sum(dim='time').compute()
cltype_count_alltime['am'] = cltype_count_alltime['am'].expand_dims('time', axis=0)

print('output data')
ofile='data/obs/jaxa/clp/cltype_count_alltime.pkl'
if os.path.exists(ofile): os.remove(ofile)
with open(ofile, 'wb') as f:
    pickle.dump(cltype_count_alltime, f)


'''
#-------------------------------- check
with open('data/obs/jaxa/clp/cltype_count_alltime.pkl', 'rb') as f:
    cltype_count_alltime = pickle.load(f)

ifile = -1
itype = 'Stratocumulus'
ilat = 100
ilon = 100

fl = sorted(glob.glob('data/obs/jaxa/clp/??????/??/cltype_count_????????.nc'))
ds = xr.open_dataset(fl[ifile])

print((ds.cltype_count.loc[{'types': itype}][0].values == cltype_count_alltime['daily'][ifile].loc[{'types': itype}].values).all())

print((cltype_count_alltime['mon'].loc[{'types': itype}][0, ilat, ilon] == cltype_count_alltime['daily'].loc[{'types': itype}][:28, ilat, ilon].resample({'time': '1ME'}).sum()).all().values)

print((cltype_count_alltime['sea'].loc[{'types': itype}][:, ilat, ilon] == cltype_count_alltime['mon'].loc[{'types': itype}][:, ilat, ilon].resample({'time': 'QE-FEB'}).sum()[1:-1]).all().values)

print((cltype_count_alltime['ann'].loc[{'types': itype}][:, ilat, ilon] == cltype_count_alltime['mon'].loc[{'types': itype}][:, ilat, ilon].resample({'time': '1YE'}).sum()[1:].compute()).all().values)

'''
# endregion


# region get alltime frequency of each cloud type
# Memory Used: 323.52GB, Walltime Used: 00:11:15

with open('data/obs/jaxa/clp/cltype_count_alltime.pkl', 'rb') as f:
    cltype_count_alltime = pickle.load(f)

cltype_frequency_alltime = {}
for ialltime in ['mon', 'sea', 'ann', 'mm', 'sm', 'am']:
    # ialltime = 'mon'
    print(f'#-------------------------------- {ialltime}')
    
    cltype_frequency_alltime[ialltime] = xr.zeros_like(cltype_count_alltime[ialltime][:, 1:]).rename('cltype_frequency')
    
    for itype in cltype_frequency_alltime[ialltime].types.values:
        # itype='Stratocumulus'
        print(f'#---------------- {itype}')
        cltype_frequency_alltime[ialltime].loc[{'types': itype}][:] = (cltype_count_alltime[ialltime].loc[{'types': itype}] / cltype_count_alltime[ialltime].loc[{'types': 'finite'}] * 100).compute().astype(np.float32)

ofile='data/obs/jaxa/clp/cltype_frequency_alltime.pkl'
if os.path.exists(ofile): os.remove(ofile)
with open(ofile, 'wb') as f:
    pickle.dump(cltype_frequency_alltime, f)


'''
#-------------------------------- check
with open('data/obs/jaxa/clp/cltype_count_alltime.pkl', 'rb') as f:
    cltype_count_alltime = pickle.load(f)
with open('data/obs/jaxa/clp/cltype_frequency_alltime.pkl', 'rb') as f:
    cltype_frequency_alltime = pickle.load(f)

ialltime = 'mon'
itype = 'Stratocumulus'
ilat = 100
ilon = 100

data1 = cltype_frequency_alltime[ialltime].loc[{'types': itype}][:, ilat, ilon]
data2 = (cltype_count_alltime[ialltime].loc[{'types': itype}][:, ilat, ilon] / cltype_count_alltime[ialltime].loc[{'types': 'finite'}][:, ilat, ilon] * 100).compute().astype(np.float32)
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all().values)

'''
# endregion


# region get hourly count of each cloud type
# Memory Used: 44.04GB, Walltime Used: Walltime Used: 03:11:18

ISCCP_types = {'Clear': 0,
               'Cirrus': 1, 'Cirrostratus': 2, 'Deep convection': 3,
               'Altocumulus': 4, 'Altostratus': 5, 'Nimbostratus':6,
               'Cumulus':7, 'Stratocumulus': 8, 'Stratus': 9,
            #    'Unknown':10,
               }

# year=2020; month=6
print(f'#-------------------------------- {year} {month}')

ymfolder = f'data/obs/jaxa/clp/{year}{month:02d}'
fl = sorted(glob.glob(f'{ymfolder}/??/??/CLTYPE_{year}{month:02d}??????.nc'))
if (len(fl) != 0):
    print(f'Number of Files: {len(fl)}')
    
    ds = xr.open_mfdataset(fl, parallel=True).CLTYPE
    cltype_hourly_count = xr.DataArray(
        name='cltype_hourly_count',
        data=np.zeros((1, 24, 11, len(ds.latitude), len(ds.longitude))),
        dims=['time', 'hour', 'types', 'lat', 'lon'],
        coords={
            'time': [datetime.strptime(f'{year}-{month:02d}-01', '%Y-%m-%d')],
            'hour': np.arange(0, 24, 1),
            'types': ['finite'] + list(ISCCP_types.keys()),
            'lat': ds.latitude.values,
            'lon': ds.longitude.values
            }
        )
    
    print(f'#---------------- finite')
    cltype_hourly_count.loc[{'types': 'finite'}][0] = np.isfinite(ds).groupby('time.hour').sum().values
    for itype in list(ISCCP_types.keys()):
        # itype=list(ISCCP_types.keys())[0]
        print(f'#---------------- {itype}')
        cltype_hourly_count.loc[{'types': itype}][0] = (ds == ISCCP_types[itype]).groupby('time.hour').sum().values
    
    ofile = f'{ymfolder}/cltype_hourly_count_{year}{month:02d}.nc'
    if os.path.exists(ofile): os.remove(ofile)
    cltype_hourly_count.to_netcdf(ofile)
    del ds, cltype_hourly_count
else:
    print('Warning: No file found')



'''
# qsub -I -q hugemem -l walltime=1:00:00,ncpus=1,mem=400GB,jobfs=100MB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22+gdata/gx60

#-------------------------------- check
ISCCP_types = {'Clear': 0,
               'Cirrus': 1, 'Cirrostratus': 2, 'Deep convection': 3,
               'Altocumulus': 4, 'Altostratus': 5, 'Nimbostratus':6,
               'Cumulus':7, 'Stratocumulus': 8, 'Stratus': 9,
            #    'Unknown':10,
               }
year=2020; month=6; itype='Cumulus'
ymfolder = f'data/obs/jaxa/clp/{year}{month:02d}'
fl = sorted(glob.glob(f'{ymfolder}/??/??/CLTYPE_{year}{month:02d}??????.nc'))
ds = xr.open_mfdataset(fl, parallel=True).CLTYPE
cltype_hourly_count = xr.open_dataset(f'{ymfolder}/cltype_hourly_count_{year}{month:02d}.nc').cltype_hourly_count

print((cltype_hourly_count.loc[{'types': itype}][0] == (ds == ISCCP_types[itype]).groupby('time.hour').sum().values).all())
print((cltype_hourly_count.loc[{'types': 'finite'}][0] == np.isfinite(ds).groupby('time.hour').sum().values).all())
print((cltype_hourly_count[0, :, 0] == cltype_hourly_count[0, :, 1:].sum(axis=1)).all().values)
'''
# endregion


# region get alltime hourly count of each cloud type
# Memory Used: 398.7GB, Walltime Used: 02:55:17


min_lon, max_lon, min_lat, max_lat = [110.58, 157.34, -43.69, -7.01]
fl = sorted(glob.glob(f'data/obs/jaxa/clp/??????/cltype_hourly_count_??????.nc'))
cltype_hourly_count = xr.open_mfdataset(fl, parallel=True).cltype_hourly_count.sel(lon=slice(min_lon, max_lon), lat=slice(max_lat, min_lat))
cltype_hourly_count_alltime = mon_sea_ann(
    var_monthly=cltype_hourly_count, lcopy=False, mm=True, sm=True, am=True)

ofile='data/obs/jaxa/clp/cltype_hourly_count_alltime.pkl'
if os.path.exists(ofile): os.remove(ofile)
with open(ofile, 'wb') as f:
    pickle.dump(cltype_hourly_count_alltime, f)




'''
#-------------------------------- check
with open('data/obs/jaxa/clp/cltype_hourly_count_alltime.pkl', 'rb') as f:
    cltype_hourly_count_alltime = pickle.load(f)
fl = sorted(glob.glob(f'data/obs/jaxa/clp/??????/cltype_hourly_count_??????.nc'))
ifile = 10
ds = xr.open_dataset(fl[ifile]).cltype_hourly_count
print((cltype_hourly_count_alltime['mon'][ifile] == ds).all().values)


#-------------------------------- check 2
with open('data/obs/jaxa/clp/cltype_hourly_count_alltime.pkl', 'rb') as f:
    cltype_hourly_count_alltime = pickle.load(f)
with open('data/obs/jaxa/clp/cltype_count_alltime.pkl', 'rb') as f:
    cltype_count_alltime = pickle.load(f)

min_lon, max_lon, min_lat, max_lat = [110.58, 157.34, -43.69, -7.01]
ialltime='mon'
itype = 'Stratocumulus'
itime=2
print((cltype_count_alltime[ialltime][itime].loc[{'types': itype}].sel(lon=slice(min_lon, max_lon), lat=slice(max_lat, min_lat)) == cltype_hourly_count_alltime[ialltime][itime].loc[{'types': itype}].sum(dim='hour')).all().values)


#-------------------------------- check 3
with open('data/obs/jaxa/clp/cltype_hourly_count_alltime.pkl', 'rb') as f:
    cltype_hourly_count_alltime = pickle.load(f)

itime=-1
ihour=3
ilat = 100
ilon = 100

for ialltime in cltype_hourly_count_alltime.keys():
    # ialltime='ann'
    print(f'#-------------------------------- {ialltime}')
    
    print(cltype_hourly_count_alltime[ialltime][itime, ihour, 0, ilat, ilon].values == cltype_hourly_count_alltime[ialltime][itime, ihour, 1:, ilat, ilon].sum(dim='types').values)
    print((cltype_hourly_count_alltime[ialltime][itime, ihour, 0, ilat, ilon].values - cltype_hourly_count_alltime[ialltime][itime, ihour, 1:, ilat, ilon].sum(dim='types').values) / cltype_hourly_count_alltime[ialltime][itime, ihour, 0, ilat, ilon].values)


'''
# endregion


# region get alltime hourly frequency of each cloud type
# Memory Used: 365.11GB; Walltime Used: 06:39:43

with open('data/obs/jaxa/clp/cltype_hourly_count_alltime.pkl', 'rb') as f:
    cltype_hourly_count_alltime = pickle.load(f)

cltype_hourly_frequency_alltime = {}
for ialltime in ['mon', 'sea', 'ann', 'mm', 'sm', 'am']:
    # ialltime = 'mon'
    print(f'#-------------------------------- {ialltime}')
    
    cltype_hourly_frequency_alltime[ialltime] = xr.zeros_like(cltype_hourly_count_alltime[ialltime][:, :, 1:]).rename('cltype_hourly_frequency')
    
    for itype in cltype_hourly_frequency_alltime[ialltime].types.values:
        # itype='Stratocumulus'
        print(f'#---------------- {itype}')
        cltype_hourly_frequency_alltime[ialltime].loc[{'types': itype}][:] = (cltype_hourly_count_alltime[ialltime].loc[{'types': itype}] / cltype_hourly_count_alltime[ialltime].loc[{'types': 'finite'}] * 100).compute().astype(np.float32)

ofile='data/obs/jaxa/clp/cltype_hourly_frequency_alltime.pkl'
if os.path.exists(ofile): os.remove(ofile)
with open(ofile, 'wb') as f:
    pickle.dump(cltype_hourly_frequency_alltime, f)



'''
#-------------------------------- check 1
# hourly frequency has problem, to be fixed

with open('data/obs/jaxa/clp/cltype_hourly_count_alltime.pkl', 'rb') as f:
    cltype_hourly_count_alltime = pickle.load(f)
with open('data/obs/jaxa/clp/cltype_hourly_frequency_alltime.pkl', 'rb') as f:
    cltype_hourly_frequency_alltime = pickle.load(f)

ialltime = 'ann'
itype = 'Stratocumulus'
itime=-1
ihour=3

data1 = cltype_hourly_frequency_alltime[ialltime].loc[{'types': itype}][itime, ihour].values
data2 = (cltype_hourly_count_alltime[ialltime].loc[{'types': itype}][itime, ihour] / cltype_hourly_count_alltime[ialltime].loc[{'types': 'finite'}][itime, ihour] * 100).compute().astype(np.float32).values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())


#-------------------------------- check 3
with open('data/obs/jaxa/clp/cltype_hourly_frequency_alltime.pkl', 'rb') as f:
    cltype_hourly_frequency_alltime = pickle.load(f)
with open('data/obs/jaxa/clp/cltype_hourly_count_alltime.pkl', 'rb') as f:
    cltype_hourly_count_alltime = pickle.load(f)

for ialltime in list(cltype_hourly_frequency_alltime.keys())[1:]:
    # ialltime='ann'
    print(f'#-------------------------------- {ialltime}')
    itime = -1
    ihour = 3
    ilat = 100
    ilon = 100
    print(cltype_hourly_frequency_alltime[ialltime][itime, ihour, :, ilat, ilon].values)
    print(cltype_hourly_frequency_alltime[ialltime][itime, ihour, :, ilat, ilon].values.sum())



# check nan values there
ialltime = 'ann'
itime = -1
ihour = 3
itypes, ilats, ilons = np.where(np.isnan(cltype_hourly_frequency_alltime[ialltime][itime, ihour]))
# when you sum it up, nan disappears
print(np.isnan(cltype_hourly_frequency_alltime[ialltime][itime, ihour].sum(dim='types', skipna=False)).sum())
for itype, ilat, ilon in zip(itypes, ilats, ilons):
    print(cltype_hourly_frequency_alltime[ialltime][itime, ihour, itype, ilat, ilon].values)


# check zero values there
ialltime = 'ann'
itime = -1
ihour = 3
ilats, ilons = np.where(cltype_hourly_frequency_alltime[ialltime][itime, ihour, :].sum(dim='types') == 0)

for ilat, ilon in zip(ilats, ilons):
    print(np.max(cltype_hourly_count_alltime[ialltime][itime, ihour, :, ilat, ilon]).values)

cltype_hourly_frequency_alltime[ialltime][itime, ihour, :].sum(dim='types')[ilats[0], ilons[0]]
cltype_hourly_count_alltime[ialltime][itime, ihour, :, ilats[0], ilons[0]]


# check how many nan is there
itime = -1
ihour = 3
for ialltime in list(cltype_hourly_frequency_alltime.keys())[1:]:
    # ialltime='am'
    print(f'#-------------------------------- {ialltime}')
    print((np.isnan(cltype_hourly_frequency_alltime[ialltime][itime, ihour])).sum().values)


'''
# endregion

