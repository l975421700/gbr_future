

# qsub -I -q hugemem -l walltime=4:00:00,ncpus=1,mem=1470GB,jobfs=100GB,storage=gdata/v46+scratch/v46


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

# endregion


# region get daily count of each cloud type

ISCCP_types = {'Clear': 0,
               'Cirrus': 1, 'Cirrostratus': 2, 'Deep convection': 3,
               'Altocumulus': 4, 'Altostratus': 5, 'Nimbostratus':6,
               'Cumulus':7, 'Stratocumulus': 8, 'Stratus': 9,
               'Unknown':10}

for year in np.arange(year, year+1, 1):
    # year=2015
    print(f'#-------------------------------- {year}')
    for month in np.arange(month, month+1, 1):
        # month=7
        print(f'#---------------- {month:02d}')
        ymfolder = f'/scratch/v46/qg8515/data/obs/jaxa/clp/{year}{month:02d}'
        if os.path.isdir(ymfolder):
            for day in np.arange(1, calendar.monthrange(year, month)[1]+1, 1):
                # day=4
                print(f'#-------- {day:02d}')
                dfolder = f'{ymfolder}/{day:02d}'
                if os.path.isdir(dfolder):
                    fl = sorted(glob.glob(f'{dfolder}/??/CLTYPE_{year}{month:02d}{day:02d}????.nc'))
                    ds = xr.open_mfdataset(fl)
                    cltype_count = xr.DataArray(
                        name='cltype_count',
                        data=np.zeros((1, 12, len(ds.latitude), len(ds.longitude))),
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
year=2024
month=12
day=31
icloud='Stratocumulus'

ymfolder = f'/scratch/v46/qg8515/data/obs/jaxa/clp/{year}{month:02d}'
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

fl = sorted(glob.glob('/scratch/v46/qg8515/data/obs/jaxa/clp/??????/??/cltype_count_????????.nc'))
cltype_count = xr.open_mfdataset(fl).cltype_count

cltype_count_alltime = {}

cltype_count_alltime['daily'] = cltype_count

print('get mon')
cltype_count_alltime['mon'] = cltype_count.resample({'time': '1ME'}).sum().compute()

print('get sea')
cltype_count_alltime['sea'] = cltype_count_alltime['mon'].resample({'time': 'QE-FEB'}).sum()[1:-1].compute()

print('get ann')
cltype_count_alltime['ann'] = cltype_count_alltime['mon'].resample({'time': '1YE'}).sum()[1:].compute()

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
ofile='/scratch/v46/qg8515/data/obs/jaxa/clp/cltype_count_alltime.pkl'
if os.path.exists(ofile): os.remove(ofile)
with open(ofile, 'wb') as f:
    pickle.dump(cltype_count_alltime, f)


'''
#-------------------------------- check




'''
# endregion


# region get alltime frequency of each cloud type

with open('/scratch/v46/qg8515/data/obs/jaxa/clp/cltype_count_alltime.pkl', 'rb') as f:
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

ofile='/scratch/v46/qg8515/data/obs/jaxa/clp/cltype_frequency_alltime.pkl'
if os.path.exists(ofile): os.remove(ofile)
with open(ofile, 'wb') as f:
    pickle.dump(cltype_frequency_alltime, f)


'''
#-------------------------------- check


'''
# endregion
