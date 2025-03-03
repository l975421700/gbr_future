

# qsub -I -q normal -l walltime=10:00:00,ncpus=1,mem=12GB,jobfs=100MB,storage=gdata/v46+scratch/v46


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

# endregion


# region get daily count of each cloud type

ISCCP_types = {'Clear': 0,
               'Cirrus': 1, 'Cirrostratus': 2, 'Deep convection': 3,
               'Altocumulus': 4, 'Altostratus': 5, 'Nimbostratus':6,
               'Cumulus':7, 'Stratocumulus': 8, 'Stratus': 9,
               'Unknown':10}

for year in np.arange(2022, 2022+1, 1):
    # year=2015
    print(f'#-------------------------------- {year}')
    for month in np.arange(1, 12+1, 1):
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
                    cltype_count.loc[{'types': 'finite'}][0] = np.isfinite(ds.CLTYPE.values).sum(axis=0)
                    for itype in list(ISCCP_types.keys()):
                        print(f'#---- {itype}')
                        cltype_count.loc[{'types': itype}][0] = (ds.CLTYPE.values == ISCCP_types[itype]).sum(axis=0)
                    check = (cltype_count[0, 0] == cltype_count[0, 1:].sum(axis=0)).all().values
                    if not check:
                        print('Warning 3: Sum of all cloud types != finite counts')
                    ofile = f'{dfolder}/cltype_count_{year}{month:02d}{day:02d}.nc'
                    if os.path.exists(ofile): os.remove(ofile)
                    cltype_count.to_netcdf(ofile)
                else:
                    print(f'Warning 2: No folder for year {year} month {month:02d} day {day:02d}')
        else:
            print(f'Warning 1: No folder for year {year} month {month:02d}')


# endregion


