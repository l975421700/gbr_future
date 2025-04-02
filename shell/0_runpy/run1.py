

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
import glob
from datetime import datetime
import os

# endregion


# region get hourly count of each cloud type

ISCCP_types = {'Clear': 0,
               'Cirrus': 1, 'Cirrostratus': 2, 'Deep convection': 3,
               'Altocumulus': 4, 'Altostratus': 5, 'Nimbostratus':6,
               'Cumulus':7, 'Stratocumulus': 8, 'Stratus': 9,
               'Unknown':10}

# year=2015; month=7
print(f'#-------------------------------- {year} {month}')

ymfolder = f'/scratch/v46/qg8515/data/obs/jaxa/clp/{year}{month:02d}'
fl = sorted(glob.glob(f'{ymfolder}/??/??/CLTYPE_{year}{month:02d}??????.nc'))
if (len(fl) != 0):
    print(f'Number of Files: {len(fl)}')
    
    ds = xr.open_mfdataset(fl, parallel=True).CLTYPE
    cltype_hourly_count = xr.DataArray(
        name='cltype_hourly_count',
        data=np.zeros((1, 24, 12, len(ds.latitude), len(ds.longitude))),
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





# endregion

