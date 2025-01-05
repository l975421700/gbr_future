

# region get command line options

import argparse
parser=argparse.ArgumentParser()
parser.add_argument('-y', '--year', type=str, required=True,)
parser.add_argument('-m', '--month', type=str, required=True,)
parser.add_argument('-d', '--day', type=str, required=True,)
args = parser.parse_args()

year=args.year
month=args.month
day=args.day

# endregion


# region import packages

import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
import glob
from datetime import datetime
import os
import gc

# endregion


# region import data

# year='2016'
# month='01'
# day='01'

himawari_fl = sorted(glob.glob(f'data/obs/jaxa/clp/{year}{month}/{day}/*/NC_*'))

for ifile in himawari_fl[:6]:
    hour = ifile[28:30]
    minute = ifile[49:51]
    time = datetime.strptime(f'{year}-{month}-{day} {hour}:{minute}', '%Y-%m-%d %H:%M')
    print(time)
    
    ds = xr.open_dataset(ifile)
    print('get ds_out')
    ds_out = ds.CLTYPE.copy().expand_dims(dim={'time': [time]}, axis=0).chunk({'latitude': 100, 'longitude': 100})
    ds_out.attrs.clear()
    
    print('get ofile')
    ofile = f'{ifile[:31]}CLTYPE_{year}{month}{day}{hour}{minute}.nc'
    if os.path.exists(ofile): os.remove(ofile)
    print('output file')
    ds_out.to_netcdf(ofile)
    print('done')
    
    del ds, ds_out
    gc.collect()




'''

import xarray as xr
import glob
aaa = xr.open_dataset('data/obs/jaxa/clp/201601/01/01/CLTYPE_201601010100.nc')
# , decode_times=True, use_cftime=True

aaa.CLTYPE
bbb = xr.open_mfdataset(sorted(glob.glob('data/obs/jaxa/clp/201601/01/01/CLTYPE_*')))

# check missing files
timestamps = pd.date_range(start='2016-01-01T00:00', end='2016-12-31T23:50', freq='10min')
len(himawari_fl)
len(timestamps)

    # year = ifile[18:22]
    # month = ifile[22:24]
    # day = ifile[25:27]

pip install xarray
pip install dask
'''
# endregion

