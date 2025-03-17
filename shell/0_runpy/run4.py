

# region import packages

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()

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


# region get BARRA-C2 hourly data

for var in ['clt']:
    # var = 'cll'
    # ['clh', 'clm', 'cll', 'clt', 'pr', 'tas']
    print(f'#-------------------------------- {var}')
    
    fl = sorted(glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/{var}/latest/*'))[:540]
    
    barra_c2_hourly = xr.open_mfdataset(fl, parallel=True)[var] #.sel(time=slice('1979', '2023'))
    if var in ['pr', 'evspsbl', 'evspsblpot']:
        barra_c2_hourly = barra_c2_hourly * seconds_per_d
    elif var in ['tas', 'ts']:
        barra_c2_hourly = barra_c2_hourly - zerok
    elif var in ['rlus', 'rluscs', 'rlut', 'rlutcs', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'hfls', 'hfss']:
        barra_c2_hourly = barra_c2_hourly * (-1)
    elif var in ['psl']:
        barra_c2_hourly = barra_c2_hourly / 100
    elif var in ['huss']:
        barra_c2_hourly = barra_c2_hourly * 1000
    
    ofile = f'data/sim/um/barra_c2/barra_c2_hourly_{var}.pkl'
    if os.path.exists(ofile): os.remove(ofile)
    with open(ofile,'wb') as f:
        pickle.dump(barra_c2_hourly, f)
    
    del barra_c2_hourly






# endregion
