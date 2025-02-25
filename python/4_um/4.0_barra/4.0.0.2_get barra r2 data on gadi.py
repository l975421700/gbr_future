

# qsub -I -q normal -l walltime=4:00:00,ncpus=1,mem=96GB,storage=gdata/v46+gdata/ob53


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

for var in ['rsut', 'pr']:
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
for var in ['clh', 'clm', 'cll', 'clt', 'evspsbl', 'hfls', 'hfss', 'psl', 'rlds', 'rldscs', 'rlus', 'rluscs', 'rlut', 'rlutcs', 'rsds', 'rsdscs', 'rsdt', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'sfcWind', 'tas', 'ts', 'evspsblpot', 'hurs', 'huss', 'uas', 'vas']:
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


