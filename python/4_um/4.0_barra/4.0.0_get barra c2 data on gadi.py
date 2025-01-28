

# qsub -I -q normal -l walltime=10:00:00,ncpus=1,mem=192GB,storage=gdata/v46+gdata/ob53


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

from calculations import (
    mon_sea_ann,
    )

# endregion


# region get BARRA-C2 mm data

for var in ['pr']:
    # var = 'clh'
    # 'pr', 'clh', 'clm', 'cll', 'clt', 'evspsbl', 'hfls', 'hfss', 'psl', 'rlds', 'rldscs', 'rlus', 'rluscs', 'rlut', 'rlutcs', 'rsds', 'rsdscs', 'rsdt', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'sfcWind', 'tas', 'ts',
    print(var)
    
    fl = sorted(glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/mon/{var}/latest/*'))[:540]
    
    barra_c2_mon = xr.open_mfdataset(fl)
    barra_c2_mon_alltime = mon_sea_ann(
        var_monthly=barra_c2_mon[var], lcopy=False, mm=True, sm=True, am=True,)
    
    with open(f'data/sim/um/barra_c2/barra_c2_mon_alltime_{var}.pkl','wb') as f:
        pickle.dump(barra_c2_mon_alltime, f)
    
    del barra_c2_mon, barra_c2_mon_alltime


'''
# check
barra_c2_mon_alltime = {}
for var in ['pr', 'clh', 'clm', 'cll', 'clt', 'evspsbl', 'hfls', 'hfss', 'psl', 'rlds', 'rldscs', 'rlus', 'rluscs', 'rlut', 'rlutcs', 'rsds', 'rsdscs', 'rsdt', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'sfcWind', 'tas', 'ts',]:
    print(var)
    
    with open(f'data/sim/um/barra_c2/barra_c2_mon_alltime_{var}.pkl','rb') as f:
        barra_c2_mon_alltime[var] = pickle.load(f)
    
    print(barra_c2_mon_alltime[var]['mon'].shape)
    del barra_c2_mon_alltime[var]




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


