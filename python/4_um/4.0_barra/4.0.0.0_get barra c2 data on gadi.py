

# qsub -I -q normal -l walltime=4:00:00,ncpus=1,mem=192GB,storage=gdata/v46+gdata/ob53


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

# endregion


# region get BARRA-C2 mon data

for var in ['pr']:
    # var = 'clh'
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
for var in ['evspsblpot', 'hurs', 'huss', 'uas', 'vas']:
    # 'pr', 'clh', 'clm', 'cll', 'clt', 'evspsbl', 'hfls', 'hfss', 'psl', 'rlds', 'rldscs', 'rlus', 'rluscs', 'rlut', 'rlutcs', 'rsds', 'rsdscs', 'rsdt', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'sfcWind', 'tas', 'ts'
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


# region get BARRA-C2 mon pl data

for var in ['hus', 'ta', 'ua', 'va', 'wa', 'wap', 'zg']:
    # var = 'ta'
    print(var)
    
    fl = sorted([
        file for iyear in np.arange(1979, 2024, 1)
        for file in glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/mon/{var}[0-9]*[!m]/latest/*BARRA-C2_v1_mon_{iyear}*')])
    
    def std_func(ds, var=var):
        ds = ds.expand_dims(dim='pressure', axis=1)
        varname = [varname for varname in ds.data_vars if varname.startswith(var)][0]
        ds = ds.rename({varname: var})
        return(ds)
    
    barra_c2_pl_mon = xr.open_mfdataset(fl, parallel=True, preprocess=std_func)
    barra_c2_pl_mon_alltime = mon_sea_ann(
        var_monthly=barra_c2_pl_mon[var], lcopy=False,mm=True,sm=True,am=True)
    
    with open(f'data/sim/um/barra_c2/barra_c2_pl_mon_alltime_{var}.pkl','wb') as f:
        pickle.dump(barra_c2_pl_mon_alltime, f)
    
    del barra_c2_pl_mon, barra_c2_pl_mon_alltime





'''
# check
barra_c2_pl_mon_alltime = {}
for var in ['hus', 'ta', 'ua', 'va', 'wa', 'wap', 'zg']:
    # var = 'ta'
    print(f'#-------------------------------- {var}')
    
    with open(f'data/sim/um/barra_c2/barra_c2_pl_mon_alltime_{var}.pkl','rb') as f:
        barra_c2_pl_mon_alltime[var] = pickle.load(f)
    
    print(barra_c2_pl_mon_alltime[var]['mon'])
    
    del barra_c2_pl_mon_alltime[var]


# 3 hours each


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


# region derive BARRA-C2 mon data


barra_c2_mon_alltime = {}
for var1, var2, var3 in zip(['rlutcl', 'rsntcl', 'rsutcl'], ['rlut', 'rsnt', 'rsut'], ['rlutcs', 'rsntcs', 'rsutcs']):
    # ['rsntcs'], ['rsdt'], ['rsutcs']
    # ['rsnt'], ['rsdt'], ['rsut']
    # ['rluscl', 'rsuscl'], ['rlus', 'rsus'], ['rluscs', 'rsuscs']
    # ['rlnscl', 'rsnscl', 'rldscl', 'rsdscl'], ['rlns', 'rsns', 'rlds', 'rsds'], ['rlnscs', 'rsnscs', 'rldscs', 'rsdscs']
    # ['rlnscs', 'rsnscs'], ['rldscs', 'rsdscs'], ['rluscs', 'rsuscs']
    # ['rsns'], ['rsds'], ['rsus']
    # ['rlns'], ['rlds'], ['rlus']
    print(f'Derive {var1} from {var2} and {var3}')
    print(str(datetime.datetime.now())[11:19])
    
    with open(f'data/sim/um/barra_c2/barra_c2_mon_alltime_{var2}.pkl','rb') as f:
        barra_c2_mon_alltime[var2] = pickle.load(f)
    with open(f'data/sim/um/barra_c2/barra_c2_mon_alltime_{var3}.pkl','rb') as f:
        barra_c2_mon_alltime[var3] = pickle.load(f)
    
    if var1 in ['rlns', 'rsns', 'rlnscs', 'rsnscs', 'rlnscl', 'rsnscl', 'rldscl', 'rsdscl', 'rluscl', 'rsuscl', 'rsnt', 'rsntcs', 'rlutcl', 'rsntcl', 'rsutcl']:
        print('var2 - var3')
        barra_c2_mon = (barra_c2_mon_alltime[var2]['mon'] - barra_c2_mon_alltime[var3]['mon']).rename(var1)
    
    barra_c2_mon_alltime[var1] = mon_sea_ann(
        var_monthly=barra_c2_mon, lcopy=False, mm=True, sm=True, am=True,)
    
    with open(f'data/sim/um/barra_c2/barra_c2_mon_alltime_{var1}.pkl','wb') as f:
        pickle.dump(barra_c2_mon_alltime[var1], f)
    
    del barra_c2_mon_alltime[var2], barra_c2_mon_alltime[var3], barra_c2_mon_alltime[var1], barra_c2_mon
    print(str(datetime.datetime.now())[11:19])




'''
# check
barra_c2_mon_alltime = {}
for var in ['rlns', 'rsns', 'rlnscs', 'rsnscs',
            'rlnscl', 'rsnscl', 'rldscl', 'rsdscl', 'rluscl', 'rsuscl',
            'rsnt',
            'rsntcs',
            'rlutcl', 'rsntcl', 'rsutcl']:
    print(var)
    
    with open(f'data/sim/um/barra_c2/barra_c2_mon_alltime_{var}.pkl','rb') as f:
        barra_c2_mon_alltime[var] = pickle.load(f)
    
    print(barra_c2_mon_alltime[var]['mon'].shape)
    del barra_c2_mon_alltime[var]

# check
barra_c2_mon_alltime = {}
with open(f'data/sim/um/barra_c2/barra_c2_mon_alltime_rsntcl.pkl','rb') as f:
    barra_c2_mon_alltime['rsntcl'] = pickle.load(f)
with open(f'data/sim/um/barra_c2/barra_c2_mon_alltime_rsutcl.pkl','rb') as f:
    barra_c2_mon_alltime['rsutcl'] = pickle.load(f)

print(np.max(np.abs(barra_c2_mon_alltime['rsntcl']['am'].values + barra_c2_mon_alltime['rsutcl']['am'].values)))

del barra_c2_mon_alltime['rsntcl'], barra_c2_mon_alltime['rsutcl']

'''
# endregion


