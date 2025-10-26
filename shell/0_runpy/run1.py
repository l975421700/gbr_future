

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
import argparse
from metpy.calc import specific_humidity_from_dewpoint, relative_humidity_from_dewpoint, vertical_velocity_pressure, mixing_ratio_from_specific_humidity, relative_humidity_from_specific_humidity
from metpy.units import units
from metpy.constants import water_heat_vaporization, dry_air_spec_heat_press
import time
import joblib

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

from namelist import cmip6_units, zerok, seconds_per_d

# endregion


# region get BARRA-R2 mon pl data
# Memory Used: 68.42GB; Walltime Used: 01:06:43

for var in ['hus']:
    # var = 'ua'
    # ['hus', 'ta', 'ua', 'va', 'wa', 'zg']
    print(var)
    
    fl = sorted([
        file for iyear in np.arange(2016, 2024, 1)
        for file in glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/mon/{var}[0-9]*[!m]/latest/*BARRA-R2_v1_mon_{iyear}*')])
    
    def std_func(ds_in, var=var):
        ds = ds_in.expand_dims(dim='pressure', axis=1)
        varname = [varname for varname in ds.data_vars if varname.startswith(var)][0]
        ds = ds.rename({varname: var})
        ds = ds.chunk(chunks={'time': 1, 'pressure': 1, 'lat': len(ds.lat), 'lon': len(ds.lon)})
        ds = ds.astype('float32')
        if var == 'hus':
            ds = ds * 1000
        elif var == 'ta':
            ds = ds - zerok
        return(ds)
    
    barra_r2_pl_mon = xr.open_mfdataset(fl, parallel=True, preprocess=std_func)[var]
    barra_r2_pl_mon_alltime = mon_sea_ann(
        var_monthly=barra_r2_pl_mon, lcopy=False,mm=True,sm=True,am=True)
    
    ofile = f'data/sim/um/barra_r2/barra_r2_pl_mon_alltime_{var}.pkl'
    if os.path.exists(ofile): os.remove(ofile)
    with open(ofile,'wb') as f:
        pickle.dump(barra_r2_pl_mon_alltime, f)
    
    del barra_r2_pl_mon, barra_r2_pl_mon_alltime





'''
#-------------------------------- check
ipressure = 850
itime = -10

barra_r2_pl_mon_alltime = {}
for var in ['ta', 'ua', 'va', 'wa', 'zg']:
    # var = 'hus'
    print(f'#-------------------------------- {var}')
    
    with open(f'data/sim/um/barra_r2/barra_r2_pl_mon_alltime_{var}.pkl','rb') as f:
        barra_r2_pl_mon_alltime[var] = pickle.load(f)
    
    fl = sorted([
        file for iyear in np.arange(1979, 2024, 1)
        for file in glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/mon/{var}{str(ipressure)}/latest/*BARRA-R2_v1_mon_{iyear}*')])
    
    ds = xr.open_dataset(fl[itime])[f'{var}{str(ipressure)}'].squeeze()
    ds = ds.astype('float32')
    if var == 'hus':
        ds = ds * 1000
    elif var == 'ta':
        ds = ds - zerok
    
    ds2 = barra_r2_pl_mon_alltime[var]['mon'][itime].sel(pressure=ipressure)
    
    print((ds.values[np.isfinite(ds.values)].astype(np.float32) == ds2.values[np.isfinite(ds2.values)]).all())
    del barra_r2_pl_mon_alltime[var]




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
