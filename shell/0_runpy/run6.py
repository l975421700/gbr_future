

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


# region get BARRA-C2 mon pl data

for var in ['wap']:
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

