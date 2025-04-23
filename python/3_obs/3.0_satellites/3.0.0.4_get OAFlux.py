

# qsub -I -q normal -l walltime=4:00:00,ncpus=1,mem=60GB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22


# region import packages

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
from scipy import stats
import pandas as pd
from metpy.interpolate import cross_section
from statsmodels.stats import multitest
from metpy.calc import pressure_to_height_std, geopotential_to_height
from metpy.units import units
import metpy.calc as mpcalc
import pickle
import xesmf as xe

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
from matplotlib import cm
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=10)
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})
import matplotlib.animation as animation
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.ticker import AutoMinorLocator
import geopandas as gpd
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import cartopy.feature as cfeature

# management
import os
import sys  # print(sys.path)
sys.path.append(os.getcwd() + '/code/gbr_future/module')
import psutil
process = psutil.Process()
# print(process.memory_info().rss / 2**30)
import string
import warnings
warnings.filterwarnings('ignore')
import glob
from datetime import datetime

# self defined
from mapplot import (
    globe_plot,
    regional_plot,
    ticks_labels,
    scale_bar,
    plot_maxmin_points,
    remove_trailing_zero,
    remove_trailing_zero_pos,
    )

from namelist import (
    month_jan,
    monthini,
    seasons,
    seconds_per_d,
    zerok,
    panel_labels,
    era5_varlabels,
    cmip6_era5_var,
    )

from component_plot import (
    rainbow_text,
    change_snsbar_width,
    cplot_wind_vectors,
    cplot_lon180,
    cplot_lon180_ctr,
    plt_mesh_pars,
)

from calculations import (
    mon_sea_ann,
    regrid,
    cdo_regrid,)

from statistics0 import (
    ttest_fdr_control,)

# endregion


# region get OAFlux monthly data

oaflux_cmip_var = {
    'evapr': 'evspsbl',
    'lh': 'hfls',
    'lw': 'rlns',
    'qa': 'huss',
    'qnet': 'rns',
    'sh': 'hfss',
    'sw': 'rsns',
    'ta': 'tas',
    'ts': 'sst',
    'ws': 'sfcWind'
}

def pre_process(var, year, ifile):
    import gzip
    import io
    import xarray as xr
    from datetime import datetime
    
    with gzip.open(ifile, 'rb') as f:
        with io.BytesIO(f.read()) as nc_file:
            ds = xr.open_dataset(nc_file).compute()
    
    ds = ds[[var for var in ds.data_vars][0]]
    ds['time'] = [datetime(year, int(month), 1) for month in ds['time'].values]
    ds = ds.rename(oaflux_cmip_var[var])
    if var in ['evapr']:
        ds = ds * 10 / 365
    elif var in ['lh', 'lw', 'sh']:
        ds = ds * (-1)
    
    return(ds)


for var in ['evapr', 'lh', 'lw', 'qa', 'qnet', 'sh', 'sw', 'ta', 'ts', 'ws']:
    # var = 'evapr'
    print(f'#-------------------------------- {var}')
    
    fl = sorted(glob.glob(f'/home/563/qg8515/data/obs/OAFlux/{var}*.nc.gz'))
    years = [int(ifile[-10:-6]) for ifile in fl]
    # print(pre_process(years[-1], fl[-1]))
    
    oaflux_mon = xr.merge([pre_process(var, year, ifile) for year, ifile in zip(years, fl)])
    oaflux_mon = oaflux_mon[oaflux_cmip_var[var]].sel(time=slice('1979', '2023'))
    oaflux_mon_alltime = mon_sea_ann(
        var_monthly=oaflux_mon, lcopy=True, mm=True, sm=True, am=True)
    
    ofile = f'/home/563/qg8515/data/obs/OAFlux/oaflux_mon_alltime_{oaflux_cmip_var[var]}.pkl'
    if os.path.exists(ofile): os.remove(ofile)
    with open(ofile, 'wb') as f:
        pickle.dump(oaflux_mon_alltime, f)
    del oaflux_mon, oaflux_mon_alltime





'''
#-------------------------------- check 2
import gzip
import io
oaflux_cmip_var = {
    'evapr': 'evspsbl',
    'lh': 'hfls',
    'lw': 'rlns',
    'qa': 'huss',
    'qnet': 'rns',
    'sh': 'hfss',
    'sw': 'rsns',
    'ta': 'tas',
    'ts': 'sst',
    'ws': 'sfcWind'
}
def pre_process(var, year, ifile):
    import gzip
    import io
    import xarray as xr
    from datetime import datetime
    
    with gzip.open(ifile, 'rb') as f:
        with io.BytesIO(f.read()) as nc_file:
            ds = xr.open_dataset(nc_file).compute()
    
    ds = ds[[var for var in ds.data_vars][0]]
    ds['time'] = [datetime(year, int(month), 1) for month in ds['time'].values]
    ds = ds.rename(oaflux_cmip_var[var])
    if var in ['evapr']:
        ds = ds * 10 / 365
    elif var in ['lh', 'lw', 'sh']:
        ds = ds * (-1)
    
    return(ds)

itime = -10

for var in ['evapr', 'lh', 'lw', 'qa', 'qnet', 'sh', 'sw', 'ta', 'ts', 'ws']:
    # var = 'evapr'
    print(f'#-------------------------------- {var} {oaflux_cmip_var[var]}')
    
    fl = sorted(glob.glob(f'data/obs/OAFlux/{var}*.nc.gz'))
    years = [int(ifile[-10:-6]) for ifile in fl]
    
    ofile = f'data/obs/OAFlux/oaflux_mon_alltime_{oaflux_cmip_var[var]}.pkl'
    with open(ofile, 'rb') as f:
        oaflux_mon_alltime = pickle.load(f)
    
    # check 1
    oaflux_mon = xr.merge([pre_process(var, year, ifile) for year, ifile in zip(years, fl)])
    oaflux_mon = oaflux_mon[oaflux_cmip_var[var]].sel(time=slice('1979', '2023'))
    
    data1 = oaflux_mon[itime].values.astype('float32')
    data2 = oaflux_mon_alltime['mon'][itime].values
    print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())
    
    # check 2
    with gzip.open(fl[itime], 'rb') as f:
        with io.BytesIO(f.read()) as nc_file:
            ds = xr.open_dataset(nc_file).compute()
    ds = ds[[var for var in ds.data_vars][0]]
    ds['time'] = [datetime(years[itime], int(month), 1) for month in ds['time'].values]
    ds = ds.rename(oaflux_cmip_var[var])
    if var in ['evapr']:
        ds = ds * 10 / 365
    elif var in ['lh', 'lw', 'sh']:
        ds = ds * (-1)
    
    data1 = ds.values.astype('float32')
    data2 = oaflux_mon_alltime['mon'].sel(time=ds.time).values
    print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())




#-------------------------------- check 1
import gzip
import io
for var in ['evapr', 'lh', 'lw', 'qa', 'qnet', 'sh', 'sw', 'ta', 'ts', 'ws']:
    # var = 'lh'
    print(f'#-------------------------------- {var}')
    
    fl = sorted(glob.glob(f'/home/563/qg8515/data/obs/OAFlux/{var}*.nc.gz'))
    years = [int(ifile[-10:-6]) for ifile in fl]
    
    with gzip.open(fl[-1], 'rb') as f:
        with io.BytesIO(f.read()) as nc_file:
            ds = xr.open_dataset(nc_file).compute()
    ds = ds[[var for var in ds.data_vars][0]]
    ds['time'] = [datetime(years[-1], int(month), 1) for month in ds['time'].values]
    ds = ds.rename(oaflux_cmip_var[var])
    if var in ['evapr']:
        ds = ds * 10 / 365
    elif var in ['lh', 'lw', 'sh']:
        ds = ds * (-1)
    print(ds)
'''
# endregion

