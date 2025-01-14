

# qsub -I -q normal -l walltime=08:00:00,ncpus=1,mem=192GB,storage=gdata/v46+gdata/rt52


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
import glob

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
import cartopy.crs as ccrs
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=10)
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})
import matplotlib.animation as animation
import seaborn as sns
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
    month,
    monthini,
    seasons,
    seconds_per_d,
    zerok,
    panel_labels,
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
    cdo_regrid,)

# endregion


# region get data

year=2016

era5_cc = {}
era5_ccf_mon = {}
era5_ccf_ann = {}

for cc in ['hcc', 'mcc', 'lcc', 'tcc']:
    # cc='hcc'
    print(cc)
    
    era5_cc[cc] = xr.open_mfdataset(sorted(glob.glob(f'/g/data/rt52/era5/single-levels/reanalysis/{cc}/{year}/*')))[cc]
    
    era5_ccf_mon[cc] = era5_cc[cc].groupby('time.month').mean(dim='time').compute()
    era5_ccf_ann[cc] = era5_cc[cc].mean(dim='time').compute()
    
    era5_ccf_mon[cc].to_netcdf(f'data/obs/era5/hourly/era5_{cc}f_{year}_mon.nc')
    era5_ccf_ann[cc].to_netcdf(f'data/obs/era5/hourly/era5_{cc}f_{year}_ann.nc')
    
    del era5_cc[cc], era5_ccf_mon[cc], era5_ccf_ann[cc]


'''
# check
cc='hcc'
year=2016

era5_cc = {}
era5_ccf_mon = {}
era5_ccf_ann = {}

era5_cc[cc] = xr.open_mfdataset(sorted(glob.glob(f'/g/data/rt52/era5/single-levels/reanalysis/{cc}/{year}/*')))[cc]
era5_ccf_mon[cc] = xr.open_dataset(f'data/obs/era5/hourly/era5_{cc}f_{year}_mon.nc')[cc]
era5_ccf_ann[cc] = xr.open_dataset(f'data/obs/era5/hourly/era5_{cc}f_{year}_ann.nc')[cc]

era5_cc[cc][:, 0, 0]
np.max(np.abs(era5_cc[cc][:, 0, 0].groupby('time.month').mean(dim='time').values - era5_ccf_mon[cc][:, 0, 0].values))
era5_cc[cc][:, 0, 0].mean(dim='time').values == era5_ccf_ann[cc][0, 0].values


'''
# endregion


# region plot HML ccf

year=2016

opng = f'figures/3_satellites/3.0_hamawari_cl/3.0.0_era5_ccf_{year}_HML.png'
era5_ccf_ann = {}

nrow = 1
ncol = 3
fm_bottom = 1.6 / (6.6*nrow + 2)

cloudgroups = ['High cloud', 'Medium cloud', 'Low cloud', ]
cbar_label = r'Cloud occurrence frequency in 2016 from ERA5 [$\%$]'
pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=80, cm_interval1=5, cm_interval2=10, cmap='Blues_r',)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([6.6*ncol, 6.6*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
    gridspec_kw={'hspace': 0.1, 'wspace': 0.02},)

for jcol, cc in zip(range(ncol), ['hcc', 'mcc', 'lcc']):
    print(f'{panel_labels[jcol]} {cloudgroups[jcol]} {cc}')
    axs[jcol] = regional_plot(
        extent=[80, 200, -60, 60], central_longitude=180, ax_org=axs[jcol],)
    plt.text(
        0, 1.02, f'{panel_labels[jcol]} {cloudgroups[jcol]}',
        transform=axs[jcol].transAxes, fontsize=10,
        ha='left', va='bottom', rotation='horizontal')
    
    era5_ccf_ann[cc] = xr.open_dataset(f'data/obs/era5/hourly/era5_{cc}f_{year}_ann.nc')[cc]
    plt_mesh = axs[jcol].pcolormesh(
        era5_ccf_ann[cc].longitude, era5_ccf_ann[cc].latitude,
        era5_ccf_ann[cc] * 100,
        norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_mesh, # cm.ScalarMappable(norm=pltnorm, cmap=pltcmp),
    ax=axs, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.5, ticks=pltticks, extend='max',
    anchor=(0.5, 0.2),)
cbar.ax.set_xlabel(cbar_label)
fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom, top = 0.99)
fig.savefig(opng)


# endregion


# region plot T ccf

opng = f'figures/3_satellites/3.0_hamawari_cl/3.0.0_era5_ccf_{year}_T.png'

cbar_label = 'Total cloud occurrence frequency\nin 2016 from ERA5 [%]'
pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=10, cmap='Blues_r',)

fig, ax = regional_plot(extent=[80, 200, -60, 60], central_longitude=180, figsize = np.array([6.6, 8.6]) / 2.54)

era5_ccf_ann = {}
cc = 'tcc'
year=2016
era5_ccf_ann[cc] = xr.open_dataset(f'data/obs/era5/hourly/era5_{cc}f_{year}_ann.nc')[cc]
plt_mesh = ax.pcolormesh(
    era5_ccf_ann[cc].longitude, era5_ccf_ann[cc].latitude, era5_ccf_ann[cc]*100,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp),
    ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='neither',
    pad=0.03, fraction=0.12,)
cbar.ax.set_xlabel(cbar_label, linespacing=1.5)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = 0.12, top = 0.99)
fig.savefig(opng)


# endregion


