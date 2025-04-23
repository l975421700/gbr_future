

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
from datetime import datetime
from skimage.measure import block_reduce
from netCDF4 import Dataset
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
from PIL import Image
from matplotlib.colors import ListedColormap

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
import argparse
import calendar
from pathlib import Path
import time

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
    month_num,
    monthini,
    seasons,
    seconds_per_d,
    zerok,
    panel_labels,
    era5_varlabels,
    cmip6_era5_var,
    ds_color,
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
    time_weighted_mean,
    mon_sea_ann,
    regrid,
    cdo_regrid,)

from statistics0 import (
    ttest_fdr_control,)

# endregion


# region plot am global OAFlux and ERA5

for var in ['evspsbl', 'hfls', 'rlns', 'rns', 'hfss', 'rsns', 'tas', 'sst', 'sfcWind']:
    # 'huss'
    # var = 'sfcWind'
    print(f'#-------------------------------- {var} {cmip6_era5_var[var]}')
    
    with open(f'data/obs/era5/mon/era5_sl_mon_alltime_{cmip6_era5_var[var]}.pkl','rb') as f:
        era5_sl_mon_alltime = pickle.load(f)
    # print(era5_sl_mon_alltime['ann'])
    with open(f'data/obs/OAFlux/oaflux_mon_alltime_{var}.pkl', 'rb') as f:
        oaflux_mon_alltime = pickle.load(f)
    # print(oaflux_mon_alltime['ann'].time)
    
    years = str(oaflux_mon_alltime['ann'].time[0].dt.year.values)
    yeare = str(oaflux_mon_alltime['ann'].time[-1].dt.year.values)
    
    print(stats.describe(np.concat((oaflux_mon_alltime['ann'].values.flatten(), era5_sl_mon_alltime['ann'].sel(time=slice(years, yeare)).values.flatten())), axis=None, nan_policy='omit'))
    
    if var in ['evspsbl']:
        pltlevel = np.array([-0.1, 0, 0.1, 0.2, 0.5, 1, 2, 4, 6, 8, 10])
        pltticks = np.array([-0.1, 0, 0.1, 0.2, 0.5, 1, 2, 4, 6, 8, 10])
        pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
        pltcmp = plt.get_cmap('viridis', len(pltlevel)-1)
        pltlevel2 = np.array([-2, -1.5, -1, -0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5, 1, 1.5, 2])
        pltticks2 = np.array([-2, -1.5, -1, -0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5, 1, 1.5, 2])
        pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
        pltcmp2 = plt.get_cmap('BrBG', len(pltlevel2)-1)
    elif var in ['hfls']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-280, cm_max=0, cm_interval1=20, cm_interval2=40, cmap='viridis',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10, cmap='BrBG')
    elif var in ['rlns']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-170, cm_max=0, cm_interval1=10, cm_interval2=20, cmap='viridis',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10, cmap='BrBG',)
    elif var in ['rns']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-160, cm_max=160, cm_interval1=20, cm_interval2=40, cmap='BrBG',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-80, cm_max=80, cm_interval1=10, cm_interval2=20, cmap='BrBG',)
    elif var in ['hfss']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-120, cm_max=60, cm_interval1=10, cm_interval2=20, cmap='BrBG', asymmetric=True,)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-20, cm_max=20, cm_interval1=2.5, cm_interval2=5, cmap='BrBG',)
    elif var in ['rsns']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=300, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10, cmap='BrBG',)
    elif var in ['tas']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-32, cm_max=32, cm_interval1=2, cm_interval2=8, cmap='BrBG')
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-4, cm_max=4, cm_interval1=0.5, cm_interval2=1, cmap='BrBG')
    elif var in ['sst']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-2, cm_max=30, cm_interval1=2, cm_interval2=4, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-1, cm_max=1, cm_interval1=0.2, cm_interval2=0.4, cmap='BrBG', asymmetric=True)
    elif var in ['sfcWind']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=2, cm_max=12, cm_interval1=1, cm_interval2=1, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-2, cm_max=2, cm_interval1=0.25, cm_interval2=0.5, cmap='BrBG')
    
    opng = f'figures/5_era5/5.1_era5_obs/5.1.0_oaflux vs. era5 {var} {years}_{yeare}.png'
    cbar_label1 = f'Annual mean ({years}-{yeare}) {era5_varlabels[cmip6_era5_var[var]]}'
    cbar_label2 = f'Difference in annual mean ({years}-{yeare}) {era5_varlabels[cmip6_era5_var[var]]}'
    plt_colnames = ['OAFlux', 'ERA5', 'ERA5-OAFlux']
    
    nrow = 1
    ncol = 3
    fm_bottom = 1.5 / (4.4*nrow + 2)
    
    fig, axs = plt.subplots(
        nrow, ncol, figsize=np.array([8.8*ncol, 4.4*nrow + 2]) / 2.54,
        subplot_kw={'projection': ccrs.Mollweide(central_longitude=180)},
        gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)
    
    for jcol in range(ncol):
        axs[jcol] = globe_plot(ax_org=axs[jcol])
        axs[jcol].text(
            0.5, 1.02, f'({string.ascii_lowercase[jcol]}) {plt_colnames[jcol]}',
            ha='center', va='bottom', transform=axs[jcol].transAxes)
        # axs[jcol].add_feature(cfeature.LAND,color='white',zorder=2,edgecolor=None,lw=0)
    
    plt_mesh = axs[0].pcolormesh(
        oaflux_mon_alltime['ann'].lon, oaflux_mon_alltime['ann'].lat,
        oaflux_mon_alltime['ann'].mean(dim='time'),
        norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree())
    plt_mesh = axs[1].pcolormesh(
        era5_sl_mon_alltime['ann'].lon, era5_sl_mon_alltime['ann'].lat,
        era5_sl_mon_alltime['ann'].sel(time=slice(years, yeare)).mean(dim='time'),
        norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree())
    
    if not 'regridder' in globals():
        regridder = xe.Regridder(era5_sl_mon_alltime['ann'], oaflux_mon_alltime['ann'], 'bilinear')
    era5_ann_regrid = regridder(era5_sl_mon_alltime['ann'].sel(time=slice(years, yeare)))
    plt_data = era5_ann_regrid.mean(dim='time') - oaflux_mon_alltime['ann'].mean(dim='time')
    ttest_fdr_res = ttest_fdr_control(era5_ann_regrid, oaflux_mon_alltime['ann'])
    plt_data = plt_data.where(ttest_fdr_res, np.nan)
    plt_mesh2 = axs[2].pcolormesh(
        plt_data.lon, plt_data.lat, plt_data,
        norm=pltnorm2, cmap=pltcmp2, transform=ccrs.PlateCarree())
    
    cbar = fig.colorbar(
        plt_mesh, #cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), #
        ax=axs, format=remove_trailing_zero_pos,
        orientation="horizontal", ticks=pltticks, extend='both',
        cax=fig.add_axes([0.05, fm_bottom-0.05, 0.4, 0.04]))
    cbar.ax.set_xlabel(cbar_label1)
    cbar2 = fig.colorbar(
        plt_mesh2, #cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), #
        ax=axs, format=remove_trailing_zero_pos,
        orientation="horizontal", ticks=pltticks2, extend='both',
        cax=fig.add_axes([0.55, fm_bottom-0.05, 0.4, 0.04]))
    cbar2.ax.set_xlabel(cbar_label2)
    
    fig.subplots_adjust(left=0.01, right=0.99, bottom=fm_bottom, top=0.92)
    fig.savefig(opng)
    del era5_sl_mon_alltime, oaflux_mon_alltime



'''
print(stats.describe(oaflux_mon_alltime['ann'].values, axis=None, nan_policy='omit'))
print(stats.describe(era5_sl_mon_alltime['ann'].sel(time=slice(years, yeare)).values.flatten(), axis=None, nan_policy='omit'))
rlns, rns, rsns: 1983-2009

'''
# endregion


# region plot am domain OAFlux vs. ERA5, BARRA-R2, BARRA-C2


# endregion




# region plot surface energy imblance


dss = ['ERA5', 'BARRA-R2', 'BARRA-C2']
extent = [110.58, 157.34, -43.69, -7.01]
min_lon, max_lon, min_lat, max_lat = extent
pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-160, cm_max=160, cm_interval1=20, cm_interval2=40, cmap='BrBG',)
nrow=1
ncol=len(dss)


surface_radiation = {}
for ids in dss: surface_radiation[ids] = {}
for var2 in ['rsus', 'rlus', 'rsds', 'rlds', 'hfls', 'hfss']:
    # var2='rsus'
    var1 = cmip6_era5_var[var2]
    print(f'#-------------------------------- {var1} and {var2}')
    
    with open(f'data/obs/era5/mon/era5_sl_mon_alltime_{var1}.pkl', 'rb') as f:
        surface_radiation['ERA5'][var2] = pickle.load(f)['am'].squeeze().sel(lon=slice(min_lon, max_lon), lat=slice(max_lat, min_lat))
    with open(f'data/sim/um/barra_r2/barra_r2_mon_alltime_{var2}.pkl','rb') as f:
        surface_radiation['BARRA-R2'][var2] = pickle.load(f)['am'].squeeze().sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))
    with open(f'data/sim/um/barra_c2/barra_c2_mon_alltime_{var2}.pkl','rb') as f:
        surface_radiation['BARRA-C2'][var2] = pickle.load(f)['am'].squeeze().sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))

surface_radiation_imbalance = {}
for ids in dss:
    surface_radiation_imbalance[ids] = sum(surface_radiation[ids][var] for var in surface_radiation[ids].keys())


fm_bottom=1.5/(4*nrow+1.5)
fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([4.4*ncol, 4*nrow + 1.5]) / 2.54,
    subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
    gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)

for jcol in range(ncol):
    axs[jcol] = regional_plot(extent=extent, central_longitude=180, ax_org=axs[jcol])
    plt_mean = surface_radiation_imbalance[dss[jcol]].weighted(np.cos(np.deg2rad(surface_radiation_imbalance[dss[jcol]].lat))).mean().values
    if jcol==0:
        plt_text = f'({string.ascii_lowercase[jcol]}) {dss[jcol]}, Mean: {np.round(plt_mean, 1)}'
    else:
        plt_text = f'({string.ascii_lowercase[jcol]}) {dss[jcol]}, {np.round(plt_mean, 1)}'
    axs[jcol].text(0, 1.02, plt_text, ha='left', va='bottom', transform=axs[jcol].transAxes, size=9)

for jcol, ids in enumerate(dss):
    plt_mesh = axs[jcol].pcolormesh(
        surface_radiation_imbalance[ids].lon,
        surface_radiation_imbalance[ids].lat,
        surface_radiation_imbalance[ids].values,
        norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),zorder=1)

cbar = fig.colorbar(
    plt_mesh, #cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), #
    format=remove_trailing_zero_pos,
    orientation="horizontal", ticks=pltticks, extend='both',
    cax=fig.add_axes([0.25, fm_bottom-0.05, 0.5, 0.05]))
cbar.ax.set_xlabel(r'Surface radiation excess [$W \; m^{-2}$]')

fig.subplots_adjust(left=0.005, right=0.995, bottom=fm_bottom, top=0.95)
fig.savefig(f'figures/4_um/4.0_barra/4.0.3_surface_radiation/4.0.3.0 ERA5, BARRA-R2, C2 surface radiation excess.png')


'''
print(stats.describe(era5_am_sei.squeeze().sel(lon=slice(min_lon, max_lon), lat=slice(max_lat, min_lat)).values, axis=None, nan_policy='omit'))
print(stats.describe(barra_r2_am_sei.squeeze().sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat)).values, axis=None, nan_policy='omit'))
print(stats.describe(barra_c2_am_sei.squeeze().sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat)).values, axis=None, nan_policy='omit'))


era5_am_sei.to_netcdf('data/others/test/test1.nc')
barra_r2_am_sei.to_netcdf('data/others/test/test.nc')
barra_c2_am_sei.to_netcdf('data/others/test/test2.nc')
'''
# endregion


