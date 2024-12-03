

# qsub -I -q normal -l walltime=10:00:00,ncpus=1,mem=192GB,jobfs=192GB,storage=gdata/v46


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


# region import data

with open('data/sim/cmip6/historical_Omon_tos_rgd_alltime.pkl', 'rb') as f:
    historical_Omon_tos_rgd_alltime = pickle.load(f)
with open('data/sim/cmip6/historical_Amon_tas_rgd_alltime.pkl', 'rb') as f:
    historical_Amon_tas_rgd_alltime = pickle.load(f)
with open('data/sim/cmip6/historical_Amon_pr_rgd_alltime.pkl', 'rb') as f:
    historical_Amon_pr_rgd_alltime = pickle.load(f)

# with open('data/sim/cmip6/ssp585_Omon_tos_rgd_alltime.pkl', 'rb') as f:
#     ssp585_Omon_tos_rgd_alltime = pickle.load(f)
# with open('data/sim/cmip6/ssp585_Amon_tas_rgd_alltime.pkl', 'rb') as f:
#     ssp585_Amon_tas_rgd_alltime = pickle.load(f)
# with open('data/sim/cmip6/ssp585_Amon_pr_rgd_alltime.pkl', 'rb') as f:
#     ssp585_Amon_pr_rgd_alltime = pickle.load(f)

with open('data/sim/cmip6/ssp126_Omon_tos_rgd_alltime.pkl', 'rb') as f:
    ssp126_Omon_tos_rgd_alltime = pickle.load(f)
with open('data/sim/cmip6/ssp126_Amon_tas_rgd_alltime.pkl', 'rb') as f:
    ssp126_Amon_tas_rgd_alltime = pickle.load(f)
with open('data/sim/cmip6/ssp126_Amon_pr_rgd_alltime.pkl', 'rb') as f:
    ssp126_Amon_pr_rgd_alltime = pickle.load(f)

with open('data/obs/era5/mon/era5_mon_sst_rgd_alltime.pkl', 'rb') as f: era5_mon_sst_rgd_alltime = pickle.load(f)
with open('data/obs/era5/mon/era5_mon_t2m_rgd_alltime.pkl', 'rb') as f: era5_mon_t2m_rgd_alltime = pickle.load(f)
with open('data/obs/era5/mon/era5_mon_tp_rgd_alltime.pkl', 'rb') as f: era5_mon_tp_rgd_alltime = pickle.load(f)

lon = era5_mon_sst_rgd_alltime['ann'].lon.values
lat = era5_mon_sst_rgd_alltime['ann'].lat.values

common_models = sorted(list(set(historical_Omon_tos_rgd_alltime.keys()) & set(historical_Amon_tas_rgd_alltime.keys()) & set(historical_Amon_pr_rgd_alltime.keys())))


'''
print(process.memory_info().rss / 2**30)

print(len(historical_Omon_tos_rgd_alltime.keys()))
print(len(historical_Amon_tas_rgd_alltime.keys()))
print(len(historical_Amon_pr_rgd_alltime.keys()))
print(len(common_models))

# check
with open('data/sim/cmip6/historical_Omon_tos.pkl', 'rb') as f:
    historical_Omon_tos = pickle.load(f)
with open('data/sim/cmip6/historical_Omon_tos_rgd.pkl', 'rb') as f:
    historical_Omon_tos_rgd = pickle.load(f)

imodel = 'IPSL-CM6A-LR-INCA'
historical_Omon_tos[imodel]['tos'][0].to_netcdf('data/test0.nc')
historical_Omon_tos_rgd[imodel]['tos'][0].to_netcdf('data/test1.nc')
historical_Omon_tos_rgd_alltime[imodel]['ann'][0].to_netcdf('data/test2.nc')

with open('data/sim/cmip6/historical_Omon_tos_rgd_alltime.pkl', 'rb') as f:
    historical_Omon_tos_rgd_alltime = pickle.load(f)
with open('data/obs/era5/mon/era5_mon_sst_rgd_alltime.pkl', 'rb') as f: era5_mon_sst_rgd_alltime = pickle.load(f)
print(historical_Omon_tos_rgd_alltime['ACCESS-ESM1-5']['ann'].lon.shape)
print(era5_mon_sst_rgd_alltime['ann'].lon.shape)

with open('data/sim/cmip6/historical_Omon_tos_rgd.pkl', 'rb') as f:
    historical_Omon_tos_rgd = pickle.load(f)
era5_mon_sst_rgd = xr.open_dataset('data/obs/era5/mon/era5_mon_sst_rgd.nc')
print(historical_Omon_tos_rgd['ACCESS-ESM1-5'].lon.shape)
print(era5_mon_sst_rgd.lon.shape)

with open('data/sim/cmip6/historical_Omon_tos.pkl', 'rb') as f:
    historical_Omon_tos = pickle.load(f)
era5_mon_sst = xr.open_dataset('data/obs/era5/mon/era5_mon_sst.nc')
print(historical_Omon_tos['ACCESS-ESM1-5'])
print(era5_mon_sst)

'''
# endregion


# region plot am global sst

output_png = 'figures/1_cmip6/1.0_cmip6_era5/1.0.0 era5 and historical am global sst.png'
cbar_label = r'ERA5 and CMIP6 $\mathit{historical}$' + ' annual mean SST (1979-2014) [$°C$]'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=28, cm_interval1=1, cm_interval2=2, cmap='viridis_r',)

nrow = 10
ncol = 6
fm_bottom = 2 / (4.4*nrow + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol, 4.4*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.Mollweide(central_longitude=180)},
    gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)

for irow in range(nrow):
    for jcol in range(ncol):
        if (jcol + ncol * irow <= len(common_models)):
            panel_label = f'({string.ascii_lowercase[irow]}{jcol+1})'
            model = (['ERA5'] + common_models)[jcol + ncol * irow]
            
            axs[irow, jcol] = globe_plot(ax_org=axs[irow, jcol])
            
            axs[irow, jcol].text(
                -0.04, 1.04, model + '\n' + panel_label,
                ha='left', va='top', transform=axs[irow, jcol].transAxes,
                linespacing=1.6)
        else:
            axs[irow, jcol].set_visible(False)

# plot data
for irow in range(nrow):
    for jcol in range(ncol):
        if (jcol + ncol * irow <= len(common_models)):
            model = (['ERA5'] + common_models)[jcol + ncol * irow]
            print(model)
            
            if (model == 'ERA5'):
                plot_data = era5_mon_sst_rgd_alltime['ann'].sel(
                    time=slice('1979', '2014')).mean(dim='time') - zerok
            else:
                plot_data = historical_Omon_tos_rgd_alltime[model]['ann'].sel(
                    time=slice('1979', '2014')).mean(dim='time')
            
            plt_mesh = axs[irow, jcol].pcolormesh(
                lon, lat, plot_data,
                norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
            axs[irow, jcol].add_feature(
                cfeature.LAND, color='white', zorder=2, edgecolor=None,lw=0)

cbar = fig.colorbar(
    plt_mesh, #cm.ScalarMappable(norm=pltnorm, cmap=pltcmp),
    ax=axs, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.5, ticks=pltticks, extend='both',
    anchor=(0.5, -0.84),)
cbar.ax.tick_params(length=2, width=0.4)
cbar.ax.set_xlabel(cbar_label)

fig.subplots_adjust(left=0.01, right = 0.996, bottom = fm_bottom, top = 0.99)
fig.savefig(output_png)
plt.close()

# endregion


# region plot am global sst bias

output_png = 'figures/1_cmip6/1.0_cmip6_era5/1.0.0 era5 and historical am global sst bias.png'
cbar_label = r'CMIP6 $\mathit{historical}$-ERA5' + ' annual mean SST (1979-2014) [$°C$]'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='RdBu',)

nrow = 10
ncol = 6
fm_bottom = 2 / (4.4*nrow + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol, 4.4*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.Mollweide(central_longitude=180)},
    gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)

for irow in range(nrow):
    for jcol in range(ncol):
        if (jcol + ncol * irow < len(common_models)):
            panel_label = f'({string.ascii_lowercase[irow]}{jcol+1})'
            model = common_models[jcol + ncol * irow]
            
            axs[irow, jcol] = globe_plot(ax_org=axs[irow, jcol])
            
            axs[irow, jcol].text(
                -0.04, 1.04, model + '\n' + panel_label,
                ha='left', va='top', transform=axs[irow, jcol].transAxes,
                linespacing=1.6)
        else:
            axs[irow, jcol].set_visible(False)

# plot data
for irow in range(nrow):
    for jcol in range(ncol):
        if (jcol + ncol * irow < len(common_models)):
            model = common_models[jcol + ncol * irow]
            print(model)
            
            plot_data = historical_Omon_tos_rgd_alltime[model]['ann'].sel(time=slice('1979', '2014')).mean(dim='time') - (era5_mon_sst_rgd_alltime['ann'].sel(time=slice('1979', '2014')).mean(dim='time').values - zerok)
            
            plt_mesh = axs[irow, jcol].pcolormesh(
                lon, lat, plot_data,
                norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
            axs[irow, jcol].add_feature(
                cfeature.LAND, color='white', zorder=2, edgecolor=None,lw=0)

cbar = fig.colorbar(
    plt_mesh, #cm.ScalarMappable(norm=pltnorm, cmap=pltcmp),
    ax=axs, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.5, ticks=pltticks, extend='both',
    anchor=(0.5, -0.84),)
cbar.ax.tick_params(length=2, width=0.4)
cbar.ax.set_xlabel(cbar_label)

fig.subplots_adjust(left=0.01, right = 0.996, bottom = fm_bottom, top = 0.99)
fig.savefig(output_png)
plt.close()


# endregion


# region plot am global sst ssp585-historical

common_models = sorted(list(set(historical_Omon_tos_rgd_alltime.keys()) & set(historical_Amon_tas_rgd_alltime.keys()) & set(historical_Amon_pr_rgd_alltime.keys()) & set(ssp585_Omon_tos_rgd_alltime.keys()) & set(ssp585_Amon_tas_rgd_alltime.keys()) & set(ssp585_Amon_pr_rgd_alltime.keys())))

# output_png = 'figures/1_cmip6/1.0_cmip6_era5/1.0.0 ssp585_historical am global sst.png'
# cbar_label = r'CMIP6 $\mathit{ssp585}$-$\mathit{historical}$' + ' annual mean SST (2090-2099 vs. 2005-2014) [$°C$]'
output_png = 'figures/1_cmip6/1.0_cmip6_era5/1.0.0 ssp585_historical am global sst_2050.png'
cbar_label = r'CMIP6 $\mathit{ssp585}$-$\mathit{historical}$' + ' annual mean SST (2050-2059 vs. 2005-2014) [$°C$]'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='RdBu',)

nrow = 7
ncol = 6
fm_bottom = 2 / (4.4*nrow + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol, 4.4*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.Mollweide(central_longitude=180)},
    gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)

for irow in range(nrow):
    for jcol in range(ncol):
        if (jcol + ncol * irow < len(common_models)):
            panel_label = f'({string.ascii_lowercase[irow]}{jcol+1})'
            model = common_models[jcol + ncol * irow]
            
            axs[irow, jcol] = globe_plot(ax_org=axs[irow, jcol])
            
            axs[irow, jcol].text(
                -0.04, 1.04, model + '\n' + panel_label,
                ha='left', va='top', transform=axs[irow, jcol].transAxes,
                linespacing=1.6)
        else:
            axs[irow, jcol].set_visible(False)

# plot data
for irow in range(nrow):
    for jcol in range(ncol):
        if (jcol + ncol * irow < len(common_models)):
            model = common_models[jcol + ncol * irow]
            print(model)
            
            # plot_data = ssp585_Omon_tos_rgd_alltime[model]['ann'].sel(time=slice('2090', '2099')).mean(dim='time') - historical_Omon_tos_rgd_alltime[model]['ann'].sel(time=slice('2005', '2014')).mean(dim='time')
            plot_data = ssp585_Omon_tos_rgd_alltime[model]['ann'].sel(time=slice('2050', '2059')).mean(dim='time') - historical_Omon_tos_rgd_alltime[model]['ann'].sel(time=slice('2005', '2014')).mean(dim='time')
            
            plt_mesh = axs[irow, jcol].pcolormesh(
                lon, lat, plot_data,
                norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
            axs[irow, jcol].add_feature(
                cfeature.LAND, color='white', zorder=2, edgecolor=None,lw=0)

cbar = fig.colorbar(
    plt_mesh, #cm.ScalarMappable(norm=pltnorm, cmap=pltcmp),
    ax=axs, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.5, ticks=pltticks, extend='both',
    anchor=(0.5, -0.84),)
cbar.ax.tick_params(length=2, width=0.4)
cbar.ax.set_xlabel(cbar_label)

fig.subplots_adjust(left=0.01, right = 0.996, bottom = fm_bottom, top = 0.99)
fig.savefig(output_png)
plt.close()


# endregion


# region plot am global sst ssp126-historical

common_models = sorted(list(set(historical_Omon_tos_rgd_alltime.keys()) & set(historical_Amon_tas_rgd_alltime.keys()) & set(historical_Amon_pr_rgd_alltime.keys()) & set(ssp126_Omon_tos_rgd_alltime.keys()) & set(ssp126_Amon_tas_rgd_alltime.keys()) & set(ssp126_Amon_pr_rgd_alltime.keys())))

# output_png = 'figures/1_cmip6/1.0_cmip6_era5/1.0.0 ssp126_historical am global sst.png'
# cbar_label = r'CMIP6 $\mathit{ssp126}$-$\mathit{historical}$' + ' annual mean SST (2090-2099 vs. 2005-2014) [$°C$]'
output_png = 'figures/1_cmip6/1.0_cmip6_era5/1.0.0 ssp126_historical am global sst_2050.png'
cbar_label = r'CMIP6 $\mathit{ssp126}$-$\mathit{historical}$' + ' annual mean SST (2050-2059 vs. 2005-2014) [$°C$]'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='RdBu',)

nrow = 7
ncol = 6
fm_bottom = 2 / (4.4*nrow + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol, 4.4*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.Mollweide(central_longitude=180)},
    gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)

for irow in range(nrow):
    for jcol in range(ncol):
        if (jcol + ncol * irow < len(common_models)):
            panel_label = f'({string.ascii_lowercase[irow]}{jcol+1})'
            model = common_models[jcol + ncol * irow]
            
            axs[irow, jcol] = globe_plot(ax_org=axs[irow, jcol])
            
            axs[irow, jcol].text(
                -0.04, 1.04, model + '\n' + panel_label,
                ha='left', va='top', transform=axs[irow, jcol].transAxes,
                linespacing=1.6)
        else:
            axs[irow, jcol].set_visible(False)

# plot data
for irow in range(nrow):
    for jcol in range(ncol):
        if (jcol + ncol * irow < len(common_models)):
            model = common_models[jcol + ncol * irow]
            print(model)
            
            # plot_data = ssp126_Omon_tos_rgd_alltime[model]['ann'].sel(time=slice('2090', '2099')).mean(dim='time') - historical_Omon_tos_rgd_alltime[model]['ann'].sel(time=slice('2005', '2014')).mean(dim='time')
            plot_data = ssp126_Omon_tos_rgd_alltime[model]['ann'].sel(time=slice('2050', '2059')).mean(dim='time') - historical_Omon_tos_rgd_alltime[model]['ann'].sel(time=slice('2005', '2014')).mean(dim='time')
            
            plt_mesh = axs[irow, jcol].pcolormesh(
                lon, lat, plot_data,
                norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
            axs[irow, jcol].add_feature(
                cfeature.LAND, color='white', zorder=2, edgecolor=None,lw=0)

cbar = fig.colorbar(
    plt_mesh, #cm.ScalarMappable(norm=pltnorm, cmap=pltcmp),
    ax=axs, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.5, ticks=pltticks, extend='both',
    anchor=(0.5, -0.84),)
cbar.ax.tick_params(length=2, width=0.4)
cbar.ax.set_xlabel(cbar_label)

fig.subplots_adjust(left=0.01, right = 0.996, bottom = fm_bottom, top = 0.99)
fig.savefig(output_png)
plt.close()


# endregion



