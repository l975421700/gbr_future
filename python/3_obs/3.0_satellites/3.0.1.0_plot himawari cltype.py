

# qsub -I -q normal -l walltime=4:00:00,ncpus=1,mem=192GB,jobfs=100MB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22


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
from metpy.calc import pressure_to_height_std, geopotential_to_height
from metpy.units import units
import metpy.calc as mpcalc
import pickle
import glob

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib import cm
import cartopy.crs as ccrs
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
from matplotlib.patches import Rectangle
import matplotlib.ticker as mticker
import matplotlib.animation as animation

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
    month_jan,
    monthini,
    seasons,
    seconds_per_d,
    zerok,
    era5_varlabels,
    ds_color,
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
    time_weighted_mean,
    mon_sea_ann,
    cdo_regrid,)

# endregion


# region import data

with open('/scratch/v46/qg8515/data/obs/jaxa/clp/cltype_frequency_alltime.pkl', 'rb') as f:
    cltype_frequency_alltime = pickle.load(f)

cltypes = {
    'hcc': ['Cirrus', 'Cirrostratus', 'Deep convection'],
    'mcc': ['Altocumulus', 'Altostratus', 'Nimbostratus'],
    'lcc': ['Cumulus', 'Stratocumulus', 'Stratus'],
    'tcc': ['Cirrus', 'Cirrostratus', 'Deep convection', 'Altocumulus', 'Altostratus', 'Nimbostratus', 'Cumulus', 'Stratocumulus', 'Stratus']}


'''
cloudtypes = [
    'Cirrus', 'Cirrostratus', 'Deep convection',
    'Altocumulus', 'Altostratus', 'Nimbostratus',
    'Cumulus', 'Stratocumulus', 'Stratus']
CL_RelFreq = xr.open_dataset('data/obs/jaxa/clp/CL_RelFreq_2016.nc').CL_RelFreq
'''
# endregion


# region plot mm h/m/l/tcc

# iregion='himawari'
iregion='barra_c2'

if iregion=='himawari':
    panelh=6.6
    panelw=6.6
    extent=[80, 200, -60, 60]
elif iregion=='barra_c2':
    panelh=6
    panelw=6.6
    extent=[110.58, 157.34, -43.69, -7.01]

nrow = 3
ncol = 4
fm_bottom = 1.6 / (panelh*nrow + 2)
min_lon, max_lon, min_lat, max_lat = [110.58, 157.34, -43.69, -7.01]

for icltype in ['hcc', 'mcc', 'lcc', 'tcc']:
    # icltype='hcc'
    print(f'#-------------------------------- {icltype}')
    print(cltypes[icltype])
    
    opng = f'figures/3_satellites/3.0_hamawari/3.0.1_cltype/3.0.1.0_himawari_{iregion}_{icltype}_frequency_mm.png'
    cbar_label = f'2015-2024 Himawari 8/9 {era5_varlabels[icltype]}'
    
    if icltype in ['hcc', 'mcc', 'lcc']:
        extend='max'
        if iregion=='himawari':
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=60, cm_interval1=5, cm_interval2=10, cmap='Blues_r',)
        elif iregion=='barra_c2':
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=60, cm_interval1=5, cm_interval2=10, cmap='Blues_r',)
    elif icltype=='tcc':
        if iregion=='himawari':
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=100, cm_interval1=5, cm_interval2=10, cmap='Blues_r',)
            extend='neither'
        elif iregion=='barra_c2':
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=80, cm_interval1=5, cm_interval2=10, cmap='Blues_r',)
            extend='max'
    
    fig, axs = plt.subplots(
        nrow, ncol, figsize=np.array([panelw*ncol, panelh*nrow + 2]) / 2.54,
        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
        gridspec_kw={'hspace': 0.1, 'wspace': 0.02},)
    
    for irow in range(nrow):
        for jcol in range(ncol):
            # irow=0; jcol=0
            print(f'#---------------- {irow} {jcol} {month_jan[irow*4+jcol]}')
            axs[irow, jcol] = regional_plot(
                extent=extent, central_longitude=180, ax_org=axs[irow, jcol])
            if iregion=='himawari':
                axs[irow, jcol].add_patch(
                    Rectangle((min_lon, min_lat), max_lon-min_lon,
                              max_lat-min_lat, ec = 'red', color = 'None',
                              lw = 0.5, transform=ccrs.PlateCarree(), zorder=2))
            
            plt_data = cltype_frequency_alltime['mm'][irow*4+jcol].sel(types=cltypes[icltype]).sum(dim='types')
            if iregion=='barra_c2':
                plt_data = plt_data.sel(lon=slice(min_lon, max_lon), lat=slice(max_lat, min_lat))
            plt_mesh = axs[irow, jcol].pcolormesh(
                plt_data.lon, plt_data.lat, plt_data,
                norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
            
            pltmean = plt_data.weighted(np.cos(np.deg2rad(plt_data.lat))).mean().values
            if ((irow==0) & (jcol==0)):
                plt_text = f'({string.ascii_lowercase[irow]}{jcol+1}) {month_jan[irow*4+jcol]} Mean: {np.round(pltmean, 1)}'
            else:
                plt_text = f'({string.ascii_lowercase[irow]}{jcol+1}) {month_jan[irow*4+jcol]} {np.round(pltmean, 1)}'
            
            # plt_text = f'({string.ascii_lowercase[irow]}{jcol+1}) {month_jan[irow*4+jcol]}'
            plt.text(
                0, 1.02, plt_text,
                transform=axs[irow, jcol].transAxes, fontsize=10,
                ha='left', va='bottom', rotation='horizontal')
    
    cbar = fig.colorbar(
        plt_mesh, #cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), #
        ax=axs, aspect=30, format=remove_trailing_zero_pos,
        orientation="horizontal", shrink=0.5, ticks=pltticks, extend=extend,
        anchor=(0.5, -0.56),)
    cbar.ax.set_xlabel(cbar_label)
    fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom, top = 0.97)
    fig.savefig(opng)


# endregion


# region plot am/sm h/m/l/tcc

# endregion


# region plot the frequency

opng = 'figures/3_satellites/3.0_hamawari/3.0.0_cltype_frequency_2016.png'
nrow = 3
ncol = 3
fm_bottom = 1.5 / (6.6*nrow + 2)

cbar_label = r'Cloud occurrence frequency in 2016 from Himawari 8 [$\%$]'
pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=30, cm_interval1=5, cm_interval2=5, cmap='Blues_r',)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([6.6*ncol, 6.6*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
    gridspec_kw={'hspace': 0.1, 'wspace': 0.02},)

ipanel=0
for irow in range(nrow):
    for jcol in range(ncol):
        print(cloudtypes[ipanel])
        axs[irow, jcol] = regional_plot(
            extent=[80, 200, -60, 60], central_longitude=180,
            ax_org=axs[irow, jcol],)
        
        plt.text(
            0, 1.02, f'{panel_labels[ipanel]} {cloudtypes[ipanel]}',
            transform=axs[irow, jcol].transAxes, fontsize=10,
            ha='left', va='bottom', rotation='horizontal')
        
        plt_data = cltype_frequency_alltime['ann'].sel(time='2016', types=cloudtypes[ipanel]).squeeze()
        plt_mesh = axs[irow, jcol].pcolormesh(
            plt_data.lon, plt_data.lat, plt_data,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
        ipanel += 1

cbar = fig.colorbar(
    plt_mesh, # cm.ScalarMappable(norm=pltnorm, cmap=pltcmp),
    ax=axs, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.5, ticks=pltticks, extend='max',
    anchor=(0.5, -0.56),)
cbar.ax.set_xlabel(cbar_label)

fig.subplots_adjust(left=0.04, right = 0.99, bottom = fm_bottom, top = 0.97)
fig.savefig(opng)


# endregion


# region plot high, medium, and low cloud frequency

opng = 'figures/3_satellites/3.0_hamawari/3.0.0_cltype_frequency_2016_hml.png'

nrow = 1
ncol = 3
fm_bottom = 1.6 / (6.6*nrow + 2)

cloudgroups = ['High cloud', 'Middle cloud', 'Low cloud', 'Total cloud', ]

cbar_label = r'Cloud occurrence frequency in 2016 from Himawari 8 [$\%$]'
pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=80, cm_interval1=5, cm_interval2=10, cmap='Blues_r',)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([6.6*ncol, 6.6*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
    gridspec_kw={'hspace': 0.1, 'wspace': 0.02},)

for jcol in range(ncol):
    print(f'{panel_labels[jcol]} {cloudgroups[jcol]}')
    axs[jcol] = regional_plot(
        extent=[80, 200, -60, 60], central_longitude=180, ax_org=axs[jcol],)
    plt.text(
        0, 1.02, f'{panel_labels[jcol]} {cloudgroups[jcol]}',
        transform=axs[jcol].transAxes, fontsize=10,
        ha='left', va='bottom', rotation='horizontal')
    
    if jcol < 3:
        print(cltype_frequency_alltime['ann'].types[(jcol*3+1):(jcol*3+4)].values)
        plt_data = cltype_frequency_alltime['ann'].sel(time='2016').squeeze()[(jcol*3+1):(jcol*3+4)].sum(dim='types')
    elif jcol == 3:
        print('Total cloud')
        # plt_data = CL_RelFreq[1:].sum(dim='types')
    
    plt_mesh = axs[jcol].pcolormesh(
        plt_data.lon, plt_data.lat, plt_data,
        norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_mesh, ax=axs, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.5, ticks=pltticks, extend='max',
    anchor=(0.5, 0.2),)
cbar.ax.set_xlabel(cbar_label)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom, top = 0.99)
fig.savefig(opng)

# endregion


# region plot total cloud cover

opng = 'figures/3_satellites/3.0_hamawari/3.0.0_cltype_frequency_2016_t.png'

cbar_label = 'Total cloud occurrence frequency\nin 2016 from Himawari 8 [%]'
pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=10, cmap='Blues_r',)

fig, ax = regional_plot(extent=[80, 200, -60, 60], central_longitude=180, figsize = np.array([6.6, 8.6]) / 2.54)

plt_data = cltype_frequency_alltime['ann'].sel(time='2016').squeeze()[1:-1].sum(dim='types')
plt_mesh = ax.pcolormesh(
    plt_data.lon, plt_data.lat, plt_data,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_mesh,
    ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='neither',
    pad=0.03, fraction=0.12,)
cbar.ax.set_xlabel(cbar_label, linespacing=1.5)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = 0.12, top = 0.99)
fig.savefig(opng)


# endregion


# region plot ISCCP classification

ISCCP_types = {'Clear': 0,
               'Cirrus': 1, 'Cirrostratus': 2, 'Deep convection': 3,
               'Altocumulus': 4, 'Altostratus': 5, 'Nimbostratus':6,
               'Cumulus':7, 'Stratocumulus': 8, 'Stratus': 9,
            #    'Unknown':10,
               }

year, month, day, hour, minute = 2020, 6, 2, 4, 0

ds_cltype = xr.open_dataset(f'/scratch/v46/qg8515/data/obs/jaxa/clp/{year}{month:02d}/{day:02d}/{hour:02d}/CLTYPE_{year}{month:02d}{day:02d}{hour:02d}{minute:02d}.nc').CLTYPE.squeeze()

# plot ds_cltype

opng = f'figures/3_satellites/3.0_hamawari/3.0.1_cltype/3.0.1.1 himawari ISCCP cltype {year}{month:02d}{day:02d}{hour:02d}{minute:02d}.png'
extent = [-5499500., 5499500., -5499500., 5499500.]
transform = ccrs.Geostationary(central_longitude=140.7, satellite_height=35785863.0)

plt_text = f'{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d} UTC\nHimawari ISCCP cloud types'

fig, ax = plt.subplots(figsize=np.array([7+2.5, 7+1])/2.54, subplot_kw={'projection': transform})

coastline = cfeature.NaturalEarthFeature(
    'physical', 'coastline', '10m', edgecolor='k',
    facecolor='none', lw=0.1)
ax.add_feature(coastline, zorder=2, alpha=0.75)
borders = cfeature.NaturalEarthFeature(
    'cultural', 'admin_0_boundary_lines_land', '10m',
    edgecolor='k', facecolor='none', lw=0.1)
ax.add_feature(borders, zorder=2, alpha=0.75)
gl = ax.gridlines(
    crs=ccrs.PlateCarree(), lw=0.1, zorder=2, alpha=0.35,
    color='k', linestyle='--',)
gl.xlocator = mticker.FixedLocator(np.arange(0, 360 + 1e-4, 10))
gl.ylocator = mticker.FixedLocator(np.arange(-90, 90 + 1e-4, 10))
plt.text(0.5, -0.03, plt_text, transform=ax.transAxes, fontsize=8,
         ha='center', va='top', rotation='horizontal', linespacing=1.5)

colors = {
    'Clear': (1, 1, 1, 0),
    'Cirrus': plt.cm.colors.to_rgba('pink'),
    'Cirrostratus': plt.cm.colors.to_rgba('tab:pink'),
    'Deep convection': plt.cm.colors.to_rgba('tab:red'),
    'Altocumulus': plt.cm.colors.to_rgba('tab:brown'),
    'Altostratus': plt.cm.colors.to_rgba('tab:gray'),
    'Nimbostratus': plt.cm.colors.to_rgba('lightgray'),
    'Cumulus': plt.cm.colors.to_rgba('tab:blue'),
    'Stratocumulus': plt.cm.colors.to_rgba('deepskyblue'),
    'Stratus': plt.cm.colors.to_rgba('cyan'),
    # 'Unknown': (0, 0, 0, 1),
}
pltcmp = ListedColormap([colors[itype] for itype in ISCCP_types])
pltlevel = np.arange(len(ISCCP_types) + 1) - 0.5
pltticks = np.arange(len(ISCCP_types))
pltnorm = BoundaryNorm(pltlevel, len(ISCCP_types))

plt_mesh = ax.pcolormesh(
    ds_cltype.longitude, ds_cltype.latitude, ds_cltype,
    norm=pltnorm,
    cmap=pltcmp, transform=ccrs.PlateCarree())
ax.imshow(np.ones((100,100,3)),extent=extent,transform=transform,zorder=0)

cbar = fig.colorbar(
    plt_mesh,
    format=remove_trailing_zero_pos,
    orientation="vertical", ticks=pltticks, extend='neither',
    cax=fig.add_axes([7.05/(7+2.5), 0.5/(7+1), 0.02, 0.9]))
cbar.ax.minorticks_off()
cbar.ax.set_yticklabels(list(colors.keys()), fontsize=8)
cbar.ax.tick_params(length=2, width=0.5)

fig.subplots_adjust(left=0.01, right=1-2.5/(7+2.5), bottom=1/(7+1), top=0.99)
fig.savefig(opng)
plt.close()


'''
# check cltype at Willis Island at 2020-06-02 04:00
ds = xr.open_dataset('/home/563/qg8515/scratch/data/obs/jaxa/clp/202006/02/04/CLTYPE_202006020400.nc')
station = 'Willis Island'
slat = -16.288
slon = 149.965

ds.CLTYPE.sel(latitude=slat, longitude=slon, method='nearest')


colors = plt.get_cmap('viridis', len(ISCCP_types))
pltcmp = {key: colors(i) for i, key in enumerate(ISCCP_types.keys())}
bounds = np.arange(len(ISCCP_types) + 1) - 0.5
pltnorm = plt.Normalize(vmin=0, vmax=len(ISCCP_types) - 1)
'''
# endregion


# region plot ISCCP cloud type over c2_domain

# options
year, month, day, hour, minute = 2020, 6, 2, 5, 0
plt_region = 'c2_domain'

# settings
ISCCP_types = {'Clear': 0,
               'Cirrus': 1, 'Cirrostratus': 2, 'Deep convection': 3,
               'Altocumulus': 4, 'Altostratus': 5, 'Nimbostratus':6,
               'Cumulus':7, 'Stratocumulus': 8, 'Stratus': 9,}
colors = {
    'Clear': (1, 1, 1, 0),
    'Cirrus': plt.cm.colors.to_rgba('pink'),
    'Cirrostratus': plt.cm.colors.to_rgba('tab:pink'),
    'Deep convection': plt.cm.colors.to_rgba('tab:red'),
    'Altocumulus': plt.cm.colors.to_rgba('tab:brown'),
    'Altostratus': plt.cm.colors.to_rgba('tab:gray'),
    'Nimbostratus': plt.cm.colors.to_rgba('lightgray'),
    'Cumulus': plt.cm.colors.to_rgba('tab:blue'),
    'Stratocumulus': plt.cm.colors.to_rgba('deepskyblue'),
    'Stratus': plt.cm.colors.to_rgba('cyan'),}
pltcmp = ListedColormap([colors[itype] for itype in ISCCP_types])
pltlevel = np.arange(len(ISCCP_types) + 1) - 0.5
pltticks = np.arange(len(ISCCP_types))
pltnorm = BoundaryNorm(pltlevel, len(ISCCP_types))

opng = f'figures/3_satellites/3.0_hamawari/3.0.1_cltype/3.0.1.1 himawari ISCCP cltype {plt_region} {year}{month:02d}{day:02d}{hour:02d}{minute:02d}.png'
plt_text = f'{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d} UTC\nHimawari ISCCP cloud types'

ds_cltype = xr.open_dataset(f'scratch/data/obs/jaxa/clp/{year}{month:02d}/{day:02d}/{hour:02d}/CLTYPE_{year}{month:02d}{day:02d}{hour:02d}{minute:02d}.nc').CLTYPE.squeeze()

fig, ax = regional_plot(
    figsize = np.array([6.6+3, 6+1]) / 2.54,
    extent=[110.58, 157.34, -43.69, -7.01], central_longitude=180,)

plt.text(0.5, -0.03, plt_text, transform=ax.transAxes,
         ha='center', va='top', rotation='horizontal', linespacing=1.5)
plt_mesh = ax.pcolormesh(
    ds_cltype.longitude, ds_cltype.latitude, ds_cltype,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree())
cbar = fig.colorbar(
    plt_mesh,
    format=remove_trailing_zero_pos,
    orientation="vertical", ticks=pltticks, extend='neither',
    cax=fig.add_axes([6.7/(6.6+3), 0.5/(6+1), 0.02, 0.9]))
cbar.ax.minorticks_off()
cbar.ax.set_yticklabels(list(colors.keys()))
cbar.ax.tick_params(length=2, width=0.5)
fig.subplots_adjust(left=0.01, right=1-3/(6.6+3), bottom=1/(6+1), top=0.99)
fig.savefig(opng)


'''
'''
# endregion


# region plot daily cycle of observed counts

with open('data/obs/era5/hourly/era5_hourly_alltime_mtdwswrf.pkl', 'rb') as f:
    era5_hourly_alltime = pickle.load(f)

min_lon, max_lon, min_lat, max_lat = [110.58, 157.34, -43.69, -7.01]
era5_mtdwswrf_ann = era5_hourly_alltime['ann'].sel(time=slice('2016', '2023'), lon=slice(min_lon, max_lon), lat=slice(max_lat, min_lat))
era5_mtdwswrf_ann = era5_mtdwswrf_ann.weighted(np.cos(np.deg2rad(era5_mtdwswrf_ann.lat))).mean(dim=['lon', 'lat'])
era5_mtdwswrf_am = era5_mtdwswrf_ann.mean(dim='time').roll(hour=-1)
era5_mtdwswrf_astd = era5_mtdwswrf_ann.std(dim='time', ddof=1).roll(hour=-1)


with open('/scratch/v46/qg8515/data/obs/jaxa/clp/cltype_hourly_count_alltime.pkl', 'rb') as f:
    cltype_hourly_count_alltime = pickle.load(f)

daily_cycle_count_ann = cltype_hourly_count_alltime['ann'].sel(time=slice('2016', '2023')).loc[{'types': 'finite'}].weighted(np.cos(np.deg2rad(cltype_hourly_count_alltime['mon'].lat))).mean(dim=['lon', 'lat']) * 12 / 365 / 6 * 100
# daily_cycle_count_ann = cltype_hourly_count_alltime['ann'].sel(time=slice('2016', '2023')).loc[{'types': 'finite'}].mean(dim=['lon', 'lat']) * 12 / 365 / 6 * 100
daily_cycle_count_am = daily_cycle_count_ann.mean(dim='time')
daily_cycle_count_astd = daily_cycle_count_ann.std(dim='time', ddof=1)

opng = f'figures/3_satellites/3.0_hamawari/3.0.1_dm average count of hourly observations.png'

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)
ax.plot(range(0, 24, 1), daily_cycle_count_am, '.-', lw=0.75, markersize=6,
        c=ds_color['Himawari'])
ax.errorbar(range(0, 24, 1), daily_cycle_count_am,
            yerr=daily_cycle_count_astd, fmt='none',
            capsize=4, color=ds_color['Himawari'],lw=0.5, alpha=0.7)

ax.set_xticks(np.arange(0, 24, 6))
ax.set_xlabel('UTC', size=9)
ax.set_ylim(0, 100)
ax.set_yticks(np.arange(0, 100+1e-4, 20))
ax.yaxis.set_major_formatter(remove_trailing_zero_pos)
ax.set_ylabel(r'Percentage of potential Himawari observations [$\%$]', size=8)


ax2 = ax.twinx()
ax2.plot(range(0, 24, 1), era5_mtdwswrf_am, '.-', lw=0.75, markersize=6,
        c='tab:blue')
ax2.errorbar(range(0, 24, 1), era5_mtdwswrf_am,
             yerr=era5_mtdwswrf_astd, fmt='none',
             capsize=4, color='tab:blue', lw=0.5, alpha=0.7)
ax2.set_ylim(0, 1250)
ax2.set_yticks(np.arange(0, 1250+1e-4, 250))
ax2.set_yticklabels(np.arange(0, 1250+1e-4, 250), c='tab:blue')
ax2.yaxis.set_major_formatter(remove_trailing_zero_pos)
ax2.set_ylabel(r'ERA5 mean TOA downward SW radiation [$W \; m^{-2}$]', size=8, c='tab:blue')


ax.grid(True, which='both', linewidth=0.5, color='gray',
        alpha=0.5, linestyle='--')
fig.subplots_adjust(left=0.16, right=0.84, bottom=0.13, top=0.98)
fig.savefig(opng)



'''
(daily_cycle_count - cltype_hourly_count_alltime['ann'].sel(time=slice('2016', '2023')).loc[{'types': 'finite'}].mean(dim='time').mean(dim=['lon', 'lat']).values) / daily_cycle_count
'''
# endregion


# region plot daily cycle of low/middle/high/total cloud frequency
# double check with a clear mind later

with open('/scratch/v46/qg8515/data/obs/jaxa/clp/cltype_hourly_frequency_alltime.pkl', 'rb') as f:
    cltype_hourly_frequency_alltime = pickle.load(f)

cltypes = {
    'hcc': ['Cirrus', 'Cirrostratus', 'Deep convection'],
    'mcc': ['Altocumulus', 'Altostratus', 'Nimbostratus'],
    'lcc': ['Cumulus', 'Stratocumulus', 'Stratus'],
    'tcc': ['Cirrus', 'Cirrostratus', 'Deep convection', 'Altocumulus', 'Altostratus', 'Nimbostratus', 'Cumulus', 'Stratocumulus', 'Stratus'],
    'clear': ['Clear'],
    'unknown': ['Unknown']
    }

years = '2016'
yeare = '2021'

for icltype2 in ['cll', 'clm', 'clh', 'clt', 'clear']:
    # icltype2='cll'
    # ['cll', 'clm', 'clh', 'clt']
    try:
        icltype = cmip6_era5_var[icltype2]
    except KeyError:
        icltype = icltype2
    print(f'#-------------------------------- {icltype} {icltype2}')
    print(cltypes[icltype])
    
    cltype_hourly_ann = cltype_hourly_frequency_alltime['ann'].sel(time=slice(years, yeare), types=cltypes[icltype]).sum(dim='types', skipna=False).weighted(np.cos(np.deg2rad(cltype_hourly_frequency_alltime['ann'].lat))).mean(dim=['lon', 'lat'], skipna=True)
    cltype_hourly_ann[:, 7:23] = np.nan
    # cltype_hourly_ann = cltype_hourly_frequency_alltime['ann'].sel(time=slice(years, yeare), types=cltypes[icltype]).sum(dim='types').weighted(np.cos(np.deg2rad(cltype_hourly_frequency_alltime['ann'].lat))).mean(dim=['lon', 'lat'])
    cltype_hourly_am = cltype_hourly_ann.mean(dim='time')
    cltype_hourly_astd = cltype_hourly_ann.std(dim='time', ddof=1)
    
    opng = f'figures/3_satellites/3.0_hamawari/3.0.1_cltype/3.0.1.2_himawari dm hourly {icltype2}.png'
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)
    ax.plot(range(0, 24, 1), cltype_hourly_am, '.-', lw=0.75, markersize=6,
            c=ds_color['Himawari'])
    ax.errorbar(range(0, 24, 1), cltype_hourly_am,
                yerr=cltype_hourly_astd, fmt='none',
                capsize=4, color=ds_color['Himawari'],lw=0.5, alpha=0.7)
    
    ax.set_xticks(np.arange(0, 24, 6))
    ax.set_xlabel('UTC', size=9)
    # ax.set_ylim(0, 40)
    # ax.set_yticks(np.arange(0, 40+1e-4, 5))
    ax.yaxis.set_major_formatter(remove_trailing_zero_pos)
    ax.set_ylabel(f'Himawari {era5_varlabels[icltype]}', size=8)
    
    ax.grid(True, which='both', linewidth=0.5, color='gray',
            alpha=0.5, linestyle='--')
    fig.subplots_adjust(left=0.16, right=0.99, bottom=0.13, top=0.98)
    fig.savefig(opng)



'''
(np.isnan(cltype_hourly_frequency_alltime['mon'][0, :, 0])).sum().values
'''
# endregion

