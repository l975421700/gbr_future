

# qsub -I -q normal -l walltime=5:00:00,ncpus=1,mem=192GB,jobfs=100MB,storage=gdata/v46+scratch/v46


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
from matplotlib.colors import BoundaryNorm
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
    panel_labels,
    era5_varlabels,
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
