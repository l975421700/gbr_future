

# qsub -I -q normal -l walltime=10:00:00,ncpus=1,mem=192GB,storage=gdata/v46+gdata/rt52+gdata/ob53


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
import cartopy.crs as ccrs
from matplotlib import cm
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
import warnings
warnings.filterwarnings('ignore')

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


# region plot era5 and cmip6 data

era5_sl_mon_alltime = {}
cmip6_data_regridded_alltime_ens = {}

for experiment_id in ['historical', 'amip']:
    # experiment_id = 'historical'
    print(f'#-------------------------------- {experiment_id}')
    cmip6_data_regridded_alltime_ens[experiment_id] = {}
    
    for table_id, variable_id in zip(['Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Amon'], ['clt', 'hfls', 'hfss', 'pr', 'rlds', 'rlus', 'rsds', 'rsus']):
        # table_id = 'Amon'; variable_id = 'tas'
        # ['Amon', 'Amon', 'Amon', 'Amon', 'Omon'], ['tas', 'rsut', 'rsdt', 'rlut', 'tos']
        var = cmip6_era5_var[variable_id]
        print(f'#---------------- {table_id} {variable_id} {var}')
        cmip6_data_regridded_alltime_ens[experiment_id][table_id] = {}
        
        with open(f'/home/563/qg8515/scratch/data/sim/cmip6/{experiment_id}_{table_id}_{variable_id}_regridded_alltime_ens.pkl', 'rb') as f:
            cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id] = pickle.load(f)
        with open(f'data/obs/era5/mon/era5_sl_mon_alltime_{var}.pkl', 'rb') as f:
            era5_sl_mon_alltime[var] = pickle.load(f)
        
        source_ids = cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id]['am'].source_id.values.astype('object')
        
        opng = f'figures/1_cmip6/1.0_cmip6_era5/1.0.0 era5 and {experiment_id} am global {variable_id}.png'
        cbar_label = r'ERA5 and CMIP6 $' + experiment_id + '$ annual mean (1979-2014) ' + era5_varlabels[var]
        
        if variable_id=='tos':
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=-2, cm_max=30, cm_interval1=2, cm_interval2=4, cmap='viridis_r',)
        elif variable_id=='rlut':
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=-300, cm_max=-130, cm_interval1=10, cm_interval2=20, cmap='viridis',)
        elif variable_id=='rsdt':
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=180, cm_max=420, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
        elif variable_id=='rsut':
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=-250, cm_max=-20, cm_interval1=10, cm_interval2=20, cmap='viridis')
        elif variable_id=='tas':
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=-48, cm_max=32, cm_interval1=2, cm_interval2=8, cmap='BrBG', asymmetric=True,)
        elif variable_id=='clt':
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=10, cmap='viridis_r',)
        elif variable_id=='hfls':
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=-280, cm_max=0, cm_interval1=20, cm_interval2=40, cmap='viridis',)
        elif variable_id=='hfss':
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=-120, cm_max=60, cm_interval1=10, cm_interval2=20, cmap='BrBG', asymmetric=True,)
        elif variable_id=='pr':
            pltlevel = np.array([0, 0.1, 0.2, 0.5, 1, 2, 4, 6, 8, 10, 12, 16, 20,])
            pltticks = np.array([0, 0.1, 0.2, 0.5, 1, 2, 4, 6, 8, 10, 12, 16, 20,])
            pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
            pltcmp = plt.get_cmap('viridis_r', len(pltlevel)-1)
        elif variable_id=='rlds':
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=80, cm_max=430, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
        elif variable_id=='rlus':
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=-540, cm_max=-120, cm_interval1=10, cm_interval2=40, cmap='viridis')
        elif variable_id=='rsds':
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=60, cm_max=330, cm_interval1=10, cm_interval2=20, cmap='viridis_r')
        elif variable_id=='rsus':
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=-220, cm_max=0, cm_interval1=10, cm_interval2=20, cmap='viridis')
        
        ncol = 8
        nrow = int(np.ceil((len(source_ids) + 1) / ncol))
        
        fm_bottom = 2 / (4.4*nrow + 2)
        fig, axs = plt.subplots(
            nrow, ncol, figsize=np.array([8.8*ncol, 4.4*nrow + 2]) / 2.54,
            subplot_kw={'projection': ccrs.Mollweide(central_longitude=180)},
            gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)
        
        for irow in range(nrow):
            for jcol in range(ncol):
                if (jcol + ncol * irow <= len(source_ids)):
                    panel_label = f'({string.ascii_lowercase[irow]}{jcol+1})'
                    source_id = (['ERA5'] + list(source_ids))[jcol + ncol * irow]
                    print(f'#-------- {source_id}')
                    axs[irow, jcol] = globe_plot(ax_org=axs[irow, jcol])
                    axs[irow, jcol].text(
                        -0.04, 1.04, source_id + '\n' + panel_label,
                        ha='left', va='top', transform=axs[irow, jcol].transAxes, linespacing=1.6)
                    
                    # plot data
                    if source_id=='ERA5':
                        plt_data = era5_sl_mon_alltime[var]['ann'].sel(time=slice('1979', '2014')).mean(dim='time')
                    else:
                        plt_data = cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id]['ann'].sel(time=slice('1979', '2014'), source_id=source_id).mean(dim='time')
                    plt_mesh = axs[irow, jcol].pcolormesh(
                        plt_data.lon, plt_data.lat, plt_data,
                        norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree())
                    if variable_id=='tos':
                        axs[irow, jcol].add_feature(cfeature.LAND, color='white', zorder=2, edgecolor=None,lw=0)
                    
                else:
                    axs[irow, jcol].set_visible(False)
        
        cbar = fig.colorbar(
            plt_mesh, #cm.ScalarMappable(norm=pltnorm, cmap=pltcmp),
            ax=axs, aspect=40, format=remove_trailing_zero_pos,
            orientation="horizontal", shrink=0.5, ticks=pltticks, extend='both',
            anchor=(0.5, -0.8),)
        cbar.ax.tick_params(length=2, width=0.4)
        cbar.ax.set_xlabel(cbar_label)
        
        fig.subplots_adjust(left=0.01, right=0.996, bottom=fm_bottom, top=0.99)
        fig.savefig(opng)
        plt.close()
        
        del cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id], era5_sl_mon_alltime[var]


# endregion


# region plot cmip6-era5 data

era5_sl_mon_alltime = {}
cmip6_data_regridded_alltime_ens = {}
for experiment_id in ['historical', 'amip']:
    # experiment_id = 'amip'
    print(f'#-------------------------------- {experiment_id}')
    cmip6_data_regridded_alltime_ens[experiment_id] = {}
    
    for table_id, variable_id in zip(['Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Amon'], ['clt', 'hfls', 'hfss', 'pr', 'rlds', 'rlus', 'rsds', 'rsus']):
        # table_id = 'Amon'; variable_id = 'tas'
        # ['Amon', 'Amon', 'Amon', 'Amon', 'Omon'], ['tas', 'rsut', 'rsdt', 'rlut', 'tos']
        var = cmip6_era5_var[variable_id]
        print(f'#---------------- {table_id} {variable_id} {var}')
        cmip6_data_regridded_alltime_ens[experiment_id][table_id] = {}
        
        with open(f'/home/563/qg8515/scratch/data/sim/cmip6/{experiment_id}_{table_id}_{variable_id}_regridded_alltime_ens.pkl', 'rb') as f:
            cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id] = pickle.load(f)
        source_ids = cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id]['am'].source_id.values.astype('object')
        with open(f'data/obs/era5/mon/era5_sl_mon_alltime_{var}.pkl', 'rb') as f:
            era5_sl_mon_alltime[var] = pickle.load(f)
        plt_era5_ann = regrid(era5_sl_mon_alltime[var]['ann'].sel(time=slice('1979', '2014')))
        plt_era5 = plt_era5_ann.mean(dim='time')
        
        if variable_id in ['tos', 'tas']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=-5,cm_max=5,cm_interval1=0.5,cm_interval2=1,cmap='BrBG')
        elif variable_id in ['rlut', 'rsdt', 'rsut', 'hfls', 'hfss', 'rlds', 'rlus', 'rsds', 'rsus']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=-20,cm_max=20,cm_interval1=2,cm_interval2=4,cmap='BrBG')
        elif variable_id in ['clt']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=10, cmap='BrBG')
        elif variable_id in ['pr']:
            pltlevel = np.array([-6, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 6])
            pltticks = np.array([-6, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 6])
            pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
            pltcmp = plt.get_cmap('BrBG', len(pltlevel)-1)
        
        opng = f'figures/1_cmip6/1.0_cmip6_era5/1.0.0 {experiment_id}-era5 am global {variable_id}.png'
        cbar_label = r'CMIP6 $' + experiment_id + '$-ERA5 annual mean (1979-2014) ' + era5_varlabels[var]
        ncol = 8
        nrow = int(np.ceil(len(source_ids) / ncol))
        fm_bottom = 2 / (4.4*nrow + 2)
        
        fig, axs = plt.subplots(
            nrow, ncol, figsize=np.array([8.8*ncol, 4.4*nrow + 2]) / 2.54,
            subplot_kw={'projection': ccrs.Mollweide(central_longitude=180)},
            gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)
        for irow in range(nrow):
            for jcol in range(ncol):
                if (jcol + ncol * irow < len(source_ids)):
                    panel_label = f'({string.ascii_lowercase[irow]}{jcol+1})'
                    source_id = list(source_ids)[jcol + ncol * irow]
                    print(f'#-------- {source_id}')
                    axs[irow, jcol] = globe_plot(ax_org=axs[irow, jcol])
                    axs[irow, jcol].text(
                        -0.04, 1.04, source_id + '\n' + panel_label,
                        ha='left', va='top', transform=axs[irow, jcol].transAxes, linespacing=1.6)
                    
                    # plot data
                    plt_data = cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id]['ann'].sel(time=slice('1979', '2014'), source_id=source_id).mean(dim='time') - plt_era5
                    
                    ttest_fdr_res = ttest_fdr_control(
                        cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id]['ann'].sel(time=slice('1979', '2014'), source_id=source_id),
                        plt_era5_ann)
                    plt_data = plt_data.where(ttest_fdr_res, np.nan)
                    
                    plt_mesh = axs[irow, jcol].pcolormesh(
                        plt_data.lon, plt_data.lat, plt_data,
                        norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree())
                    if variable_id=='tos':
                        axs[irow, jcol].add_feature(cfeature.LAND, color='white', zorder=2, edgecolor=None,lw=0)
                    
                else:
                    axs[irow, jcol].set_visible(False)
        
        cbar = fig.colorbar(
            plt_mesh, #cm.ScalarMappable(norm=pltnorm, cmap=pltcmp),
            ax=axs, aspect=40, format=remove_trailing_zero_pos,
            orientation="horizontal", shrink=0.5, ticks=pltticks, extend='both',
            anchor=(0.5, -0.8),)
        cbar.ax.tick_params(length=2, width=0.4)
        cbar.ax.set_xlabel(cbar_label)
        
        fig.subplots_adjust(left=0.01, right=0.996, bottom=fm_bottom, top=0.99)
        fig.savefig(opng)
        plt.close()
        
        del cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id], era5_sl_mon_alltime[var]



# endregion

