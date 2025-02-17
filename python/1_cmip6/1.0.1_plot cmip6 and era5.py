

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
    
    for table_id, variable_id in zip(['Amon', 'Amon', 'Amon', 'Amon', 'Omon'], ['tas', 'rsut', 'rsdt', 'rlut', 'tos']):
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


# endregion

