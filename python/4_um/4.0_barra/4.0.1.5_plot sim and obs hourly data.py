

# qsub -I -q express -l walltime=4:00:00,ncpus=1,mem=192GB,storage=gdata/v46+gdata/rt52+gdata/ob53+gdata/zv2+scratch/v46


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


# region plot dm am

extent = [110.58, 157.34, -43.69, -7.01]
min_lon, max_lon, min_lat, max_lat = extent

for var in ['clh', 'clm', 'clt', 'pr', 'tas']:
    # ['cll', 'clh', 'clm', 'clt', 'pr', 'tas']
    # var = 'cll'
    print(f'#-------------------------------- {var}')
    
    with open(f'data/sim/um/barra_c2/barra_c2_hourly_alltime_{var}.pkl','rb') as f:
        ds_hourly_alltime = pickle.load(f)
    
    ds_ann = ds_hourly_alltime['ann'].sel(time=slice('2016', '2023'), lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))
    ds_ann_dm = ds_ann.weighted(np.cos(np.deg2rad(ds_ann.lat))).mean(dim=['lon', 'lat'])
    ds_dm_am = ds_ann_dm.mean(dim='time')
    ds_dm_astd = ds_ann_dm.std(dim='time', ddof=1)
    
    opng = f'figures/4_um/4.0_barra/4.0.1_domain average/4.0.1.0 dm am hourly {var} BARRA-C2.png'
    fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)
    ax.plot(range(0, 24, 1), ds_dm_am, '.-', lw=0.75, markersize=6,
            label='BARRA-C2', color=ds_color['BARRA-C2'])
    ax.errorbar(range(0, 24, 1), ds_dm_am, yerr=ds_dm_astd, fmt='none',
                capsize=4, color=ds_color['BARRA-C2'],lw=0.5, alpha=0.7)
    ax.legend(ncol=1, frameon=True, loc='upper center', handletextpad=0.5)
    
    ax.set_xticks(np.arange(0, 24, 6))
    ax.set_ylabel(f'Area-weighted mean {era5_varlabels[cmip6_era5_var[var]]}', size=9)
    ax.set_xlabel('UTC', size=9)
    ax.yaxis.set_major_formatter(remove_trailing_zero_pos)
    ax.grid(True, which='both', linewidth=0.5, color='gray',
            alpha=0.5, linestyle='--')
    fig.subplots_adjust(left=0.16, right=0.99, bottom=0.13, top=0.98)
    fig.savefig(opng)
    
    del ds_hourly_alltime




'''
    fl = sorted(glob.glob(f'scratch/data/sim/um/barra_c2/{var}/{var}_hourly_*.nc'))[-96:]
    barra_c2_hourly = xr.open_mfdataset(fl)[var].sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))
    
    barra_c2_hourly_dm = barra_c2_hourly.weighted(np.cos(np.deg2rad(barra_c2_hourly.lat))).mean(dim=['lon', 'lat'])
    
    barra_c2_hourly_dm_ann = barra_c2_hourly_dm.groupby('time.year').map(time_weighted_mean).compute()
    barra_c2_hourly_dm_am = barra_c2_hourly_dm_ann.mean(dim='year')
    barra_c2_hourly_dm_astd = barra_c2_hourly_dm_ann.std(dim='year', ddof=1)
    
    print(np.max(np.abs(ds_dm_am - barra_c2_hourly_dm_am)))
    print(np.max(np.abs(ds_dm_astd - barra_c2_hourly_dm_astd)))

'''
# endregion


# region plot hourly am

mpl.rc('font', family='Times New Roman', size=10)
extent = [110.58, 157.34, -43.69, -7.01]
min_lon, max_lon, min_lat, max_lat = extent
panelh = 4
panelw = 4.4
nrow = 4
ncol = 6
fm_bottom = 1.4 / (panelh*nrow + 1.4)

for ids in ['BARRA-C2']:
    # ids = 'BARRA-C2'
    print(f'#-------------------------------- {ids}')
    for var in ['cll']:
        # var = 'cll'
        print(f'#---------------- {var}')
        if ids=='BARRA-C2':
            with open(f'data/sim/um/barra_c2/barra_c2_hourly_alltime_{var}.pkl','rb') as f:
                ds_hourly_alltime = pickle.load(f)
        
        ds_am = ds_hourly_alltime['ann'].sel(time=slice('2016', '2023'), lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat)).mean(dim='time')
        
        if var in ['cll', 'clm', 'clh']:
            extend='max'
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=60, cm_interval1=5, cm_interval2=10, cmap='Blues_r',)
        elif var in ['clt']:
            extend='neither'
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=100, cm_interval1=5, cm_interval2=10, cmap='Blues_r',)
        
        opng = f'figures/4_um/4.0_barra/4.0.0_whole region/4.0.0.2 am hourly {var} {ids}.png'
        cbar_label = f'2016-2023 {ids} {era5_varlabels[cmip6_era5_var[var]]}'
        
        fig, axs = plt.subplots(
            nrow, ncol, figsize=np.array([panelw*ncol, panelh*nrow+1.4])/2.54,
            subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
            gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)
        
        for irow in range(nrow):
            for jcol in range(ncol):
                # irow=0; jcol=0
                print(f'#---------------- {irow} {jcol} {irow*ncol+jcol}')
                axs[irow, jcol] = regional_plot(extent=extent, central_longitude=180, ax_org=axs[irow, jcol])
                
                plt_data = ds_am[ds_am.hour==(irow*ncol+jcol)].squeeze()
                plt_mesh = axs[irow, jcol].pcolormesh(
                    plt_data.lon, plt_data.lat, plt_data,
                    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
                
                plt_mean = plt_data.weighted(np.cos(np.deg2rad(plt_data.lat))).mean().values
                if ((irow==0) & (jcol==0)):
                    plt_text = f'({string.ascii_lowercase[irow]}{jcol+1}) {(irow*ncol+jcol):02d}:00 UTC Mean: {str(np.round(plt_mean, 1))}'
                else:
                    plt_text = f'({string.ascii_lowercase[irow]}{jcol+1}) {(irow*ncol+jcol):02d}:00 UTC {str(np.round(plt_mean, 1))}'
                # plt_text = f'({string.ascii_lowercase[irow]}{jcol+1}) {(irow*ncol+jcol):02d}:00 UTC'
                plt.text(
                    0, 1.02, plt_text,
                    transform=axs[irow, jcol].transAxes,
                    ha='left', va='bottom', rotation='horizontal')
        
        cbar = fig.colorbar(
            cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), #plt_mesh, #
            format=remove_trailing_zero_pos,
            orientation="horizontal", ticks=pltticks, extend=extend,
            cax=fig.add_axes([0.25, fm_bottom-0.01, 0.5, 0.02]))
        cbar.ax.set_xlabel(cbar_label)
        fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom, top = 0.98)
        fig.savefig(opng)



# endregion

