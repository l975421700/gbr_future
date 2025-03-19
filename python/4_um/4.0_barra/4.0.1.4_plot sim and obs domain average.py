

# qsub -I -q express -l walltime=2:00:00,ncpus=1,mem=192GB,storage=gdata/v46+gdata/rt52+gdata/ob53+gdata/zv2+scratch/v46


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
    mon_sea_ann,
    regrid,
    cdo_regrid,)

from statistics0 import (
    ttest_fdr_control,)

# endregion


# region plot domain average of mm

cltypes = {
    'hcc': ['Cirrus', 'Cirrostratus', 'Deep convection'],
    'mcc': ['Altocumulus', 'Altostratus', 'Nimbostratus'],
    'lcc': ['Cumulus', 'Stratocumulus', 'Stratus'],
    'tcc': ['Cirrus', 'Cirrostratus', 'Deep convection', 'Altocumulus', 'Altostratus', 'Nimbostratus', 'Cumulus', 'Stratocumulus', 'Stratus']}
extent = [110.58, 157.34, -43.69, -7.01]
min_lon, max_lon, min_lat, max_lat = extent

ds_alltime = {}
ds_mon = {}
dm_mon = {}
dm_mstd = {}
dm_mm = {}
for var2 in ['rsut', 'rlut', 'rsdt']:
    # var2='rsdt'
    # ['cll', 'clm', 'clh', 'clt']
    var1 = cmip6_era5_var[var2]
    print(f'#-------------------------------- {var1} and {var2}')
    
    if var2 in ['cll', 'clm', 'clh', 'clt']:
        obs = 'Himawari'
        with open('/scratch/v46/qg8515/data/obs/jaxa/clp/cltype_frequency_alltime.pkl', 'rb') as f:
            ds_alltime[obs] = pickle.load(f)
        ds_mon[obs] = ds_alltime[obs]['mon'].sel(time=slice('2016', '2023'), types=cltypes[var1], lat=slice(max_lat, min_lat), lon=slice(min_lon, max_lon)).sum(dim='types')
    elif var2 in ['rsut', 'rlut', 'rsdt']:
        obs = 'CERES'
        ds_alltime[obs] = xr.open_dataset('data/obs/CERES/CERES_EBAF-TOA_Ed4.2.1_Subset_200003-202411.nc').sel(time=slice('2016', '2023'))
        ds_alltime[obs] = ds_alltime[obs].rename({
            'toa_sw_all_mon': 'mtuwswrf',
            'toa_lw_all_mon': 'mtnlwrf',
            'solar_mon': 'mtdwswrf'})
        ds_alltime[obs]['mtuwswrf'] *= (-1)
        ds_alltime[obs]['mtnlwrf'] *= (-1)
        ds_mon[obs] = ds_alltime[obs][var1].sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))
    
    with open(f'data/obs/era5/mon/era5_sl_mon_alltime_{var1}.pkl', 'rb') as f:
        ds_alltime['ERA5'] = pickle.load(f)
    with open(f'data/sim/um/barra_r2/barra_r2_mon_alltime_{var2}.pkl','rb') as f:
        ds_alltime['BARRA-R2'] = pickle.load(f)
    with open(f'data/sim/um/barra_c2/barra_c2_mon_alltime_{var2}.pkl','rb') as f:
        ds_alltime['BARRA-C2'] = pickle.load(f)
    
    ds_mon['ERA5'] = ds_alltime['ERA5']['mon'].sel(time=slice('2016', '2023'), lat=slice(max_lat, min_lat), lon=slice(min_lon, max_lon))
    for ids in ['BARRA-R2', 'BARRA-C2']:
        # ids = 'BARRA-R2'
        ds_mon[ids] = ds_alltime[ids]['mon'].sel(time=slice('2016', '2023'), lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))
    
    dss = [obs, 'ERA5', 'BARRA-R2', 'BARRA-C2']
    for ids in dss:
        # ids = obs
        dm_mon[ids] = ds_mon[ids].weighted(np.cos(np.deg2rad(ds_mon[ids].lat))).mean(dim=['lon', 'lat']).compute()
        dm_mstd[ids] = dm_mon[ids].groupby('time.month').std(ddof=1).compute()
        dm_mm[ids] = dm_mon[ids].groupby('time.month').mean().compute()
    
    opng = f'figures/4_um/4.0_barra/4.0.1_domain average/4.0.1.0 dm mm {var2} {obs}, ERA5, BARRA-R2, BARRA-C2.png'
    fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)
    for ids in dss:
        plt1 = ax.plot(month_num, dm_mm[ids], '.-', lw=0.75, markersize=6,
                       label=ids, color=ds_color[ids])
        plt2 = ax.errorbar(month_num, dm_mm[ids], yerr=dm_mstd[ids],
                           fmt='none', capsize=4, color=ds_color[ids],
                           lw=0.5, alpha=0.7)
    if var2 in ['clh', 'rlut', 'rsdt']:
        ax.legend(ncol=1, frameon=True, loc='upper center', handletextpad=0.5)
    elif var2=='clt':
        ax.legend(ncol=1, frameon=True, loc='lower left', handletextpad=0.5)
    elif var2=='rsut':
        ax.legend(ncol=1, frameon=True, loc='lower center', handletextpad=0.5)
    else:
        ax.legend(ncol=1, frameon=True, loc='upper right', handletextpad=0.5)
    
    if var2=='cll':
        ax.set_ylim(15, 36)
        ax.set_yticks(np.arange(16, 36+1e-4, 2))
    elif var2=='clm':
        ax.set_ylim(9, 24)
        ax.set_yticks(np.arange(10, 24+1e-4, 2))
    elif var2=='clh':
        ax.set_ylim(5, 45)
        ax.set_yticks(np.arange(5, 45+1e-4, 5))
    elif var2=='clt':
        ax.set_ylim(35, 67)
        ax.set_yticks(np.arange(35, 67+1e-4, 5))
    ax.set_xticks(month_num)
    ax.set_xticklabels(month_jan)
    if var2 in ['rsut', 'rlut']:
        ax.set_ylabel(f'Area-weighted mean {era5_varlabels[var1]}', size=9)
    elif var2 in ['rsdt']:
        ax.set_ylabel(f'Area-weighted mean {era5_varlabels[var1]}', size=8)
    else:
        ax.set_ylabel(f'Area-weighted mean {era5_varlabels[var1]}')
    ax.yaxis.set_major_formatter(remove_trailing_zero_pos)
    ax.grid(True, which='both', linewidth=0.5, color='gray',
            alpha=0.5, linestyle='--')
    if var2 in ['rsut', 'rlut', 'rsdt']:
        fig.subplots_adjust(left=0.16, right=0.99, bottom=0.08, top=0.98)
    else:
        fig.subplots_adjust(left=0.14, right=0.99, bottom=0.08, top=0.98)
    fig.savefig(opng)
    
    for ids in dss:
        del ds_alltime[ids], ds_mon[ids]




# endregion


# region plot RMSE

# endregion



