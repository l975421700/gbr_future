

# qsub -I -q normal -P nf33 -l walltime=3:00:00,ncpus=1,mem=192GB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22+gdata/qx55


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
from metpy.calc import pressure_to_height_std, geopotential_to_height, potential_temperature, equivalent_potential_temperature, specific_humidity_from_mixing_ratio, wind_components, relative_humidity_from_specific_humidity, dewpoint_from_specific_humidity
from metpy.units import units
import metpy.calc as mpcalc
import pickle
from datetime import datetime
from skimage.measure import block_reduce
from netCDF4 import Dataset
import xesmf as xe
import healpy as hp

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
import matplotlib.ticker as ticker
import seaborn as sns

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
    era5_varlabels_sim,
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
    get_nn_lon_lat_index,
)

from calculations import (
    time_weighted_mean,
    mon_sea_ann,
    regrid,
    cdo_regrid,)

from statistics0 import (
    ttest_fdr_control,)

# endregion


# region plot hourly var at Willis Island

periods = ['am', 'sea', 'mon']
vars = ['pr']
# ['pr', 'ts', 'clh', 'clm', 'cll', 'clt', 'evspsbl', 'hfls', 'hfss', 'psl', 'rlds', 'rldscs', 'rlus', 'rluscs', 'rlut', 'rlutcs', 'rsds', 'rsdscs', 'rsdt', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'sfcWind', 'tas', 'uas', 'vas', 'clivi', 'clwvi']
# 'hurs', 'huss',
dss = ['ERA5', 'BARRA-R2', 'BARRA-C2', 'UM']
station = 'Willis Island'
slat = -16.288
slon = 149.965
years = 2020
yeare = 2021

izlev=8
if 'UM' in dss:
    hk_um_z10_1H = xr.open_zarr(f'/g/data/qx55/uk_node/glm.n2560_RAL3p3/data.healpix.PT1H.z{izlev}.zarr', chunks={'time': 10489, 'cell': 20})

for var2 in vars:
    var1 = cmip6_era5_var[var2]
    # var2='pr'; var1='tp'
    print(f'#-------------------------------- {var2} vs. {var1}')
    ds = {}
    
    time1 = time.perf_counter()
    try:
        if var1 in ['t2m', 'si10', 'd2m', 'u10', 'v10', 'u100', 'v100']:
            if var1 == 't2m': vart='2t'
            if var1 == 'si10': vart='10si'
            if var1 == 'd2m': vart='2d'
            if var1 == 'u10': vart='10u'
            if var1 == 'v10': vart='10v'
            if var1 == 'u100': vart='100u'
            if var1 == 'v100': vart='100v'
            ds['ERA5'] = xr.open_mfdataset(
                sorted([file for iyear in np.arange(years, yeare + 1, 1)
                        for file in glob.glob(f'/g/data/rt52/era5/single-levels/reanalysis/{vart}/{iyear}/*.nc')]),
                parallel=True,
                preprocess=lambda ds: ds[var1].sel(longitude=slon, latitude=slat, method='nearest').compute()
                )[var1].sel(time=slice('2020-03', '2021-02'))
        else:
            ds['ERA5'] = xr.open_mfdataset(
                sorted([file for iyear in np.arange(years, yeare + 1, 1)
                        for file in glob.glob(f'/g/data/rt52/era5/single-levels/reanalysis/{var1}/{iyear}/*.nc')]),
                parallel=True,
                preprocess=lambda ds: ds[var1].sel(longitude=slon, latitude=slat, method='nearest').compute()
                )[var1].sel(time=slice('2020-03', '2021-02'))
        if var1 in ['tp', 'e', 'cp', 'lsp', 'pev']:
            ds['ERA5'] *= 1000
        elif var1 in ['msl']:
            ds['ERA5'] /= 100
        elif var1 in ['sst', 't2m', 'd2m', 'skt']:
            ds['ERA5'] -= zerok
        elif var1 in ['hcc', 'mcc', 'lcc', 'tcc']:
            ds['ERA5'] *= 100
        elif var1 in ['z']:
            ds['ERA5'] /= 9.80665
        elif var1 in ['mper']:
            ds['ERA5'] *= 3600
    except OSError:
        pass
    time2 = time.perf_counter()
    print(f'Execution time: {time2 - time1:.1f} s')
    
    time1 = time.perf_counter()
    try:
        ds['BARRA-R2'] = xr.open_mfdataset(
            sorted([file for iyear in np.arange(years, yeare + 1, 1)
                    for file in glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/{var2}/latest/*{iyear}??.nc')]),
            parallel=True,
            preprocess=lambda ds: ds[var2].sel(lon=slon, lat=slat, method='nearest').compute()
            )[var2].sel(time=slice('2020-03', '2021-02'))
        if var2 in ['pr', 'evspsbl', 'evspsblpot']:
            ds['BARRA-R2'] *= 3600
        elif var2 in ['tas', 'ts']:
            ds['BARRA-R2'] -= zerok
        elif var2 in ['rlus', 'rluscs', 'rlut', 'rlutcs', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'hfls', 'hfss']:
            ds['BARRA-R2'] *= (-1)
        elif var2 in ['psl']:
            ds['BARRA-R2'] /= 100
        elif var2 in ['huss']:
            ds['BARRA-R2'] *= 1000
    except OSError:
        pass
    time2 = time.perf_counter()
    print(f'Execution time: {time2 - time1:.1f} s')
    
    time1 = time.perf_counter()
    try:
        ds['BARRA-C2'] = xr.open_mfdataset(
            sorted([file for iyear in np.arange(years, yeare + 1, 1)
                    for file in glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/{var2}/latest/*{iyear}??.nc')]),
            parallel=True,
            preprocess=lambda ds: ds[var2].sel(lon=slon, lat=slat, method='nearest').compute()
            )[var2].sel(time=slice('2020-03', '2021-02'))
        if var2 in ['pr', 'evspsbl', 'evspsblpot']:
            ds['BARRA-C2'] *= 3600
        elif var2 in ['tas', 'ts']:
            ds['BARRA-C2'] -= zerok
        elif var2 in ['rlus', 'rluscs', 'rlut', 'rlutcs', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'hfls', 'hfss']:
            ds['BARRA-C2'] *= (-1)
        elif var2 in ['psl']:
            ds['BARRA-C2'] /= 100
        elif var2 in ['huss']:
            ds['BARRA-C2'] *= 1000
    except OSError:
        pass
    time2 = time.perf_counter()
    print(f'Execution time: {time2 - time1:.1f} s')
    
    time1 = time.perf_counter()
    if 'UM' in dss:
        try:
            ds['UM'] = hk_um_z10_1H[var2].sel(cell=hp.ang2pix(2**izlev, slon, slat, nest=True, lonlat=True), time=slice('2020-03', '2021-02')).compute()
            if var2 in ['pr', 'evspsbl', 'evspsblpot']:
                ds['UM'] *= 3600
            elif var2 in ['tas', 'ts']:
                ds['UM'] -= zerok
            elif var2 in ['rlus', 'rluscs', 'rlut', 'rlutcs', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'hfls', 'hfss']:
                ds['UM'] *= (-1)
            elif var2 in ['psl']:
                ds['UM'] /= 100
            elif var2 in ['huss']:
                ds['UM'] *= 1000
            elif var2 in ['clt']:
                ds['UM'] *= 100
        except KeyError:
            pass
    time2 = time.perf_counter()
    print(f'Execution time: {time2 - time1:.1f} s')
    
    opng = f'figures/4_um/4.0_barra/4.0.6_site_analysis/4.0.6.3 {years}-{yeare} {var2} in {', '.join(list(ds.keys()))} at {station}.png'
    
    fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)
    
    xlog = var2 == 'pr'
    
    if var2=='pr':
        lowerbound = 1e-3
    elif var2 in ['clh', 'clm', 'cll', 'clt']:
        lowerbound = 0
    else:
        lowerbound = -10**4
    
    for ids in list(ds.keys()):
        # print(f'#---------------- {ids}')
        ax.axvline(x=np.mean(ds[ids]), color=ds_color[ids], linestyle='--', linewidth=0.5)
    ax = sns.histplot(
        data=pd.concat([pd.DataFrame({
            'x': ds[ids][ds[ids] > lowerbound],
            'hue': ids}) for ids in list(ds.keys())]),
        x='x', hue='hue', linewidth=1, ax=ax,
        log_scale=xlog, bins=30, element='step', fill=False,
        palette = {ids: ds_color[ids] for ids in list(ds.keys())},
        legend=False)
    
    if var2 == 'pr':
        ax.set_xscale('log')
        ax.set_xticks(10 ** np.arange(-3, 2+1e-4, 1))
        ax.set_xlim(10**-3, 60)
    elif var2 == 'ts':
        ax.set_xticks(np.arange(24, 32+1))
        ax.set_xlim(24, 32)
    
    custom_lines = [Line2D([0],[0],color=ds_color[ids],lw=1)
                    for ids in list(ds.keys())]
    ax.legend(custom_lines,
              [f'{ids}: {str(np.round(np.mean(ds[ids]).values, 2))}'
               for ids in list(ds.keys())],
              fontsize=10, handlelength=0.8, handletextpad=0.6)
    
    ax.grid(lw=0.5, alpha=0.5, ls='--')
    ax.set_xlabel(era5_varlabels[var1].replace('day^{-1}', 'hour^{-1}'),
                  labelpad=2)
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.14, top=0.99)
    fig.savefig(opng)
    plt.close()



'''
for ids in dss:
    print(f'{ids}: {str(np.round(np.mean(ds[ids]).values, 3))}')
# ERA5: 0.133
# BARRA-R2: 0.148
# BARRA-C2: 0.137
# UM: 0.086

    # # plot them together
    # for ids in dss:
    #     print(f'#---------------- {ids}')
    #     ds[ids][ds[ids] <= 1e-3] = np.nan
    # sns.histplot(data=pd.DataFrame({f'{ids}': ds[ids] for ids in dss}), ax=ax,
    #              log_scale=xlog,
    #              bins=30,
    #              element='step', fill=False,
    #              stat='count',
    #              kde=True,
    #              legend=True
    #              )
        # kde=True, line_kws={'linewidth': 0.5}

    # for ids in dss:
    #     print(f'#---------------- {ids}')
    #     # ds[ids] = ds[ids][ds[ids] > 1e-3]
    #     ds[ids][ds[ids] <= 1e-3] = np.nan
    #     sns.histplot(data=ds[ids].values, ax=ax, label=ids,
    #                  log_scale=xlog,
    #                  bins=30,
    #                  element='step', fill=False, color=ds_color[ids],
    #                  stat='count',
    #                  kde=True,
    #                  )

                    #  common_bins=False,
                    #  cumulative=True,

    for ids in dss:
        print(f'#---------------- {ids}')
        print(np.sum(ds[ids].values))
        print(np.sum(ds[ids][ds[ids] <= 1e-3].values))
        print((np.sum(ds[ids][ds[ids] <= 1e-3].values) / np.sum(ds[ids].values)) * 100)

    for ids in dss:
        print(f'#---------------- {ids}')
        print(f'{str(np.round(np.min(ds[ids].values), 2))}, {str(np.round(np.mean(ds[ids].values), 2))}, {str(np.round(np.max(ds[ids].values), 1))}')
'''
# endregion

