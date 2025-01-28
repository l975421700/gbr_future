

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


# region plot global era5 am data

era5_gridarea = xr.open_dataset('data/obs/era5/era5_gridarea.nc').cell_area

era5_sl_mon_alltime = {}
for var in ['e']:
    # var = 'tcw'
    # 'tp', 'msl', 'sst', 'hcc', 'mcc', 'lcc', 'tcc', 't2m', 'wind', 'mslhf', 'msnlwrf', 'msnswrf', 'msshf', 'mtdwswrf', 'mtnlwrf', 'mtnswrf', 'msdwlwrf', 'msdwswrf', 'msdwlwrfcs', 'msdwswrfcs', 'msnlwrfcs', 'msnswrfcs', 'mtnlwrfcs', 'mtnswrfcs', 'cbh', 'e', 'z', 'tcw', 'tcwv', 'tcsw', 'tcrw', 'tcslw', 'tciw', 'tclw'
    print(var)
    with open(f'data/obs/era5/mon/era5_sl_mon_alltime_{var}.pkl', 'rb') as f:
        era5_sl_mon_alltime[var] = pickle.load(f)
    print(era5_sl_mon_alltime[var]['mon'].units)
    
    if var=='tp':
        plt_data = era5_sl_mon_alltime[var]['am'].squeeze() * 1000
        plt_data_gm = np.average(plt_data, weights=era5_gridarea)
        cbar_label = r'ERA5 annual mean precipitation (1979-2023) [$mm \; day^{-1}$]' + '\nglobal mean: ' + str(np.round(plt_data_gm, 3)) + r' $mm \; day^{-1}$'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel = np.array([0, 0.1, 0.2, 0.5, 1, 2, 4, 6, 8, 10, 12, 16, 20,])
        pltticks = np.array([0, 0.1, 0.2, 0.5, 1, 2, 4, 6, 8, 10, 12, 16, 20,])
        pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
        pltcmp = plt.get_cmap('viridis_r', len(pltlevel)-1)
        
        extend = 'max'
    elif var=='msl':
        plt_data = era5_sl_mon_alltime[var]['am'].squeeze() / 100
        plt_data_gm = np.average(plt_data, weights=era5_gridarea)
        cbar_label = r'ERA5 annual mean sea level pressure (1979-2023) [$hPa$]' + '\nglobal mean: ' + str(int(plt_data_gm)) + r' $hPa$'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=980, cm_max=1025, cm_interval1=2.5, cm_interval2=5, cmap='viridis_r',)
        
        extend = 'both'
    elif var=='sst':
        plt_data = era5_sl_mon_alltime[var]['am'].squeeze() - zerok
        cbar_label = r'ERA5 annual mean sea surface temperature (1979-2023) [$°C$]'
        
        print(stats.describe(plt_data.values, axis=None, nan_policy='omit'))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-2, cm_max=30, cm_interval1=2, cm_interval2=4, cmap='viridis_r',)
        
        extend = 'both'
    elif var=='hcc':
        plt_data = era5_sl_mon_alltime[var]['am'].squeeze() * 100
        plt_data_gm = np.average(plt_data, weights=era5_gridarea)
        cbar_label = r'ERA5 annual mean high cloud cover (1979-2023) [$\%$]' + '\nglobal mean: ' + str(np.round(plt_data_gm, 1)) + r' $\%$'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=10, cmap='viridis_r',)
        
        extend = 'neither'
    elif var=='mcc':
        plt_data = era5_sl_mon_alltime[var]['am'].squeeze() * 100
        plt_data_gm = np.average(plt_data, weights=era5_gridarea)
        cbar_label = r'ERA5 annual mean medium cloud cover (1979-2023) [$\%$]' + '\nglobal mean: ' + str(np.round(plt_data_gm, 1)) + r' $\%$'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=10, cmap='viridis_r',)
        
        extend = 'neither'
    elif var=='lcc':
        plt_data = era5_sl_mon_alltime[var]['am'].squeeze() * 100
        plt_data_gm = np.average(plt_data, weights=era5_gridarea)
        cbar_label = r'ERA5 annual mean low cloud cover (1979-2023) [$\%$]' + '\nglobal mean: ' + str(np.round(plt_data_gm, 1)) + r' $\%$'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=10, cmap='viridis_r',)
        
        extend = 'neither'
    elif var=='tcc':
        plt_data = era5_sl_mon_alltime[var]['am'].squeeze() * 100
        plt_data_gm = np.average(plt_data, weights=era5_gridarea)
        cbar_label = r'ERA5 annual mean total cloud cover (1979-2023) [$\%$]' + '\nglobal mean: ' + str(np.round(plt_data_gm, 1)) + r' $\%$'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=10, cmap='viridis_r',)
        
        extend = 'neither'
    elif var=='t2m':
        plt_data = era5_sl_mon_alltime[var]['am'].squeeze() - zerok
        cbar_label = r'ERA5 annual mean 2 m temperature (1979-2023) [$°C$]'
        
        print(stats.describe(plt_data.values, axis=None, nan_policy='omit'))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-32, cm_max=32, cm_interval1=2, cm_interval2=8, cmap='viridis_r',)
        
        extend = 'both'
    elif var=='wind':
        plt_data = era5_sl_mon_alltime[var]['am'].squeeze()
        cbar_label = r'ERA5 annual mean 10 m wind speed (1979-2023) [$m\;s^{-1}$]'
        
        print(stats.describe(plt_data.values, axis=None, nan_policy='omit'))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=2, cm_max=12, cm_interval1=1, cm_interval2=1, cmap='viridis_r',)
        
        extend = 'both'
    elif var=='mslhf':
        plt_data = era5_sl_mon_alltime[var]['am'].squeeze()
        plt_data_gm = np.average(plt_data, weights=era5_gridarea)
        cbar_label = r'ERA5 annual mean surface latent heat flux (1979-2023) [$W \; m^{-2}$]' + '\nglobal mean: ' + str(np.round(plt_data_gm, 1)) + r' $W \; m^{-2}$'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-280, cm_max=0, cm_interval1=20, cm_interval2=40, cmap='viridis',)
        
        extend = 'both'
    elif var=='msshf':
        plt_data = era5_sl_mon_alltime[var]['am'].squeeze()
        plt_data_gm = np.average(plt_data, weights=era5_gridarea)
        cbar_label = r'ERA5 annual mean surface sensible heat flux (1979-2023) [$W \; m^{-2}$]' + '\nglobal mean: ' + str(np.round(plt_data_gm, 1)) + r' $W \; m^{-2}$'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-120, cm_max=60, cm_interval1=10, cm_interval2=20, cmap='BrBG', asymmetric=True,)
        
        extend = 'both'
    elif var=='msnlwrf':
        plt_data = era5_sl_mon_alltime[var]['am'].squeeze()
        plt_data_gm = np.average(plt_data, weights=era5_gridarea)
        cbar_label = r'ERA5 annual mean surface net longwave radiation flux (1979-2023) [$W \; m^{-2}$]' + '\nglobal mean: ' + str(np.round(plt_data_gm, 1)) + r' $W \; m^{-2}$'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-160, cm_max=0, cm_interval1=10, cm_interval2=20, cmap='viridis',)
        
        extend = 'both'
    elif var=='msnswrf':
        plt_data = era5_sl_mon_alltime[var]['am'].squeeze()
        plt_data_gm = np.average(plt_data, weights=era5_gridarea)
        cbar_label = r'ERA5 annual mean surface net shortwave radiation flux (1979-2023) [$W \; m^{-2}$]' + '\nglobal mean: ' + str(np.round(plt_data_gm, 1)) + r' $W \; m^{-2}$'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=260, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
        
        extend = 'both'
    elif var=='mtdwswrf':
        plt_data = era5_sl_mon_alltime[var]['am'].squeeze()
        plt_data_gm = np.average(plt_data, weights=era5_gridarea)
        cbar_label = r'ERA5 annual mean TOA downward shortwave radiation flux (1979-2023) [$W \; m^{-2}$]' + '\nglobal mean: ' + str(np.round(plt_data_gm, 1)) + r' $W \; m^{-2}$'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=180, cm_max=420, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
        
        extend = 'both'
    elif var=='mtnlwrf':
        plt_data = era5_sl_mon_alltime[var]['am'].squeeze()
        plt_data_gm = np.average(plt_data, weights=era5_gridarea)
        cbar_label = r'ERA5 annual mean TOA net longwave radiation flux (1979-2023) [$W \; m^{-2}$]' + '\nglobal mean: ' + str(np.round(plt_data_gm, 1)) + r' $W \; m^{-2}$'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-300, cm_max=-120, cm_interval1=10, cm_interval2=20, cmap='viridis',)
        
        extend = 'both'
    elif var=='mtnswrf':
        plt_data = era5_sl_mon_alltime[var]['am'].squeeze()
        plt_data_gm = np.average(plt_data, weights=era5_gridarea)
        cbar_label = r'ERA5 annual mean TOA net shortwave radiation flux (1979-2023) [$W \; m^{-2}$]' + '\nglobal mean: ' + str(np.round(plt_data_gm, 1)) + r' $W \; m^{-2}$'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=50, cm_max=350, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
        
        extend = 'both'
    elif var=='msdwlwrf':
        plt_data = era5_sl_mon_alltime[var]['am'].squeeze()
        plt_data_gm = np.average(plt_data, weights=era5_gridarea)
        cbar_label = r'ERA5 annual mean surface downward longwave radiation flux (1979-2023) [$W \; m^{-2}$]' + '\nglobal mean: ' + str(np.round(plt_data_gm, 1)) + r' $W \; m^{-2}$'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=90, cm_max=430, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
        
        extend = 'both'
    elif var=='msdwswrf':
        plt_data = era5_sl_mon_alltime[var]['am'].squeeze()
        plt_data_gm = np.average(plt_data, weights=era5_gridarea)
        cbar_label = r'ERA5 annual mean surface downward shortwave radiation flux (1979-2023) [$W \; m^{-2}$]' + '\nglobal mean: ' + str(np.round(plt_data_gm, 1)) + r' $W \; m^{-2}$'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=60, cm_max=300, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
        
        extend = 'both'
    elif var=='e':
        plt_data = era5_sl_mon_alltime[var]['am'].squeeze() * 1000
        plt_data_gm = np.average(plt_data, weights=era5_gridarea)
        cbar_label = r'ERA5 annual mean evaporation (1979-2023) [$mm \; day^{-1}$]' + '\nglobal mean: ' + str(np.round(plt_data_gm, 3)) + r' $mm \; day^{-1}$'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel = np.array([-10, -8, -6, -4, -2, -1, -0.5, -0.2, -0.1, 0, 0.1])
        pltticks = np.array([-10, -8, -6, -4, -2, -1, -0.5, -0.2, -0.1, 0, 0.1])
        pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
        pltcmp = plt.get_cmap('viridis', len(pltlevel)-1)
        
        extend = 'both'
    elif var=='z':
        plt_data = era5_sl_mon_alltime[var]['am'].squeeze() / 9.80665
        cbar_label = r'ERA5 orography [$m$]'
        
        print(stats.describe(plt_data.values, axis=None, nan_policy='omit'))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=6000, cm_interval1=250, cm_interval2=500, cmap='viridis_r',)
        
        extend = 'max'
    elif var=='tcw':
        plt_data = era5_sl_mon_alltime[var]['am'].squeeze()
        plt_data_gm = np.average(plt_data, weights=era5_gridarea)
        cbar_label = r'ERA5 annual mean total column water (1979-2023) [$kg \; m^{-2}$]' + '\nglobal mean: ' + str(np.round(plt_data_gm, 1)) + r' $kg \; m^{-2}$'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=60, cm_interval1=3, cm_interval2=6, cmap='viridis_r',)
        
        extend = 'max'
    
    fig, ax = globe_plot(figsize=np.array([12, 8]) / 2.54, fm_bottom=0.13)
    
    plt_mesh1 = ax.pcolormesh(
        plt_data.lon, plt_data.lat, plt_data.values,
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),zorder=1,)
    
    if var in ['sst', 'wind']:
        ax.add_feature(cfeature.LAND,color='white',zorder=1,edgecolor=None,lw=0)
    if var in ['z']:
        ax.add_feature(cfeature.OCEAN,color='white',zorder=1,edgecolor=None,lw=0)
    
    cbar = fig.colorbar(
        plt_mesh1, ax=ax, aspect=40, format=remove_trailing_zero_pos,
        orientation="horizontal", shrink=0.8, ticks=pltticks, extend=extend,
        pad=0.02, fraction=0.13,)
    cbar.ax.set_xlabel(cbar_label, ha='center', linespacing=1.3, labelpad=4)
    
    fig.savefig(f'figures/5_era5/5.0_global/5.0.0 global era5 annual mean {var}.png')
    
    del era5_sl_mon_alltime[var]


# endregion


# region plot barra-c2 am data

barra_c2_mon_alltime = {}
for var in ['pr']:
    # 'pr', 'clh', 'clm', 'cll', 'clt', 'evspsbl', 'hfls', 'hfss', 'tas', 'ts', 'rlds', 'rldscs', 'rlus', 'rluscs', 'rlut', 'rlutcs', 'rsds', 'rsdscs', 'rsdt', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'sfcWind', 'psl',
    # var = 'rsut'
    print(var)
    with open(f'data/sim/um/barra_c2/barra_c2_mon_alltime_{var}.pkl','rb') as f:
        barra_c2_mon_alltime[var] = pickle.load(f)
    print(barra_c2_mon_alltime[var]['mon'])
    print(barra_c2_mon_alltime[var]['mon'].units)
    
    if var=='pr':
        plt_data = barra_c2_mon_alltime[var]['am'].squeeze() * seconds_per_d
        cbar_label = 'BARRA-C2 annual mean precipitation\n' + r'(1979-2023) [$mm \; day^{-1}$]'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel = np.array([0, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20,])
        pltticks = np.array([0, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20,])
        pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
        pltcmp = plt.get_cmap('viridis_r', len(pltlevel)-1)
        
        extend = 'max'
    elif var=='clh':
        plt_data = barra_c2_mon_alltime[var]['am'].squeeze()
        cbar_label = 'BARRA-C2 annual mean high cloud cover\n' + r'(1979-2023) [$\%$]'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=10, cmap='viridis_r',)
        
        extend = 'neither'
    elif var=='clm':
        plt_data = barra_c2_mon_alltime[var]['am'].squeeze()
        cbar_label = 'BARRA-C2 annual mean medium cloud cover\n' + r'(1979-2023) [$\%$]'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=10, cmap='viridis_r',)
        
        extend = 'neither'
    elif var=='cll':
        plt_data = barra_c2_mon_alltime[var]['am'].squeeze()
        cbar_label = 'BARRA-C2 annual mean low cloud cover\n' + r'(1979-2023) [$\%$]'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=10, cmap='viridis_r',)
        
        extend = 'neither'
    elif var=='clt':
        plt_data = barra_c2_mon_alltime[var]['am'].squeeze()
        cbar_label = 'BARRA-C2 annual mean total cloud cover\n' + r'(1979-2023) [$\%$]'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=10, cmap='viridis_r',)
        
        extend = 'neither'
    elif var=='evspsbl':
        plt_data = barra_c2_mon_alltime[var]['am'].squeeze() * seconds_per_d
        cbar_label = 'BARRA-C2 annual mean evaporation\n' + r'(1979-2023) [$mm \; day^{-1}$]'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel = np.array([0, 0.1, 0.2, 0.5, 1, 2, 4, 6, 8])
        pltticks = np.array([0, 0.1, 0.2, 0.5, 1, 2, 4, 6, 8])
        pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
        pltcmp = plt.get_cmap('viridis_r', len(pltlevel)-1)
        
        extend = 'max'
    elif var=='hfls':
        plt_data = barra_c2_mon_alltime[var]['am'].squeeze()
        cbar_label = 'BARRA-C2 annual mean surface upward latent heat flux\n' + r'(1979-2023) [$W \; m^{-2}$]'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=280, cm_interval1=20, cm_interval2=40, cmap='viridis_r',)
        
        extend = 'both'
    elif var=='hfss':
        plt_data = barra_c2_mon_alltime[var]['am'].squeeze()
        cbar_label = 'BARRA-C2 annual mean surface upward sensible heat flux\n' + r'(1979-2023) [$W \; m^{-2}$]'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-60, cm_max=120, cm_interval1=10, cm_interval2=20, cmap='BrBG_r', asymmetric=True,)
        
        extend = 'both'
    elif var=='tas':
        plt_data = barra_c2_mon_alltime[var]['am'].squeeze() - zerok
        cbar_label = 'BARRA-C2 annual mean near-surface air temperature\n' + r'(1979-2023) [$°C$]'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=4, cm_max=28, cm_interval1=2, cm_interval2=4, cmap='viridis_r',)
        
        extend = 'both'
    elif var=='ts':
        plt_data = barra_c2_mon_alltime[var]['am'].squeeze() - zerok
        cbar_label = 'BARRA-C2 annual mean surface temperature\n' + r'(1979-2023) [$W \; m^{-2}$]'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=4, cm_max=32, cm_interval1=2, cm_interval2=4, cmap='viridis_r',)
        
        extend = 'both'
    elif var=='rlds':
        plt_data = barra_c2_mon_alltime[var]['am'].squeeze()
        cbar_label = 'BARRA-C2 annual mean surface downward longwave radiation\n' + r'(1979-2023) [$°C$]'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=280, cm_max=440, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
        
        extend = 'both'
    elif var=='rlus':
        plt_data = barra_c2_mon_alltime[var]['am'].squeeze()
        cbar_label = 'BARRA-C2 annual mean surface upward longwave radiation\n' + r'(1979-2023) [$W \; m^{-2}$]'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=340, cm_max=490, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
        
        extend = 'both'
    elif var=='rlut':
        plt_data = barra_c2_mon_alltime[var]['am'].squeeze()
        cbar_label = 'BARRA-C2 annual mean TOA outgoing longwave radiation\n' + r'(1979-2023) [$W \; m^{-2}$]'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=180, cm_max=280, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
        
        extend = 'both'
    elif var=='rsds':
        plt_data = barra_c2_mon_alltime[var]['am'].squeeze()
        cbar_label = 'BARRA-C2 annual mean surface downward shortwave radiation\n' + r'(1979-2023) [$W \; m^{-2}$]'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=90, cm_max=270, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
        
        extend = 'both'
    elif var=='rsdt':
        plt_data = barra_c2_mon_alltime[var]['am'].squeeze()
        cbar_label = 'BARRA-C2 annual mean TOA incident shortwave radiation\n' + r'(1979-2023) [$W \; m^{-2}$]'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=310, cm_max=410, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
        
        extend = 'both'
    elif var=='rsus':
        plt_data = barra_c2_mon_alltime[var]['am'].squeeze()
        cbar_label = 'BARRA-C2 annual mean surface upwelling shortwave radiation\n' + r'(1979-2023) [$W \; m^{-2}$]'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=110, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
        
        extend = 'both'
    elif var=='rsut':
        plt_data = barra_c2_mon_alltime[var]['am'].squeeze()
        cbar_label = 'BARRA-C2 annual mean TOA outgoing shortwave radiation\n' + r'(1979-2023) [$W \; m^{-2}$]'
        
        print(stats.describe(plt_data.values, axis=None))
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=50, cm_max=230, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
        
        extend = 'both'
    
    fig, ax = regional_plot(extent=[108, 160, -45.7, -5], central_longitude=180,)
    
    plt_mesh1 = ax.pcolormesh(
        plt_data.lon, plt_data.lat, plt_data.values,
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),zorder=1,)
    
    cbar = fig.colorbar(
        plt_mesh1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
        orientation="horizontal", shrink=0.9, ticks=pltticks, extend=extend,
        pad=0.02, fraction=0.12,)
    cbar.ax.set_xlabel(cbar_label, ha='center', linespacing=1.5, labelpad=4)
    fig.subplots_adjust(left=0.01, right = 0.99, bottom = 0.12, top = 0.99)
    fig.savefig(f'figures/4_um/4.0_barra/4.0.0_whole region/4.0.0.0 whole region barra_c2 annual mean {var}.png')
    
    del barra_c2_mon_alltime[var]


'''
Precipitation: pr
High Level Cloud Fraction: clh
Mid Level Cloud Fraction: clm
Low Level Cloud Fraction: cll
Total Cloud Cover Percentage: clt
Evaporation Including Sublimation and Transpiration: evspsbl
Surface Upward Latent Heat Flux: hfls
Surface Upward Sensible Heat Flux: hfss

sea level pressure: psl
Surface downwelling LW radiation: rlds
Surface Downwelling Clear-Sky Longwave Radiation: rldscs
Surface Upwelling Longwave Radiation: rlus
Surface Upwelling Clear-Sky Longwave Radiation: rluscs
TOA Outgoing Longwave Radiation: rlut
TOA Outgoing Clear-Sky Longwave Radiation: rlutcs

Surface downwelling SW radiation: rsds
Surface Downwelling Clear-Sky Shortwave Radiation: rsdscs
TOA Incident Shortwave Radiation: rsdt
Surface Upwelling Shortwave Radiation: rsus
Surface Upwelling Clear-Sky Shortwave Radiation: rsuscs
TOA Outgoing Shortwave Radiation: rsut
TOA Outgoing Clear-Sky Shortwave Radiation: rsutcs
near surface wind speed: sfcWind
Near surface air temperature: tas
Surface temperature: ts

'''
# endregion

