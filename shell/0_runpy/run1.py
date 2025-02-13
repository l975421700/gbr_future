

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


# region plot global era5 am data

era5_gridarea = xr.open_dataset('data/obs/era5/era5_gridarea.nc').cell_area

era5_sl_mon_alltime = {}
for var in ['tp', 'msl', 'sst', 'hcc', 'mcc', 'lcc', 'tcc', 't2m', 'msnlwrf', 'msnswrf', 'mtdwswrf', 'mtnlwrf', 'mtnswrf', 'msdwlwrf', 'msdwswrf', 'msdwlwrfcs', 'msdwswrfcs', 'msnlwrfcs', 'msnswrfcs', 'mtnlwrfcs', 'mtnswrfcs', 'cbh', 'tciw', 'tclw', 'e', 'z', 'mslhf', 'msshf', 'tcw', 'tcwv', 'tcsw', 'tcrw', 'tcslw', 'si10', 'd2m', 'cp', 'lsp', 'deg0l', 'mper', 'pev', 'skt', 'u10', 'v10', 'u100', 'v100',    'msuwlwrf',  'msuwswrf',  'msuwlwrfcs',  'msuwswrfcs',  'msnlwrfcl', 'msnswrfcl', 'msdwlwrfcl', 'msdwswrfcl', 'msuwlwrfcl', 'msuwswrfcl',  'mtuwswrf',  'mtuwswrfcs',  'mtnlwrfcl', 'mtnswrfcl', 'mtuwswrfcl']:
    # var = 'sst'
    print(f'#-------------------------------- {var}')
    with open(f'data/obs/era5/mon/era5_sl_mon_alltime_{var}.pkl', 'rb') as f:
        era5_sl_mon_alltime[var] = pickle.load(f)
    
    plt_data = era5_sl_mon_alltime[var]['am'].squeeze()
    print(stats.describe(plt_data.values, axis=None, nan_policy='omit'))
    # del era5_sl_mon_alltime[var]
    plt_data_gm = np.average(
        plt_data.values.ravel()[np.isfinite(plt_data.values.ravel())],
        weights=era5_gridarea.values.ravel()[np.isfinite(plt_data.values.ravel())])
    
    cbar_label = 'ERA5 annual mean (1979-2023) ' + era5_varlabels[var] + '\nglobal mean: ' + str(np.round(plt_data_gm, 2))
    
    if var in ['tp', 'tciw', 'tclw', 'z', 'tcw', 'tcwv', 'tcsw', 'tcrw', 'tcslw', 'cp', 'lsp']:
        extend = 'max'
    elif var in ['hcc', 'mcc', 'lcc', 'tcc']:
        extend = 'neither'
    else:
        extend = 'both'
    
    if var in ['tp', 'cp', 'lsp']:
        pltlevel = np.array([0, 0.1, 0.2, 0.5, 1, 2, 4, 6, 8, 10, 12, 16, 20,])
        pltticks = np.array([0, 0.1, 0.2, 0.5, 1, 2, 4, 6, 8, 10, 12, 16, 20,])
        pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
        pltcmp = plt.get_cmap('viridis_r', len(pltlevel)-1)
    elif var=='msl':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=980, cm_max=1025, cm_interval1=2.5, cm_interval2=5, cmap='viridis_r',)
    elif var=='sst':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-2, cm_max=30, cm_interval1=2, cm_interval2=4, cmap='viridis_r',)
    elif var in ['hcc', 'mcc', 'lcc', 'tcc']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=10, cmap='viridis_r',)
    elif var in ['t2m', 'skt']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-48, cm_max=32, cm_interval1=2, cm_interval2=8, cmap='BrBG', asymmetric=True,)
    elif var in ['wind', 'si10']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=2, cm_max=12, cm_interval1=1, cm_interval2=1, cmap='viridis_r',)
    elif var in ['msnlwrf', 'msnlwrfcs']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-170, cm_max=0, cm_interval1=10, cm_interval2=20, cmap='viridis',)
    elif var in ['msnswrf', 'msnswrfcs']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=300, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
    elif var=='mtdwswrf':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=180, cm_max=420, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
    elif var in ['mtnlwrf', 'mtnlwrfcs']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-300, cm_max=-130, cm_interval1=10, cm_interval2=20, cmap='viridis',)
    elif var in ['mtnswrf', 'mtnswrfcs']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=50, cm_max=380, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
    elif var in ['msdwlwrf', 'msdwlwrfcs']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=80, cm_max=430, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
    elif var in ['msdwswrf', 'msdwswrfcs']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=60, cm_max=330, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
    elif var=='cbh':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=7600, cm_interval1=200, cm_interval2=800, cmap='viridis_r',)
    elif var=='tciw':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=0.14, cm_interval1=0.01, cm_interval2=0.02, cmap='viridis_r',)
    elif var=='tclw':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=0.5, cm_interval1=0.05, cm_interval2=0.05, cmap='viridis_r',)
    elif var in ['e', 'mper', 'pev']:
        pltlevel = np.array([-10, -8, -6, -4, -2, -1, -0.5, -0.2, -0.1, 0, 0.1])
        pltticks = np.array([-10, -8, -6, -4, -2, -1, -0.5, -0.2, -0.1, 0, 0.1])
        pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
        pltcmp = plt.get_cmap('viridis', len(pltlevel)-1)
    elif var=='z':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=6000, cm_interval1=250, cm_interval2=500, cmap='viridis_r',)
    elif var=='mslhf':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-280, cm_max=0, cm_interval1=20, cm_interval2=40, cmap='viridis',)
    elif var=='msshf':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-120, cm_max=60, cm_interval1=10, cm_interval2=20, cmap='BrBG', asymmetric=True,)
    elif var in ['tcw', 'tcwv']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=60, cm_interval1=3, cm_interval2=6, cmap='viridis_r',)
    elif var in ['tcsw']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=0.9, cm_interval1=0.1, cm_interval2=0.1, cmap='viridis_r',)
    elif var=='tcrw':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=0.5, cm_interval1=0.05, cm_interval2=0.05, cmap='viridis_r',)
    elif var=='tcslw':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=0.2, cm_interval1=0.02, cm_interval2=0.04, cmap='viridis_r',)
    elif var=='d2m':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-56, cm_max=24, cm_interval1=2, cm_interval2=8, cmap='BrBG', asymmetric=True,)
    elif var=='deg0l':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=5200, cm_interval1=200, cm_interval2=400, cmap='BrBG', asymmetric=True,)
    elif var=='u10':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-12, cm_max=10, cm_interval1=1, cm_interval2=2, cmap='BrBG', asymmetric=True,)
    elif var=='v10':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-8, cm_max=14, cm_interval1=1, cm_interval2=2, cmap='BrBG', asymmetric=True,)
    elif var=='u100':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-18, cm_max=12, cm_interval1=1, cm_interval2=2, cmap='BrBG', asymmetric=True,)
    elif var=='v100':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-12, cm_max=20, cm_interval1=1, cm_interval2=2, cmap='BrBG', asymmetric=True,)
    elif var in ['msuwlwrf', 'msuwlwrfcs']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-540, cm_max=-120, cm_interval1=10, cm_interval2=40, cmap='viridis')
    elif var in ['msuwswrf', 'msuwswrfcs']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-220, cm_max=0, cm_interval1=10, cm_interval2=20, cmap='viridis')
    elif var in ['msnlwrfcl']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-20, cm_max=80, cm_interval1=10, cm_interval2=20, cmap='BrBG', asymmetric=True,)
    elif var in ['msnswrfcl']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-140, cm_max=60, cm_interval1=10, cm_interval2=40, cmap='BrBG', asymmetric=True,)
    elif var in ['msdwlwrfcl']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=65, cm_interval1=5, cm_interval2=10, cmap='viridis_r')
    elif var in ['msdwswrfcl']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-160, cm_max=0, cm_interval1=10, cm_interval2=20, cmap='viridis')
    elif var in ['msuwlwrfcl']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-45, cm_max=45, cm_interval1=5, cm_interval2=10, cmap='BrBG')
    elif var in ['msuwswrfcl']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-80, cm_max=80, cm_interval1=5, cm_interval2=20, cmap='BrBG')
    elif var in ['mtuwswrf', 'mtuwswrfcs']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-250, cm_max=-20, cm_interval1=10, cm_interval2=20, cmap='viridis')
    elif var in ['mtnlwrfcl']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-5, cm_max=85, cm_interval1=5, cm_interval2=10, cmap='BrBG', asymmetric=True,)
    elif var in ['mtnswrfcl', 'mtuwswrfcl']:
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-150, cm_max=60, cm_interval1=10, cm_interval2=20, cmap='BrBG', asymmetric=True,)
    
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

'''
'''
# endregion


