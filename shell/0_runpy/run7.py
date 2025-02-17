

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


# region plot ERA5 - BARRA-C2 am sm data

# settings
plt_colnames = ['Annual mean', 'DJF', 'MAM', 'JJA', 'SON']
plt_rownames = ['ERA5', 'BARRA-C2', 'ERA5 - BARRA-C2']
min_lon, max_lon, min_lat, max_lat = [110.58, 157.34, -43.69, -7.01]
# min_lon, max_lon, min_lat, max_lat = [108, 160, -45.7, -5]

era5_sl_mon_alltime = {}
barra_c2_mon_alltime = {}

for var2 in ['rlnscs', 'rsnscs',  'rlnscl', 'rsnscl', 'rldscl']:
    # var2='pr'
    # 'pr', 'clh', 'clm', 'cll', 'clt', 'evspsbl', 'hfls', 'hfss', 'psl', 'rlds', 'rldscs', 'rlus', 'rluscs', 'rlut', 'rlutcs', 'rsds', 'rsdscs', 'rsdt', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'sfcWind', 'tas', 'ts', 'evspsblpot', 'uas', 'vas', 'rlns',  'rsns',  'rlnscs', 'rsnscs',  'rlnscl', 'rsnscl', 'rldscl', 'rsdscl',  'rluscl', 'rsuscl',  'rsnt',  'rsntcs',  'rlutcl', 'rsntcl', 'rsutcl'
    var1 = cmip6_era5_var[var2]
    print(f'#-------------------------------- {var1} in ERA5 vs. {var2} in BARRA-C2')
    
    with open(f'data/obs/era5/mon/era5_sl_mon_alltime_{var1}.pkl', 'rb') as f:
        era5_sl_mon_alltime[var1] = pickle.load(f)
    with open(f'data/sim/um/barra_c2/barra_c2_mon_alltime_{var2}.pkl','rb') as f:
        barra_c2_mon_alltime[var2] = pickle.load(f)
    
    plt_data={}
    plt_data['ERA5']={}
    plt_data['BARRA-C2']={}
    plt_data['ERA5 - BARRA-C2']={}
    
    for jcolnames in plt_colnames:
        print(f'#---------------- {jcolnames}')
        if jcolnames=='Annual mean':
            # jcolnames='Annual mean'
            plt_data['ERA5'][jcolnames] = era5_sl_mon_alltime[var1]['am'].sel(lat=slice(max_lat, min_lat), lon=slice(min_lon, max_lon)).squeeze()
            plt_data['BARRA-C2'][jcolnames] = barra_c2_mon_alltime[var2]['am'].sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon)).squeeze()
            plt_data['ERA5 - BARRA-C2'][jcolnames] = plt_data['ERA5'][jcolnames] - regrid(plt_data['BARRA-C2'][jcolnames], ds_out=plt_data['ERA5'][jcolnames])
            
            ttest_fdr_res = ttest_fdr_control(
                era5_sl_mon_alltime[var1]['ann'].sel(lat=slice(max_lat, min_lat), lon=slice(min_lon, max_lon)),
                regrid(barra_c2_mon_alltime[var2]['ann'].sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon)), ds_out=plt_data['ERA5'][jcolnames])
            )
            plt_data['ERA5 - BARRA-C2'][jcolnames] = plt_data['ERA5 - BARRA-C2'][jcolnames].where(ttest_fdr_res, np.nan)
        else:
            # jcolnames='MAM'
            plt_data['ERA5'][jcolnames] = era5_sl_mon_alltime[var1]['sm'].sel(time=jcolnames, lat=slice(max_lat, min_lat), lon=slice(min_lon, max_lon), drop=True)
            plt_data['BARRA-C2'][jcolnames] = barra_c2_mon_alltime[var2]['sm'].sel(time=jcolnames, lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))
            plt_data['ERA5 - BARRA-C2'][jcolnames] = plt_data['ERA5'][jcolnames] - regrid(plt_data['BARRA-C2'][jcolnames], ds_out=plt_data['ERA5'][jcolnames])
            
            ttest_fdr_res = ttest_fdr_control(
                era5_sl_mon_alltime[var1]['sea'][era5_sl_mon_alltime[var1]['sea'].time.dt.season==jcolnames].sel(lat=slice(max_lat, min_lat), lon=slice(min_lon, max_lon)),
                regrid(barra_c2_mon_alltime[var2]['sea'][barra_c2_mon_alltime[var2]['sea'].time.dt.season==jcolnames].sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon)), ds_out=plt_data['ERA5'][jcolnames])
                    )
            plt_data['ERA5 - BARRA-C2'][jcolnames] = plt_data['ERA5 - BARRA-C2'][jcolnames].where(ttest_fdr_res, np.nan)
    
    print(stats.describe(np.concatenate([np.concatenate([plt_data['ERA5']['Annual mean'].values, plt_data['ERA5']['DJF'].values, plt_data['ERA5']['MAM'].values, plt_data['ERA5']['JJA'].values, plt_data['ERA5']['SON'].values]).ravel(), np.concatenate([plt_data['BARRA-C2']['Annual mean'].values, plt_data['BARRA-C2']['DJF'].values, plt_data['BARRA-C2']['MAM'].values, plt_data['BARRA-C2']['JJA'].values, plt_data['BARRA-C2']['SON'].values]).ravel()]), axis=None, nan_policy='omit'))
    print(stats.describe(np.concatenate([plt_data['ERA5 - BARRA-C2']['Annual mean'].values, plt_data['ERA5 - BARRA-C2']['DJF'].values, plt_data['ERA5 - BARRA-C2']['MAM'].values, plt_data['ERA5 - BARRA-C2']['JJA'].values, plt_data['ERA5 - BARRA-C2']['SON'].values]).ravel(), axis=None, nan_policy='omit'))
    
    cbar_label1 = '1979-2023 ' + era5_varlabels[var1]
    cbar_label2 = 'Difference in 1979-2023 ' + era5_varlabels[var1]
    
    if var1 in ['tp']:
        extend1 = 'max'
    elif var1 in ['hcc', 'mcc', 'lcc', 'tcc']:
        extend1 = 'neither'
    else:
        extend1 = 'both'
    extend2 = 'both'
    
    if var1=='tp':
        pltlevel1 = np.array([0, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20,])
        pltticks1 = np.array([0, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20,])
        pltnorm1 = BoundaryNorm(pltlevel1, ncolors=len(pltlevel1)-1, clip=True)
        pltcmp1 = plt.get_cmap('viridis_r', len(pltlevel1)-1)
        pltlevel2 = np.array([-6, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 6])
        pltticks2 = np.array([-6, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 6])
        pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
        pltcmp2 = plt.get_cmap('BrBG', len(pltlevel2)-1)
    elif var1 in ['hcc', 'mcc', 'lcc', 'tcc']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=10, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=10, cmap='BrBG',)
    elif var1 in ['e', 'pev']:
        pltlevel1 = np.array([0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8])
        pltticks1 = np.array([0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8])
        pltnorm1 = BoundaryNorm(pltlevel1, ncolors=len(pltlevel1)-1, clip=True)
        pltcmp1 = plt.get_cmap('viridis_r', len(pltlevel1)-1)
        pltlevel2 = np.array([-5, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 5])
        pltticks2 = np.array([-5, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 5])
        pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
        pltcmp2 = plt.get_cmap('BrBG', len(pltlevel2)-1)
    elif var1=='mslhf':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-460, cm_max=30, cm_interval1=10, cm_interval2=40, cmap='PRGn', asymmetric=True)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-200, cm_max=200, cm_interval1=10, cm_interval2=40, cmap='BrBG',)
    elif var1=='msshf':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-200, cm_max=80, cm_interval1=10, cm_interval2=40, cmap='PRGn', asymmetric=True,)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-160, cm_max=160, cm_interval1=10, cm_interval2=40, cmap='BrBG',)
    elif var1=='msl':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=1005, cm_max=1022, cm_interval1=1, cm_interval2=2, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-0.5, cm_max=4.5, cm_interval1=0.5, cm_interval2=1, cmap='BrBG', asymmetric=True,)
    elif var1 in ['msdwlwrf', 'msdwlwrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=200, cm_max=440, cm_interval1=10, cm_interval2=40, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-50, cm_max=90, cm_interval1=5, cm_interval2=20, cmap='BrBG', asymmetric=True,)
    elif var1 in ['msuwlwrf', 'msuwlwrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-540, cm_max=-300, cm_interval1=10, cm_interval2=40, cmap='viridis',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-60, cm_max=60, cm_interval1=5, cm_interval2=20, cmap='BrBG',)
    elif var1 in ['mtnlwrf', 'mtnlwrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-310, cm_max=-180, cm_interval1=5, cm_interval2=20, cmap='viridis',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-20, cm_max=20, cm_interval1=2, cm_interval2=4, cmap='BrBG',)
    elif var1 in ['msdwswrf', 'msdwswrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=40, cm_max=400, cm_interval1=10, cm_interval2=40, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-50, cm_max=100, cm_interval1=5, cm_interval2=20, cmap='BrBG', asymmetric=True)
    elif var1=='mtdwswrf':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=150, cm_max=490, cm_interval1=10, cm_interval2=40, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-0.3, cm_max=0.6, cm_interval1=0.1, cm_interval2=0.1, cmap='BrBG',)
    elif var1 in ['msuwswrf', 'msuwswrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-160, cm_max=0, cm_interval1=5, cm_interval2=20, cmap='viridis',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-40, cm_max=100, cm_interval1=5, cm_interval2=20, cmap='BrBG', asymmetric=True)
    elif var1 in ['mtuwswrf', 'mtuwswrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-250, cm_max=-20, cm_interval1=10, cm_interval2=40, cmap='viridis',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=100, cm_interval1=5, cm_interval2=20, cmap='BrBG', asymmetric=True)
    elif var1 in ['si10']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=1, cm_max=11, cm_interval1=1, cm_interval2=1, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='BrBG')
    elif var1=='t2m':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-2, cm_max=34, cm_interval1=1, cm_interval2=4, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-4, cm_max=10, cm_interval1=1, cm_interval2=1, cmap='BrBG', asymmetric=True)
    elif var1=='skt':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-2, cm_max=40, cm_interval1=1, cm_interval2=4, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-7, cm_max=10, cm_interval1=1, cm_interval2=1, cmap='BrBG', asymmetric=True)
    elif var1 in ['u10', 'v10']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-9, cm_max=9, cm_interval1=1, cm_interval2=2, cmap='PRGn',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-5, cm_max=5, cm_interval1=1, cm_interval2=2, cmap='BrBG',)
    elif var1 in ['msnlwrf', 'msnlwrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-150, cm_max=0, cm_interval1=10, cm_interval2=20, cmap='viridis',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10, cmap='BrBG',)
    elif var1 in ['msnswrf', 'msnswrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=30, cm_max=350, cm_interval1=10, cm_interval2=40, cmap='viridis',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-40, cm_max=100, cm_interval1=5, cm_interval2=20, cmap='BrBG',)
    elif var1=='msnlwrfcl':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-25, cm_max=70, cm_interval1=5, cm_interval2=10, cmap='PRGn', asymmetric=True)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=2.5, cm_interval2=5, cmap='BrBG',)
    elif var1=='msnswrfcl':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-230, cm_max=30, cm_interval1=10, cm_interval2=40, cmap='PRGn', asymmetric=True)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=80, cm_interval1=5, cm_interval2=10, cmap='BrBG', asymmetric=True)
    elif var1=='msdwlwrfcl':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=0, cm_max=70, cm_interval1=2.5, cm_interval2=10, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-40, cm_max=10, cm_interval1=2.5, cm_interval2=5, cmap='BrBG', asymmetric=True)
    elif var1=='msdwswrfcl':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-270, cm_max=0, cm_interval1=10, cm_interval2=40, cmap='viridis',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=100, cm_interval1=5, cm_interval2=10, cmap='BrBG', asymmetric=True)
    elif var1=='msuwlwrfcl':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10, cmap='PRGn',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10, cmap='BrBG',)
    elif var1=='msuwswrfcl':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-40, cm_max=60, cm_interval1=5, cm_interval2=10, cmap='PRGn', asymmetric=True)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-50, cm_max=50, cm_interval1=5, cm_interval2=10, cmap='BrBG',)
    elif var1 in ['mtnswrf', 'mtnswrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=70, cm_max=440, cm_interval1=10, cm_interval2=40, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=100, cm_interval1=5, cm_interval2=10, cmap='BrBG', asymmetric=True)
    elif var1=='mtnlwrfcl':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=0, cm_max=80, cm_interval1=5, cm_interval2=10, cmap='viridis_r',)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-16, cm_max=12, cm_interval1=2, cm_interval2=4, cmap='BrBG', asymmetric=True)
    elif var1 in ['mtnswrfcl', 'mtuwswrfcl']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-180, cm_max=30, cm_interval1=10, cm_interval2=20, cmap='PRGn', asymmetric=True)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=60, cm_interval1=5, cm_interval2=10, cmap='BrBG', asymmetric=True)
    else:
        print(f'Warning unspecified colorbar for {var1}')
    
    nrow=3
    ncol=5
    fm_bottom=1.5/(4*nrow+1.5)
    
    fig, axs = plt.subplots(
        nrow, ncol, figsize=np.array([4.4*ncol, 4*nrow + 1.5]) / 2.54,
        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
        gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)
    
    for irow in range(nrow):
        axs[irow, 0].text(-0.05, 0.5, plt_rownames[irow], ha='right', va='center', rotation='vertical', transform=axs[irow, 0].transAxes,weight='bold')
        for jcol in range(ncol):
            axs[irow, jcol] = regional_plot(extent=[min_lon, max_lon, min_lat, max_lat], central_longitude=180, ax_org=axs[irow, jcol])
            axs[irow, jcol].text(0, 1.02, f'({string.ascii_lowercase[irow]}{jcol+1})', ha='left', va='bottom', transform=axs[irow, jcol].transAxes,)
            if irow==0:
                axs[0, jcol].text(0.5, 1.02, plt_colnames[jcol], ha='center', va='bottom', transform=axs[0, jcol].transAxes,weight='bold')
    
    for irow in range(nrow-1):
        for jcol in range(ncol):
            plt_mesh1 = axs[irow, jcol].pcolormesh(
                plt_data[plt_rownames[irow]][plt_colnames[jcol]].lon,
                plt_data[plt_rownames[irow]][plt_colnames[jcol]].lat,
                plt_data[plt_rownames[irow]][plt_colnames[jcol]].values,
                norm=pltnorm1, cmap=pltcmp1,
                transform=ccrs.PlateCarree(),zorder=1,)
    
    for jcol in range(ncol):
        plt_mesh2 = axs[2, jcol].pcolormesh(
            plt_data[plt_rownames[2]][plt_colnames[jcol]].lon,
            plt_data[plt_rownames[2]][plt_colnames[jcol]].lat,
            plt_data[plt_rownames[2]][plt_colnames[jcol]].values,
            norm=pltnorm2, cmap=pltcmp2,
            transform=ccrs.PlateCarree(),zorder=1,)
    
    cbar1 = fig.colorbar(
        plt_mesh1, #cm.ScalarMappable(norm=pltnorm1, cmap=pltcmp1), #
        aspect=30, format=remove_trailing_zero_pos,
        orientation="horizontal", ticks=pltticks1, extend=extend1,
        cax=fig.add_axes([0.05, fm_bottom-0.02, 0.4, 0.02]))
    cbar1.ax.set_xlabel(cbar_label1)
    cbar2 = fig.colorbar(
        plt_mesh2, #cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), #
        aspect=30, format=remove_trailing_zero_pos,
        orientation="horizontal", ticks=pltticks2, extend=extend2,
        cax=fig.add_axes([0.55, fm_bottom-0.02, 0.4, 0.02]))
    cbar2.ax.set_xlabel(cbar_label2)
    
    fig.subplots_adjust(left=0.03, right=0.995, bottom=fm_bottom, top=0.98)
    fig.savefig(f'figures/4_um/4.0_barra/4.0.0_whole region/4.0.0.1 barra_c2 vs. era5 am sm {var1}.png')
    
    del era5_sl_mon_alltime[var1], barra_c2_mon_alltime[var2]


# endregion

