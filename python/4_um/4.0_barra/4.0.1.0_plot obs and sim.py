

# qsub -I -q normal -P v46 -l walltime=3:00:00,ncpus=1,mem=96GB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22+gdata/qx55+gdata/gx60


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
import xesmf as xe
import calendar
import glob
from metpy.calc import specific_humidity_from_dewpoint, relative_humidity_from_dewpoint, vertical_velocity_pressure, mixing_ratio_from_specific_humidity
from xmip.preprocessing import replace_x_y_nominal_lat_lon

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
from matplotlib import cm
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=12)
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
    monthini,
    seasons,
    seconds_per_d,
    zerok,
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
    time_weighted_mean,
    coslat_weighted_mean,
    coslat_weighted_rmsd,
    mon_sea_ann,
    regrid,
    cdo_regrid,)

from statistics0 import (
    ttest_fdr_control,)

from um_postprocess import (
    preprocess_umoutput,
    stash2var, stash2var_gal, stash2var_ral,
    var2stash, var2stash_gal, var2stash_ral,
    suite_res, suite_label,)

# endregion


# region plot obs and sim am

# options
years = '2016'; yeare = '2023'
# ['rsut', 'rlut'， 'cll', 'clm', 'clh', 'clt', 'clwvi', 'clivi', 'inversionh', 'LCL', 'LTS', 'EIS', 'pr']
vars = ['clt']
# ['CERES', 'CM SAF', 'Himawari', 'ERA5', 'BARRA-R2', 'BARRA-C2', 'BARPA-C', 'MOD08_M3', 'MYD08_M3', 'IMERG']
ds_names = ['Himawari', 'ERA5', 'BARRA-R2', 'BARRA-C2', 'BARPA-C']
plt_regions = ['c2_domain'] # ['global', 'c2_domain', 'h9_domain']
plt_modes = ['difference'] # ['original', 'difference']

# settings
min_lon, max_lon, min_lat, max_lat = [110.58, 157.34, -43.69, -7.01]
min_lonh9, max_lonh9, min_lath9, max_lath9 = [80, 200, -60, 60]
nrow = 1
ncol = len(ds_names)
cm_saf_varnames = {'rlut': 'LW_flux', 'rsut': 'SW_flux', 'CDNC': 'cdnc_liq',
                   'clwvi': 'lwp_allsky', 'clivi': 'iwp_allsky',}
cltypes = {
    'hcc': ['Cirrus', 'Cirrostratus', 'Deep convection'],
    'mcc': ['Altocumulus', 'Altostratus', 'Nimbostratus'],
    'lcc': ['Cumulus', 'Stratocumulus', 'Stratus'],
    'tcc': ['Cirrus', 'Cirrostratus', 'Deep convection', 'Altocumulus', 'Altostratus', 'Nimbostratus', 'Cumulus', 'Stratocumulus', 'Stratus']}
extend2 = 'both'

for ivar in vars:
    # ivar = 'cll'
    print(f'#-------------------------------- {ivar}')
    
    ds_data = {'ann': {}, 'am': {}}
    for ids in ds_names:
        print(f'Get {ids}')
        
        if ids == 'CERES':
            # ids = 'CERES'
            if cmip6_era5_var[ivar] in ['mtdwswrf', 'mtnlwrf', 'mtuwswrf']:
                ceres = xr.open_dataset('data/obs/CERES/CERES_EBAF-TOA_Ed4.2.1_Subset_200003-202411.nc').sel(time=slice(years, yeare))
                ceres = ceres.rename({
                    'toa_sw_all_mon': 'mtuwswrf',
                    'toa_lw_all_mon': 'mtnlwrf',
                    'solar_mon': 'mtdwswrf'})
            elif cmip6_era5_var[ivar] in ['msdwswrf', 'msuwswrf', 'msdwlwrf', 'msuwlwrf']:
                ceres = xr.open_dataset('data/obs/CERES/CERES_EBAF_Ed4.2_Subset_200003-202407 (1).nc').sel(time=slice(years, yeare))
                ceres = ceres.rename({
                    'sfc_sw_down_all_mon': 'msdwswrf',
                    'sfc_sw_up_all_mon': 'msuwswrf',
                    'sfc_lw_down_all_mon': 'msdwlwrf',
                    'sfc_lw_up_all_mon': 'msuwlwrf'})
            elif cmip6_era5_var[ivar] in ['tclw', 'tciw']:
                ceres = xr.open_dataset('data/obs/CERES/CERES_SYN1deg-Month_Terra-Aqua-NOAA20_Ed4.2_Subset_201601-202412.nc').sel(time=slice(years, yeare))
                ceres = ceres.rename({
                    'lwp_total_mon': 'tclw', 'iwp_total_mon': 'tciw'})
            
            if cmip6_era5_var[ivar] in ['mtuwswrf', 'mtnlwrf', 'msuwswrf', 'msuwlwrf']:
                ceres[cmip6_era5_var[ivar]] *= (-1)
            
            ds_data['ann'][ids] = ceres[cmip6_era5_var[ivar]].resample({'time': '1YE'}).map(time_weighted_mean).compute()
        elif ids == 'CM SAF':
            # ids = 'CM SAF'
            fl = sorted(glob.glob(f'data/obs/CM_SAF/{ivar}/*.nc'))
            cm_saf = xr.open_mfdataset(fl).sel(time=slice(years, yeare))
            
            if ivar in ['rsut', 'rlut']:
                cm_saf[cm_saf_varnames[ivar]] *= (-1)
            elif ivar in ['clwvi', 'clivi']:
                cm_saf[cm_saf_varnames[ivar]] *= 1000
            elif ivar == 'CDNC':
                cm_saf[cm_saf_varnames[ivar]] /= 1e6
            
            ds_data['ann'][ids] = cm_saf[cm_saf_varnames[ivar]].resample({'time': '1YE'}).map(time_weighted_mean).compute()
        elif ids == 'ERA5':
            # ids = 'ERA5'
            with open(f'data/sim/era5/mon/era5_sl_mon_alltime_{cmip6_era5_var[ivar]}.pkl', 'rb') as f:
                era5_sl_mon_alltime = pickle.load(f)
            ds_data['ann'][ids] = era5_sl_mon_alltime['ann'].sel(time=slice(years, yeare))
            if ivar in ['clwvi', 'clivi']:
                ds_data['ann'][ids] *= 1000
        elif ids == 'BARRA-R2':
            # ids = 'BARRA-R2'
            with open(f'data/sim/um/barra_r2/barra_r2_mon_alltime_{ivar}.pkl','rb') as f:
                barra_r2_mon_alltime = pickle.load(f)
            ds_data['ann'][ids] = barra_r2_mon_alltime['ann'].sel(time=slice(years, yeare))
            if ivar in ['clwvi', 'clivi']:
                ds_data['ann'][ids] *= 1000
        elif ids == 'BARRA-C2':
            # ids = 'BARRA-C2'
            with open(f'data/sim/um/barra_c2/barra_c2_mon_alltime_{ivar}.pkl','rb') as f:
                barra_c2_mon_alltime = pickle.load(f)
            ds_data['ann'][ids] = barra_c2_mon_alltime['ann'].sel(time=slice(years, yeare))
            if ivar in ['clwvi', 'clivi']:
                ds_data['ann'][ids] *= 1000
        elif ids == 'BARPA-C':
            # ids = 'BARPA-C'
            with open(f'data/sim/um/barpa_c/barpa_c_mon_alltime_{ivar}.pkl','rb') as f:
                barpa_c_mon_alltime = pickle.load(f)
            ds_data['ann'][ids] = barpa_c_mon_alltime['ann'].sel(time=slice(years, yeare))
            if ivar in ['clwvi', 'clivi']:
                ds_data['ann'][ids] *= 1000
        elif ids == 'Himawari':
            # ids = 'Himawari'
            if 'cltype_frequency_alltime' not in globals():
                with open('data/obs/jaxa/clp/cltype_frequency_alltime.pkl', 'rb') as f:
                    cltype_frequency_alltime = pickle.load(f)
            ds_data['ann'][ids] = cltype_frequency_alltime['ann'].sel(types=cltypes[cmip6_era5_var[ivar]], time=slice(years, yeare)).sum(dim='types')
        elif ids in ['MOD08_M3', 'MYD08_M3']:
            # ids = 'MYD08_M3'
            modis = xr.open_mfdataset(glob.glob(f'data/obs/MODIS/{ids}/*{ivar}*.nc')).sel(time=slice(years, yeare))
            ds_data['ann'][ids] = modis[ivar].resample({'time': '1YE'}).map(time_weighted_mean).compute()
        elif ids == 'IMERG':
            with open(f'data/obs/IMERG/imerg_mon_alltime_pr.pkl', 'rb') as f:
                imerg_mon_alltime = pickle.load(f)
            ds_data['ann'][ids] = imerg_mon_alltime['ann']
        elif ids == 'AGCD':
            # ids = 'AGCD'
            with open(f'data/obs/agcd/agcd_alltime_{ivar}.pkl','rb') as f:
                agcd_alltime = pickle.load(f)
            ds_data['ann'][ids] = agcd_alltime['ann'].sel(time=slice(years, yeare))
        else:
            print('Warning: unknown dataset')
        
        ds_data['ann'][ids]['lon'] = ds_data['ann'][ids]['lon'] % 360
        ds_data['ann'][ids] = ds_data['ann'][ids].sortby(['lon', 'lat'])
        
        ds_data['am'][ids] = ds_data['ann'][ids].mean(dim='time').compute()
    
    cbar_label1 = f'{years}-{yeare} {era5_varlabels[cmip6_era5_var[ivar]]}'
    cbar_label2 = f'Difference in {era5_varlabels[cmip6_era5_var[ivar]]}'
    
    for plt_region in plt_regions:
        # plt_region = 'global'
        print(f'#---------------- {plt_region}')
        
        extend1 = 'both'
        if plt_region == 'global':
            if ivar in ['rsut']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=-200, cm_max=-40, cm_interval1=10, cm_interval2=20,
                    cmap='viridis',)
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-10, cm_max=10, cm_interval1=1, cm_interval2=2,
                    cmap='BrBG')
            elif ivar in ['rlut']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=-300, cm_max=-120, cm_interval1=10, cm_interval2=20,
                    cmap='viridis',)
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-10, cm_max=10, cm_interval1=1, cm_interval2=2,
                    cmap='BrBG')
            elif ivar in ['clwvi']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=0, cm_max=160, cm_interval1=10, cm_interval2=20,
                    cmap='viridis',)
                extend1 = 'max'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-100, cm_max=100, cm_interval1=10, cm_interval2=20,
                    cmap='BrBG_r')
            elif ivar in ['clivi']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=0, cm_max=240, cm_interval1=10, cm_interval2=20,
                    cmap='viridis',)
                extend1 = 'max'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-100, cm_max=100, cm_interval1=10, cm_interval2=20,
                    cmap='BrBG_r')
        elif plt_region == 'c2_domain':
            if ivar in ['rsut']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=-150, cm_max=-50, cm_interval1=5, cm_interval2=10,
                    cmap='viridis',)
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    # cm_min=-10, cm_max=10, cm_interval1=1, cm_interval2=2,
                    cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10,
                    cmap='BrBG')
            elif ivar in ['rlut']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=-290, cm_max=-210, cm_interval1=5, cm_interval2=10,
                    cmap='viridis',)
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    # cm_min=-10, cm_max=10, cm_interval1=1, cm_interval2=2,
                    cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10,
                    cmap='BrBG')
            elif ivar in ['clh', 'clm', 'cll']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=0, cm_max=60, cm_interval1=5, cm_interval2=10,
                    cmap='Blues_r',)
                extend1 = 'max'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=10,
                    cmap='BrBG_r',)
            elif ivar in ['clt']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=10,
                    cmap='Blues_r',)
                extend1 = 'neither'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=10,
                    cmap='BrBG_r',)
            elif ivar in ['clwvi']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=0, cm_max=160, cm_interval1=10, cm_interval2=20,
                    cmap='viridis',)
                extend1 = 'max'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-100, cm_max=100, cm_interval1=10, cm_interval2=20,
                    cmap='BrBG_r')
            elif ivar in ['clivi']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=0, cm_max=240, cm_interval1=10, cm_interval2=20,
                    cmap='viridis',)
                extend1 = 'max'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-100, cm_max=100, cm_interval1=10, cm_interval2=20,
                    cmap='BrBG_r')
            elif ivar in ['inversionh']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=0, cm_max=3200, cm_interval1=200, cm_interval2=400,
                    cmap='viridis',)
                extend1 = 'max'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-400, cm_max=400, cm_interval1=50, cm_interval2=100,
                    cmap='BrBG_r')
            elif ivar in ['LCL']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=0, cm_max=2000, cm_interval1=100, cm_interval2=200,
                    cmap='viridis',)
                extend1 = 'max'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-200, cm_max=200, cm_interval1=25, cm_interval2=50,
                    cmap='BrBG_r')
            elif ivar in ['LTS']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=0, cm_max=16, cm_interval1=1, cm_interval2=2,
                    cmap='viridis')
                extend1 = 'max'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-3, cm_max=3, cm_interval1=0.5, cm_interval2=1,
                    cmap='BrBG_r')
            elif ivar in ['EIS']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=0, cm_max=10, cm_interval1=1, cm_interval2=1,
                    cmap='viridis')
                extend1 = 'max'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-3, cm_max=3, cm_interval1=0.5, cm_interval2=1,
                    cmap='BrBG_r')
            elif ivar=='pr':
                pltlevel1 = np.array([0, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20,])
                pltticks1 = np.array([0, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20,])
                pltnorm1 = BoundaryNorm(pltlevel1, ncolors=len(pltlevel1)-1, clip=True)
                pltcmp1 = plt.get_cmap('viridis_r', len(pltlevel1)-1)
                extend1 = 'max'
                pltlevel2 = np.array([-6, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 6])
                pltticks2 = np.array([-6, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 6])
                pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
                pltcmp2 = plt.get_cmap('BrBG', len(pltlevel2)-1)
        elif plt_region == 'h9_domain':
            if ivar in ['clh', 'clm', 'cll']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=0, cm_max=80, cm_interval1=5, cm_interval2=10,
                    cmap='Blues_r',)
                extend1 = 'max'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=10,
                    cmap='BrBG_r',)
            elif ivar in ['clt']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=10,
                    cmap='Blues_r',)
                extend1 = 'neither'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=10,
                    cmap='BrBG_r',)
        else:
            print('Warning: unspecified colorbar')
        
        plt_org = {}
        plt_ann = {}
        plt_mean = {}
        for ids in ds_names:
            print(f'org: {ids}')
            if plt_region == 'global':
                plt_org[ids] = ds_data['am'][ids]
                plt_ann[ids] = ds_data['ann'][ids]
            elif plt_region == 'c2_domain':
                plt_org[ids] = ds_data['am'][ids].sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))
                plt_ann[ids] = ds_data['ann'][ids].sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))
            elif plt_region == 'h9_domain':
                plt_org[ids] = ds_data['am'][ids].sel(lon=slice(min_lonh9, max_lonh9), lat=slice(min_lath9, max_lath9))
                plt_ann[ids] = ds_data['ann'][ids].sel(lon=slice(min_lonh9, max_lonh9), lat=slice(min_lath9, max_lath9))
            plt_mean[ids] = coslat_weighted_mean(plt_org[ids])
        
        plt_diff = {}
        plt_rmsd = {}
        plt_md = {}
        for ids in ds_names[1:]:
            print(f'diff: {ids}')
            if (ds_names[0] in ['Himawari']) & (plt_region == 'h9_domain'):
                plt_diff[ids] = xe.Regridder(plt_org[ids], plt_org[ds_names[0]], 'bilinear')(plt_org[ids]) - plt_org[ds_names[0]]
            else:
                plt_diff[ids] = regrid(plt_org[ids], plt_org[ds_names[0]]) - plt_org[ds_names[0]]
            plt_rmsd[ids] = coslat_weighted_rmsd(plt_diff[ids])
            plt_md[ids] = coslat_weighted_mean(plt_diff[ids])
            
            if ivar not in ['pr']:
                ttest_fdr_res = ttest_fdr_control(
                    xe.Regridder(plt_ann[ids], plt_ann[ds_names[0]], 'bilinear')(plt_ann[ids]),
                    # regrid(plt_ann[ids], plt_ann[ds_names[0]]),
                    plt_ann[ds_names[0]])
                plt_diff[ids] = plt_diff[ids].where(ttest_fdr_res, np.nan)
        
        for plt_mode in plt_modes:
            print(f'#-------- {plt_mode}')
            
            if plt_region == 'global':
                fig, axs = plt.subplots(
                    nrow, ncol,
                    figsize=np.array([8.8*ncol, 4.4*nrow + 2]) / 2.54,
                    subplot_kw={'projection': ccrs.Mollweide(central_longitude=180)},
                    gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)
                for jcol in range(ncol):
                    axs[jcol] = globe_plot(ax_org=axs[jcol])
            elif plt_region == 'c2_domain':
                fig, axs = plt.subplots(
                    nrow, ncol,
                    figsize=np.array([4.4*ncol, 4*nrow+1.6])/2.54,
                    subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
                    gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)
                for jcol in range(ncol):
                    axs[jcol] = regional_plot(extent=[min_lon, max_lon, min_lat, max_lat], central_longitude=180, ax_org=axs[jcol])
            elif plt_region == 'h9_domain':
                fig, axs = plt.subplots(
                    nrow, ncol,
                    figsize=np.array([6.6*ncol, 6*nrow+3.5])/2.54,
                    subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
                    gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)
                for jcol in range(ncol):
                    axs[jcol] = regional_plot(extent=[min_lonh9, max_lonh9, min_lath9, max_lath9], central_longitude=180, ax_org=axs[jcol])
            else:
                print('Warning: plt_region unspecified')
            
            plt_colnames = [f'{ds_names[0]}']
            plt_text = [f'Mean: {str(np.round(plt_mean[ds_names[0]], 1))}']
            if plt_mode in ['original']:
                plt_colnames += [f'{ids}' for ids in ds_names[1:]]
                plt_text += [f'{str(np.round(plt_mean[ids], 1))}' for ids in ds_names[1:]]
            elif plt_mode in ['difference']:
                plt_colnames += [f'{ds_names[1]} - {ds_names[0]}']
                plt_text += [f'RMSD: {str(np.round(plt_rmsd[ds_names[1]], 1))}, MD: {str(np.round(plt_md[ds_names[1]], 1))}']
                plt_colnames += [f'{ids} - {ds_names[0]}' for ids in ds_names[2:]]
                plt_text += [f'{str(np.round(plt_rmsd[ids], 1))}, {str(np.round(plt_md[ids], 1))}' for ids in ds_names[2:]]
            
            for jcol in range(ncol):
                axs[jcol].text(
                    0, 1.02,
                    f'({string.ascii_lowercase[jcol]}) {plt_colnames[jcol]}',
                    ha='left',va='bottom',transform=axs[jcol].transAxes)
                axs[jcol].text(
                    0.01, 0.01, plt_text[jcol], ha='left', va='bottom',
                    transform=axs[jcol].transAxes,)
            
            plt_mesh1 = axs[0].pcolormesh(
                plt_org[ds_names[0]].lon,
                plt_org[ds_names[0]].lat,
                plt_org[ds_names[0]],
                norm=pltnorm1, cmap=pltcmp1,
                transform=ccrs.PlateCarree(), zorder=1)
            
            if plt_mode in ['original']:
                for jcol in range(ncol-1):
                    plt_mesh1 = axs[jcol+1].pcolormesh(
                        plt_org[ds_names[jcol+1]].lon,
                        plt_org[ds_names[jcol+1]].lat,
                        plt_org[ds_names[jcol+1]],
                        norm=pltnorm1, cmap=pltcmp1,
                        transform=ccrs.PlateCarree(),zorder=1)
                cbar1 = fig.colorbar(
                    plt_mesh1,#cm.ScalarMappable(norm=pltnorm1, cmap=pltcmp1),#
                    format=remove_trailing_zero_pos,
                    orientation="horizontal", ticks=pltticks1, extend=extend1,
                    cax=fig.add_axes([0.26, 0.2, 0.48, 0.04]))
                cbar1.ax.set_xlabel(cbar_label1)
            elif plt_mode in ['difference']:
                for jcol in range(ncol-1):
                    plt_mesh2 = axs[jcol+1].pcolormesh(
                        plt_diff[ds_names[jcol+1]].lon,
                        plt_diff[ds_names[jcol+1]].lat,
                        plt_diff[ds_names[jcol+1]],
                        norm=pltnorm2, cmap=pltcmp2,
                        transform=ccrs.PlateCarree(),zorder=1)
                cbar1 = fig.colorbar(
                    plt_mesh1,#cm.ScalarMappable(norm=pltnorm1, cmap=pltcmp1),#
                    format=remove_trailing_zero_pos,
                    orientation="horizontal", ticks=pltticks1, extend=extend1,
                    cax=fig.add_axes([0.01, 0.2, 0.48, 0.04]))
                cbar1.ax.set_xlabel(cbar_label1)
                cbar2 = fig.colorbar(
                    plt_mesh2,#cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2),#
                    format=remove_trailing_zero_pos,
                    orientation="horizontal", ticks=pltticks2, extend=extend2,
                    cax=fig.add_axes([0.51, 0.2, 0.48, 0.04]))
                cbar2.ax.set_xlabel(cbar_label2)
            else:
                print('Warning: unspecified plot mode')
            
            opng = f'figures/4_um/4.0_barra/4.0.7_obs_sim/4.0.7.0 {ivar} {', '.join(ds_names)} {plt_region} {plt_mode} {years}-{yeare}.png'
            fig.subplots_adjust(left=0.005, right=0.995, bottom=0.2, top=0.97)
            fig.savefig(opng)
    
    del ds_data









'''
plt_data[ids].weighted(np.cos(np.deg2rad(plt_data[ids].lat))).mean()
'''
# endregion

