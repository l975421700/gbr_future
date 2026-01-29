

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
import rioxarray as rxr

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
    coslat_weighted_mean,
    coslat_weighted_rmsd,
    global_land_ocean_rmsd,
    global_land_ocean_mean,
    mon_sea_ann,
    regrid,
    cdo_regrid,)

from statistics0 import (
    ttest_fdr_control,)

from um_postprocess import (
    preprocess_umoutput,
    stash2var, stash2var_gal, stash2var_ral,
    var2stash, var2stash_gal, var2stash_ral,
    suite_res, suite_label,
    interp_to_pressure_levels,
    amstash2var, amvar2stash, preprocess_amoutput, amvargroups, am3_label)

# endregion


# region plot obs and sim am sl
# Memory Used: 38.13GB, Walltime Used: 00:50:53 for 4 ds
# Memory Used: 42.17GB, Walltime Used: 00:59:30 for 3 ds

# options
years = '1983'; yeare = '1987'
vars = [
    'hfms', 'huss', 'hurs'
    
    # # monthly
    # 'seaice', 'sst'
    # 'rsn_trop', 'rsu_trop', 'rln_trop', 'rld_trop', 'cosp_isccp_Tb', 'cosp_isccp_Tbcs', 'GPP', 'PNPP', 'zg_freeze', 'p_freeze', 'p_trop', 't_trop', 'h_trop',
    
    # # hourly
    # 'ts', 'blh', 'rlds', 'ps', 'rsns', 'rsdt', 'rsutcs', 'rsdscs', 'rsuscs', 'rsds', 'rlns', 'rlutcs', 'rldscs', 'uas', 'vas', 'sfcWind', 'tas', 'das', 'psl', 'prw', 'rlut', 'rsut', 'hfss', 'hfls', 'pr', 'clh', 'clm', 'cll', 'clt', 'clwvi', 'clivi'
    # 'rlu_t_s', 'rss_dir', 'rss_dif', 'cosp_isccp_albedo', 'cosp_isccp_tau', 'cosp_isccp_ctp', 'cosp_isccp_tcc', 'cosp_c_lcc', 'cosp_c_mcc', 'cosp_c_hcc', 'cosp_c_tcc', 'mlh', 'mlentrain', 'blentrain', 'wind_gust', 'lsrf', 'lssf', 'crf', 'csf', 'rain', 'snow', 'deep_pr', 'clvl', 'CAPE', 'CIN', 'dmvi', 'wmvi', 'fog2m', 'qt2m', 'hfms', 'huss', 'hurs'
    ]
ds_names = ['access-am3-configs', 'am3-plus4k', 'am3-climaerosol', 'am3-climaerop4k']
plt_regions = ['global']
plt_modes = ['original', 'difference']
nrow = 1 # 2 #
ncol = len(ds_names) # 3 #
if ncol==2:
    mpl.rc('font', family='Times New Roman', size=10)
elif ncol==3:
    mpl.rc('font', family='Times New Roman', size=12)
elif ncol==4:
    mpl.rc('font', family='Times New Roman', size=14)

# settings
min_lonh9, max_lonh9, min_lath9, max_lath9 = [80, 200, -60, 60]
cm_saf_varnames = {'rlut': 'LW_flux', 'rsut': 'SW_flux', 'CDNC': 'cdnc_liq',
                   'clwvi': 'lwp_allsky', 'clivi': 'iwp_allsky',}
cltypes = {
    'hcc': ['Cirrus', 'Cirrostratus', 'Deep convection'],
    'mcc': ['Altocumulus', 'Altostratus', 'Nimbostratus'],
    'lcc': ['Cumulus', 'Stratocumulus', 'Stratus'],
    'tcc': ['Cirrus', 'Cirrostratus', 'Deep convection', 'Altocumulus', 'Altostratus', 'Nimbostratus', 'Cumulus', 'Stratocumulus', 'Stratus']}
extend2 = 'both'
# regridder = {}

for ivar in vars:
    # ivar = 'vas'
    print(f'#-------------------------------- {ivar}')
    
    if ivar in ['sst', 'seaice']:
        iregion = 'ocean'
    elif ivar in ['GPP', 'PNPP']:
        iregion = 'land'
    else:
        iregion = 'global'
    
    if ivar in ['pr', 'lsrf', 'lssf', 'crf', 'csf', 'rain', 'snow', 'deep_pr', 'mlentrain', 'blentrain', 'cosp_isccp_albedo', 'cosp_isccp_ctp']:
        digit = 2
    elif ivar in ['GPP', 'PNPP']:
        digit = 3
    else:
        digit = 1
    
    ds_data = {'ann': {}, 'am': {}}
    for ids in ds_names:
        print(f'Get {ids}')
        
        if ids in ['access-am3-configs', 'am3-plus4k', 'am3-climaerosol', 'am3-climaerop4k']:
            # ids = 'access-am3-configs'
            istream = next((k for k, v in amvargroups.items() if ivar in v), None)
            if not istream is None:
                fl = sorted(glob.glob(f'cylc-run/{ids}/share/data/History_Data/netCDF/*a.p{istream}*.nc'))
                ds = xr.open_mfdataset(fl, preprocess=preprocess_amoutput, parallel=True)
                ds = ds[ivar].sel(time=slice(years, yeare))
                
                if 'lon_u' in ds.dims:
                    ds = ds.rename({'lon_u': 'lon'})
                if 'lat_v' in ds.dims:
                    ds = ds.rename({'lat_v': 'lat'})
                
                if ivar in ['pr', 'evspsbl', 'evspsblpot', 'GPP', 'PNPP', 'lsrf', 'lssf', 'crf', 'csf', 'rain', 'snow', 'deep_pr']:
                    ds *= seconds_per_d
                elif ivar in ['tas', 'ts', 'sst', 'das']:
                    ds -= zerok
                elif ivar in ['rlus', 'rluscs', 'rlut', 'rlutcs', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'hfls', 'hfss', 'rsu_trop', 'rlu_t_s']:
                    ds *= (-1)
                elif ivar in ['psl', 'p_freeze', 'p_trop', 'ps', 'cosp_isccp_ctp']:
                    ds /= 100
                elif ivar in ['huss', 'clwvi', 'clivi', 'cwp', 'qt2m']:
                    ds *= 1000
                elif ivar in ['cll', 'clm', 'clh', 'clt', 'seaice', 'cosp_isccp_tcc', 'cosp_c_lcc', 'cosp_c_mcc', 'cosp_c_hcc', 'cosp_c_tcc', 'fog2m', 'clvl']:
                    ds *= 100
                elif ivar in ['h_trop']:
                    ds /= 1000
                
                if istream in ['a']:
                    ds_data['ann'][ids] = ds.resample({'time': '1YE'}).map(time_weighted_mean).compute()
                elif istream in ['b']:
                    ds_data['ann'][ids] = ds.resample({'time': '1YE'}).mean().compute()
        elif ids == 'ERA5':
            # ids = 'ERA5'
            with open(f'data/sim/era5/mon/era5_sl_mon_alltime_{cmip6_era5_var[ivar]}.pkl', 'rb') as f:
                era5_sl_mon_alltime = pickle.load(f)
            ds_data['ann'][ids] = era5_sl_mon_alltime['ann'].sel(time=slice(years, yeare))
            if ivar in ['clwvi', 'clivi']:
                ds_data['ann'][ids] *= 1000
        
        ds_data['ann'][ids]['lon'] = ds_data['ann'][ids]['lon'] % 360
        ds_data['ann'][ids] = ds_data['ann'][ids].sortby(['lon', 'lat'])
        ds_data['am'][ids] = ds_data['ann'][ids].mean(dim='time').compute()
    
    cbar_label1 = f'{years}-{yeare} {era5_varlabels[cmip6_era5_var[ivar]]}'
    cbar_label2 = f'Difference in {era5_varlabels[cmip6_era5_var[ivar]]}'
    
    for plt_region in plt_regions:
        # plt_region = 'global'
        print(f'#---------------- {plt_region}')
        
        plt_org = {}
        plt_ann = {}
        plt_mean = {}
        for ids in ds_names:
            if plt_region == 'global':
                plt_org[ids] = ds_data['am'][ids]
                plt_ann[ids] = ds_data['ann'][ids]
            
            plt_mean[ids] = global_land_ocean_mean(plt_org[ids], iregion)
        
        plt_diff = {}
        plt_rmsd = {}
        plt_md = {}
        for ids in ds_names[1:]:
            print(f'{ids} - {ds_names[0]}')
            # if not f'{ids} - {ds_names[0]}' in regridder.keys():
            #     regridder[f'{ids} - {ds_names[0]}'] = xe.Regridder(
            #         plt_org[ids],
            #         plt_org[ds_names[0]],
            #         method='bilinear')
            # plt_diff[ids] = regridder[f'{ids} - {ds_names[0]}'](plt_org[ids]) - plt_org[ds_names[0]]
            plt_diff[ids] = regrid(plt_org[ids], plt_org[ds_names[0]]) - plt_org[ds_names[0]]
            plt_rmsd[ids] = global_land_ocean_rmsd(plt_diff[ids], iregion)
            plt_md[ids] = global_land_ocean_mean(plt_diff[ids], iregion)
            
            if ivar not in ['pr']:
                ttest_fdr_res = ttest_fdr_control(
                    # regridder[f'{ids} - {ds_names[0]}'](plt_ann[ids]),
                    regrid(plt_ann[ids], plt_ann[ds_names[0]]),
                    plt_ann[ds_names[0]])
                plt_diff[ids] = plt_diff[ids].where(ttest_fdr_res, np.nan)
        
        extend1 = 'both'
        if plt_region == 'global':
            if ivar in ['sst']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=-2, cm_max=30, cm_interval1=1, cm_interval2=2,
                    cmap='Oranges_r',)
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1,
                    cmap='BrBG')
            elif ivar in ['rsut']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=-200, cm_max=-40, cm_interval1=10, cm_interval2=20,
                    cmap='Greens',)
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10,
                    cmap='BrBG')
            elif ivar in ['rlut']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=-300, cm_max=-120, cm_interval1=10, cm_interval2=30,
                    cmap='Greens',)
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10,
                    cmap='BrBG')
            elif ivar in ['clwvi']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=0, cm_max=160, cm_interval1=10, cm_interval2=20,
                    cmap='Greens_r',)
                extend1 = 'max'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-100, cm_max=100, cm_interval1=10, cm_interval2=20,
                    cmap='BrBG_r')
            elif ivar in ['clivi']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=0, cm_max=240, cm_interval1=10, cm_interval2=20,
                    cmap='Greens_r',)
                extend1 = 'max'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-100, cm_max=100, cm_interval1=10, cm_interval2=20,
                    cmap='BrBG_r')
            elif ivar in ['CDNC']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=0, cm_max=150, cm_interval1=10, cm_interval2=20,
                    cmap='viridis_r',)
                extend1 = 'max'
            elif ivar in ['clh', 'clm', 'cll', 'cll_mol', 'cll_rol', 'cosp_c_lcc', 'cosp_c_mcc', 'cosp_c_hcc', 'clvl', 'fog2m']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=10,
                    cmap='Blues_r',)
                extend1 = 'neither'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10,
                    cmap='BrBG_r',)
            elif ivar in ['clt', 'cosp_isccp_tcc', 'cosp_c_tcc']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=10,
                    cmap='Blues_r',)
                extend1 = 'neither'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10,
                    cmap='BrBG_r',)
            elif ivar in ['hfls']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=-240, cm_max=0, cm_interval1=20, cm_interval2=40, cmap='Greens',)
                extend1 = 'both'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-60, cm_max=60, cm_interval1=10, cm_interval2=10, cmap='BrBG')
            elif ivar in ['hfss']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=-40, cm_max=0, cm_interval1=2.5, cm_interval2=5, cmap='Greens')
                extend1 = 'both'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-20, cm_max=20, cm_interval1=2.5, cm_interval2=5, cmap='BrBG',)
            elif ivar in ['rsutcs']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=-150, cm_max=-50, cm_interval1=5, cm_interval2=10,
                    cmap='Greens',)
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    # cm_min=-10, cm_max=10, cm_interval1=1, cm_interval2=2,
                    cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10,
                    cmap='BrBG')
            elif ivar in ['rlutcs']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=-320, cm_max=-160, cm_interval1=10, cm_interval2=20,
                    cmap='Greens',)
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10,
                    cmap='BrBG')
            elif ivar in ['rsutcl']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=-150, cm_max=-50, cm_interval1=5, cm_interval2=10,
                    cmap='Greens',)
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    # cm_min=-10, cm_max=10, cm_interval1=1, cm_interval2=2,
                    cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10,
                    cmap='BrBG')
            elif ivar in ['rlutcl']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=-290, cm_max=-210, cm_interval1=5, cm_interval2=10,
                    cmap='viridis',)
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    # cm_min=-10, cm_max=10, cm_interval1=1, cm_interval2=2,
                    cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10,
                    cmap='BrBG')
            elif ivar in ['blh', 'zmla', 'mlh']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=0, cm_max=1200, cm_interval1=100, cm_interval2=200,
                    cmap='Greens_r',)
                extend1 = 'max'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-600, cm_max=600, cm_interval1=50, cm_interval2=200,
                    cmap='BrBG_r')
            elif ivar in ['LCL']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=0, cm_max=2800, cm_interval1=100, cm_interval2=400,
                    cmap='Greens_r',)
                extend1 = 'max'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-200, cm_max=200, cm_interval1=50, cm_interval2=50,
                    cmap='BrBG_r')
            elif ivar in ['LTS']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=4, cm_max=18, cm_interval1=1, cm_interval2=2,
                    cmap='pink')
                extend1 = 'both'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-2, cm_max=2, cm_interval1=0.5, cm_interval2=0.5,
                    cmap='BrBG')
            elif ivar in ['EIS']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=0, cm_max=10, cm_interval1=1, cm_interval2=1,
                    cmap='pink')
                extend1 = 'both'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-2, cm_max=2, cm_interval1=0.5, cm_interval2=0.5,
                    cmap='BrBG')
            elif ivar in ['ECTEI']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=-8, cm_max=8, cm_interval1=1, cm_interval2=2,
                    cmap='PuOr')
                extend1 = 'both'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-2, cm_max=2, cm_interval1=0.5, cm_interval2=0.5,
                    cmap='BrBG')
            elif ivar in ['pr', 'lsrf', 'lssf', 'crf', 'csf', 'rain', 'snow', 'deep_pr']:
                pltlevel1 = np.array([0, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20,])
                pltticks1 = np.array([0, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20,])
                pltnorm1 = BoundaryNorm(pltlevel1, ncolors=len(pltlevel1)-1, clip=True)
                pltcmp1 = plt.get_cmap('Blues', len(pltlevel1)-1)
                extend1 = 'max'
                pltlevel2 = np.array([-6, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 6])
                pltticks2 = np.array([-6, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 6])
                pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
                pltcmp2 = plt.get_cmap('BrBG', len(pltlevel2)-1)
            elif ivar in ['seaice']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=20,
                    cmap='Blues_r',)
                extend1 = 'neither'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-50, cm_max=50, cm_interval1=10, cm_interval2=20,
                    cmap='BrBG')
            elif ivar in ['rsn_trop']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=50, cm_max=350, cm_interval1=25, cm_interval2=50,
                    cmap='Greens_r',)
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10,
                    cmap='BrBG')
            elif ivar in ['rsu_trop']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=-160, cm_max=-40, cm_interval1=10, cm_interval2=20,
                    cmap='Greens',)
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10,
                    cmap='BrBG')
            elif ivar in ['rln_trop']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=-300, cm_max=-100, cm_interval1=10, cm_interval2=40,
                    cmap='Greens',)
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10,
                    cmap='BrBG')
            elif ivar in ['rld_trop']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=10, cm_max=30, cm_interval1=2, cm_interval2=2,
                    cmap='Greens_r',)
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-6, cm_max=6, cm_interval1=1, cm_interval2=2,
                    cmap='BrBG')
            elif ivar in ['cosp_isccp_Tb', 'cosp_isccp_Tbcs']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=220, cm_max=300, cm_interval1=5, cm_interval2=10,
                    cmap='Greens_r',)
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-10, cm_max=10, cm_interval1=1, cm_interval2=2,
                    cmap='BrBG')
            elif ivar in ['GPP', 'PNPP']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=0, cm_max=0.01, cm_interval1=0.001, cm_interval2=0.002,
                    cmap='Greens_r',)
                extend1 = 'max'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-0.004, cm_max=0.004, cm_interval1=0.001, cm_interval2=0.002,
                    cmap='BrBG')
            elif ivar in ['zg_freeze']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=0, cm_max=6000, cm_interval1=250, cm_interval2=1000,
                    cmap='Greens_r',)
                extend1 = 'max'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-1200, cm_max=1200, cm_interval1=100, cm_interval2=400,
                    cmap='BrBG')
            elif ivar in ['p_freeze']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=500, cm_max=1200, cm_interval1=50, cm_interval2=100,
                    cmap='Greens_r',)
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-120, cm_max=120, cm_interval1=10, cm_interval2=40,
                    cmap='BrBG')
            elif ivar in ['p_trop']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=60, cm_max=300, cm_interval1=20, cm_interval2=40,
                    cmap='Greens_r',)
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=10,
                    cmap='BrBG')
            elif ivar in ['t_trop']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=180, cm_max=220, cm_interval1=2, cm_interval2=8,
                    cmap='Greens_r',)
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-10, cm_max=10, cm_interval1=2, cm_interval2=2,
                    cmap='BrBG')
            elif ivar in ['h_trop']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=10, cm_max=20, cm_interval1=0.5, cm_interval2=1,
                    cmap='Greens_r',)
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-2, cm_max=2, cm_interval1=0.5, cm_interval2=0.5,
                    cmap='BrBG')
            elif ivar in ['ts', 'tas', 'das']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=-36, cm_max=36, cm_interval1=2, cm_interval2=8,
                    cmap='RdBu',)
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-6, cm_max=6, cm_interval1=1, cm_interval2=1,
                    cmap='BrBG')
            elif ivar in ['rlds']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=100, cm_max=500, cm_interval1=25, cm_interval2=50,
                    cmap='Greens_r',)
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10,
                    cmap='BrBG')
            elif ivar in ['rlu_t_s']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=0, cm_max=280, cm_interval1=10, cm_interval2=40,
                    cmap='Greens_r',)
                extend1 = 'max'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10,
                    cmap='BrBG')
            elif ivar in ['ps']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=800, cm_max=1040, cm_interval1=10, cm_interval2=40,
                    cmap='Greens_r',)
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-10, cm_max=10, cm_interval1=1, cm_interval2=2,
                    cmap='BrBG')
            elif ivar in ['rsns']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=0, cm_max=280, cm_interval1=10, cm_interval2=40,
                    cmap='Greens_r',)
                extend1 = 'max'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10,
                    cmap='BrBG')
            elif ivar in ['rsdt']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=180, cm_max=420, cm_interval1=10, cm_interval2=20,
                    cmap='Greens_r',)
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10,
                    cmap='BrBG')
            elif ivar in ['rsds', 'rsdscs']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=0, cm_max=320, cm_interval1=10, cm_interval2=40,
                    cmap='Greens_r',)
                extend1 = 'max'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10,
                    cmap='BrBG')
            elif ivar in ['rsuscs']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=-160, cm_max=0, cm_interval1=10, cm_interval2=20,
                    cmap='Greens',)
                extend1 = 'min'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10,
                    cmap='BrBG')
            elif ivar in ['rss_dir']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=0, cm_max=280, cm_interval1=10, cm_interval2=40,
                    cmap='Greens_r',)
                extend1 = 'max'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10,
                    cmap='BrBG')
            elif ivar in ['rss_dif']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=40, cm_max=100, cm_interval1=5, cm_interval2=10,
                    cmap='Greens_r',)
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10,
                    cmap='BrBG')
            elif ivar in ['rlns']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=-180, cm_max=-20, cm_interval1=10, cm_interval2=20,
                    cmap='Greens',)
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10,
                    cmap='BrBG')
            elif ivar in ['rldscs']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=100, cm_max=500, cm_interval1=25, cm_interval2=50,
                    cmap='Greens_r',)
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10,
                    cmap='BrBG')
            elif ivar in ['cosp_isccp_albedo']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=0, cm_max=0.28, cm_interval1=0.02, cm_interval2=0.04,
                    cmap='Greens_r',)
                extend1 = 'max'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-0.04, cm_max=0.04, cm_interval1=0.01, cm_interval2=0.02,
                    cmap='BrBG')
            elif ivar in ['cosp_isccp_tau']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=0, cm_max=10, cm_interval1=0.5, cm_interval2=1,
                    cmap='Greens_r',)
                extend1 = 'max'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-3, cm_max=3, cm_interval1=0.5, cm_interval2=1,
                    cmap='BrBG')
            elif ivar in ['cosp_isccp_ctp']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=0, cm_max=360, cm_interval1=20, cm_interval2=40,
                    cmap='Greens_r',)
                extend1 = 'max'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-24, cm_max=24, cm_interval1=4, cm_interval2=8,
                    cmap='BrBG')
            elif ivar in ['uas', 'vas']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=-10, cm_max=10, cm_interval1=1, cm_interval2=2,
                    cmap='RdBu',)
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1,
                    cmap='BrBG')
            elif ivar in ['sfcWind', 'wind_gust']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=0, cm_max=14, cm_interval1=1, cm_interval2=2,
                    cmap='Greens_r',)
                extend1 = 'max'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1,
                    cmap='BrBG')
            elif ivar in ['huss', 'qt2m']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=0, cm_max=20, cm_interval1=1, cm_interval2=2,
                    cmap='Greens_r',)
                extend1 = 'max'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1,
                    cmap='BrBG')
            elif ivar in ['hurs']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=20, cm_max=120, cm_interval1=5, cm_interval2=10,
                    cmap='Greens_r',)
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-20, cm_max=20, cm_interval1=2, cm_interval2=4,
                    cmap='BrBG')
            elif ivar in ['hfms']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=-20, cm_max=0, cm_interval1=1, cm_interval2=2, cmap='Greens')
                extend1 = 'both'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-10, cm_max=10, cm_interval1=1, cm_interval2=2, cmap='BrBG',)
            elif ivar in ['mlentrain', 'blentrain']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=0, cm_max=0.2, cm_interval1=0.01, cm_interval2=0.04,
                    cmap='Greens_r',)
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-0.06, cm_max=0.06, cm_interval1=0.01, cm_interval2=0.02,
                    cmap='BrBG')
            elif ivar in ['psl']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=960, cm_max=1040, cm_interval1=2, cm_interval2=8,
                    cmap='Greens_r',)
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-10, cm_max=10, cm_interval1=1, cm_interval2=2,
                    cmap='BrBG')
            elif ivar in ['CAPE']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=0, cm_max=4000, cm_interval1=200, cm_interval2=800,
                    cmap='Greens_r',)
                extend1 = 'max'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-1000, cm_max=1000, cm_interval1=100, cm_interval2=400,
                    cmap='BrBG')
            elif ivar in ['CIN']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=-1000, cm_max=0, cm_interval1=50, cm_interval2=200, cmap='Greens')
                extend1 = 'min'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-100, cm_max=100, cm_interval1=10, cm_interval2=40, cmap='BrBG',)
            elif ivar in ['dmvi', 'wmvi']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=5500, cm_max=10500, cm_interval1=500, cm_interval2=1000, cmap='Greens_r')
                extend1 = 'both'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-160, cm_max=160, cm_interval1=20, cm_interval2=40, cmap='BrBG',)
            elif ivar in ['prw']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=0, cm_max=60, cm_interval1=2.5, cm_interval2=5, cmap='Greens_r')
                extend1 = 'max'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-16, cm_max=16, cm_interval1=2, cm_interval2=4, cmap='BrBG_r',)
            else:
                print(f'Warning: no colormap specified; automatically setup')
                all_vals = np.concatenate([da.values.ravel() for da in plt_org.values()])
                vmin = np.nanmin(all_vals)
                vmax = np.nanmax(all_vals)
                pltnorm1 = mcolors.Normalize(vmin=vmin, vmax=vmax)
                pltcmp1 = plt.get_cmap('viridis')
                pltticks1 = np.linspace(vmin, vmax, 7)
                all_vals2 = np.concatenate([da.values.ravel() for da in plt_diff.values()])
                vmin2 = np.nanmax(abs(all_vals2)) * (-1)
                vmax2 = np.nanmax(abs(all_vals2))
                pltnorm2 = mcolors.Normalize(vmin=vmin2, vmax=vmax2)
                pltcmp2 = plt.get_cmap('BrBG')
                pltticks2 = np.linspace(vmin2, vmax2, 7)
        
        for plt_mode in plt_modes:
            print(f'#-------- {plt_mode}')
            
            if plt_region == 'global':
                # plt_region = 'global'
                fig, axs = plt.subplots(
                    nrow, ncol,
                    figsize=np.array([6.6*ncol, 3.3*nrow+3]) / 2.54,
                    subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
                    gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)
                fm_bottom = 2.4 / (3.3*nrow+3)
                for irow in range(nrow):
                    for jcol in range(ncol):
                        if ncol == 1:
                            axs = globe_plot(ax_org=axs)
                            if iregion=='ocean':
                                axs.add_feature(cfeature.LAND, color='white', zorder=2, edgecolor=None,lw=0)
                            elif iregion=='land':
                                axs.add_feature(cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)
                        elif nrow == 1:
                            axs[jcol] = globe_plot(ax_org=axs[jcol])
                            if iregion=='ocean':
                                axs[jcol].add_feature(cfeature.LAND, color='white', zorder=2, edgecolor=None,lw=0)
                            elif iregion=='land':
                                axs[jcol].add_feature(cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)
                        else:
                            axs[irow, jcol] = globe_plot(ax_org=axs[irow, jcol])
                            if iregion=='ocean':
                                axs[irow, jcol].add_feature(cfeature.LAND, color='white', zorder=2, edgecolor=None,lw=0)
                            elif iregion=='land':
                                axs[irow, jcol].add_feature(cfeature.OCEAN, color='white', zorder=2, edgecolor=None,lw=0)
            
            plt_colnames = [f'{am3_label[ds_names[0]]}']
            plt_text = [f'Mean: {str(np.round(plt_mean[ds_names[0]], digit))}']
            if plt_mode in ['original']:
                plt_colnames += [f'{am3_label[ids]}' for ids in ds_names[1:]]
                plt_text += [f'{str(np.round(plt_mean[ids], digit))}' for ids in ds_names[1:]]
            elif plt_mode in ['difference']:
                plt_colnames += [f'{am3_label[ids]} - {am3_label[ds_names[0]]}' for ids in ds_names[1:]]
                plt_text += [f'RMSD: {str(np.round(plt_rmsd[ds_names[1]], digit))}, MD: {str(np.round(plt_md[ds_names[1]], digit))}']
                plt_text += [f'{str(np.round(plt_rmsd[ids], digit))}, {str(np.round(plt_md[ids], digit))}' for ids in ds_names[2:]]
            
            if nrow == 1:
                if ncol == 1:
                    cbar_label1 = f'{plt_colnames[0]} {cbar_label1}'
                    axs.text(
                        0, -0.02, plt_text[jcol], ha='left', va='top',
                        transform=axs.transAxes, size=8)
                else:
                    for jcol in range(ncol):
                        axs[jcol].text(
                            0, 1.02,
                            f'({string.ascii_lowercase[jcol]}) {plt_colnames[jcol]}',
                            ha='left',va='bottom',transform=axs[jcol].transAxes)
                        axs[jcol].text(
                            0, -0.02, plt_text[jcol], ha='left', va='top',
                            transform=axs[jcol].transAxes,)
            else:
                for irow in range(nrow):
                    for jcol in range(ncol):
                        axs[irow, jcol].text(
                            0, 1.02,
                            f'({string.ascii_lowercase[irow * ncol + jcol]}) {plt_colnames[irow * ncol + jcol]}',
                            ha='left',va='bottom',
                            transform=axs[irow, jcol].transAxes)
                        axs[irow, jcol].text(
                            0, -0.02, plt_text[irow * ncol + jcol],
                            ha='left', va='top',
                            transform=axs[irow, jcol].transAxes)
            
            if nrow == 1:
                if ncol == 1:
                    plt_mesh1 = axs.pcolormesh(
                        plt_org[ds_names[0]].lon,
                        plt_org[ds_names[0]].lat,
                        plt_org[ds_names[0]],
                        norm=pltnorm1, cmap=pltcmp1,
                        transform=ccrs.PlateCarree(), zorder=1)
                else:
                    plt_mesh1 = axs[0].pcolormesh(
                        plt_org[ds_names[0]].lon,
                        plt_org[ds_names[0]].lat,
                        plt_org[ds_names[0]],
                        norm=pltnorm1, cmap=pltcmp1,
                        transform=ccrs.PlateCarree(), zorder=1)
            else:
                plt_mesh1 = axs[0, 0].pcolormesh(
                    plt_org[ds_names[0]].lon,
                    plt_org[ds_names[0]].lat,
                    plt_org[ds_names[0]],
                    norm=pltnorm1, cmap=pltcmp1,
                    transform=ccrs.PlateCarree(), zorder=1)
            
            if plt_mode in ['original']:
                if nrow == 1:
                    for jcol in range(ncol-1):
                        plt_mesh1 = axs[jcol+1].pcolormesh(
                            plt_org[ds_names[jcol+1]].lon,
                            plt_org[ds_names[jcol+1]].lat,
                            plt_org[ds_names[jcol+1]],
                            norm=pltnorm1, cmap=pltcmp1,
                            transform=ccrs.PlateCarree(),zorder=1)
                else:
                    for irow in range(nrow):
                        for jcol in range(ncol):
                            if ((irow != 0) | (jcol != 0)):
                                plt_mesh1 = axs[irow, jcol].pcolormesh(
                                    plt_org[ds_names[irow * ncol + jcol]].lon,
                                    plt_org[ds_names[irow * ncol + jcol]].lat,
                                    plt_org[ds_names[irow * ncol + jcol]],
                                    norm=pltnorm1, cmap=pltcmp1,
                                    transform=ccrs.PlateCarree(),zorder=1)
                if ncol == 1:
                    cbar1 = fig.colorbar(
                        plt_mesh1,#cm.ScalarMappable(norm=pltnorm1, cmap=pltcmp1),#
                        format=remove_trailing_zero_pos,
                        orientation="horizontal", ticks=pltticks1, extend=extend1,
                        cax=fig.add_axes([0.05, 0.2, 0.95, 0.04]))
                    cbar1.ax.tick_params(labelsize=8)
                    cbar1.ax.set_xlabel(cbar_label1, fontsize=8)
                else:
                    cbar1 = fig.colorbar(
                        plt_mesh1,#cm.ScalarMappable(norm=pltnorm1, cmap=pltcmp1),#
                        format=remove_trailing_zero_pos,
                        orientation="horizontal", ticks=pltticks1, extend=extend1,
                        cax=fig.add_axes([0.26, fm_bottom*0.6, 0.48, fm_bottom / 6]))
                    cbar1.ax.set_xlabel(cbar_label1)
            elif plt_mode in ['difference']:
                if nrow == 1:
                    for jcol in range(ncol-1):
                        plt_mesh2 = axs[jcol+1].pcolormesh(
                            plt_diff[ds_names[jcol+1]].lon,
                            plt_diff[ds_names[jcol+1]].lat,
                            plt_diff[ds_names[jcol+1]],
                            norm=pltnorm2, cmap=pltcmp2,
                            transform=ccrs.PlateCarree(),zorder=1)
                else:
                    for irow in range(nrow):
                        for jcol in range(ncol):
                            if ((irow != 0) | (jcol != 0)):
                                plt_mesh2 = axs[irow, jcol].pcolormesh(
                                    plt_diff[ds_names[irow * ncol + jcol]].lon,
                                    plt_diff[ds_names[irow * ncol + jcol]].lat,
                                    plt_diff[ds_names[irow * ncol + jcol]],
                                    norm=pltnorm2, cmap=pltcmp2,
                                    transform=ccrs.PlateCarree(),zorder=1)
                cbar1 = fig.colorbar(
                    plt_mesh1,#cm.ScalarMappable(norm=pltnorm1, cmap=pltcmp1),#
                    format=remove_trailing_zero_pos,
                    orientation="horizontal", ticks=pltticks1, extend=extend1,
                    cax=fig.add_axes([0.01, fm_bottom*0.6, 0.48, fm_bottom / 6]))
                cbar1.ax.set_xlabel(cbar_label1)
                cbar2 = fig.colorbar(
                    plt_mesh2,#cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2),#
                    format=remove_trailing_zero_pos,
                    orientation="horizontal", ticks=pltticks2, extend=extend2,
                    cax=fig.add_axes([0.51, fm_bottom*0.6, 0.48, fm_bottom / 6]))
                cbar2.ax.set_xlabel(cbar_label2)
            
            opng = f'figures/4_um/4.2_access_am3/4.2.0_sim_obs/4.2.0.0 {ivar} {', '.join(ds_names)} {plt_region} {plt_mode} {years}-{yeare}.png'
            fig.subplots_adjust(left=0.005, right=0.995, bottom=fm_bottom, top=0.92)
            fig.savefig(opng, dpi=600)






# endregion


