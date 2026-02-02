

# qsub -I -q copyq -P v46 -l walltime=4:00:00,ncpus=1,mem=192GB,jobfs=100MB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22+gdata/py18+gdata/gx60+gdata/xp65+gdata/qx55+gdata/rv74+gdata/al33+gdata/rr3+gdata/hr22+scratch/gx60+scratch/gb02+gdata/gb02


# region import packages

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
from metpy.calc import pressure_to_height_std
from metpy.units import units
import pickle
import glob

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=12)
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})
from cartopy.mpl.ticker import LatitudeFormatter
from matplotlib.ticker import AutoMinorLocator
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
    remove_trailing_zero_pos,
    )

from namelist import (
    seconds_per_d,
    zerok,
    era5_varlabels,
    cmip6_era5_var,
    )

from component_plot import (
    plt_mesh_pars,
)

from calculations import (
    time_weighted_mean,
    global_land_ocean_rmsd,
    global_land_ocean_mean,
    regrid,)

from statistics0 import (
    ttest_fdr_control,)

from um_postprocess import (
    interp_to_pressure_levels, preprocess_amoutput, amvargroups, am3_label)

# endregion


# region plot obs and sim am sl
# Memory Used: 38.13GB, Walltime Used: 00:50:53 for 4 ds
# Memory Used: 42.17GB, Walltime Used: 00:59:30 for 3 ds

# options
years = '1983'; yeare = '1987'
vars = [
    'ncloud'
    
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
            # ids = 'am3-plus4k'
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
                elif istream in ['c']:
                    if ivar in ['ncloud']:
                        ds = ds.where(ds != 0, np.nan)
                        ds_data['ann'][ids] = ds.resample({'time': '1YE'}).mean(skipna=True).mean(dim='theta85', skipna=True).compute() / 1e+6
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
            elif ivar in ['ncloud']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=0, cm_max=300, cm_interval1=15, cm_interval2=30, cmap='pink')
                extend1 = 'max'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-100, cm_max=100, cm_interval1=20, cm_interval2=20, cmap='BrBG_r',)
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


# region plot obs and sim am zm pl

# options
years = '1983'; yeare = '1987'
vars = [
    'TCF', 'qcf', 'qcl', 'qr', 'hus', 'hur', 'ta', 'ua', 'va', 'wap', 'zg', 'pa', 'theta', 'wa', 'rsu', 'rsd', 'rsucs', 'rsdcs', 'rlu', 'rld', 'rlucs', 'rldcs', 'clslw', 'rain_evap', 'updraught_mf', 'downdraught_mf', 'deepc_mf', 'congestusc_mf', 'shallowc_mf', 'midc_mf', 'qc', 'qt', 'DMS',
    
    # monthly
    # 'TCF', 'qcf', 'qcl', 'qr', 'hus', 'hur', 'ta', 'ua', 'va', 'wap', 'zg'
    # 'pa', 'theta', 'wa', 'rsu', 'rsd', 'rsucs', 'rsdcs', 'rlu', 'rld', 'rlucs', 'rldcs', 'clslw', 'rain_evap', 'updraught_mf', 'downdraught_mf', 'deepc_mf', 'congestusc_mf', 'shallowc_mf', 'midc_mf', 'qc', 'qt', 'DMS',
    ]
ds_names = ['ERA5', 'access-am3-configs', 'am3-plus4k', 'am3-climaerosol', 'am3-climaerop4k'] #
plt_regions = ['global']
plt_modes = ['original', 'difference']
nrow = 1 # 2 #
ncol = len(ds_names) # 3 #
if ncol<=2:
    mpl.rc('font', family='Times New Roman', size=10)
elif ncol==3:
    mpl.rc('font', family='Times New Roman', size=12)
elif ncol>=4:
    mpl.rc('font', family='Times New Roman', size=14)

# settings
extend2 = 'both'
ptop = 100
plevs_hpa = np.arange(1000, ptop-1e-4, -25)

for ivar in vars:
    # ivar = 'TCF'
    print(f'#-------------------------------- {ivar}')
    
    ds_data = {'ann': {}, 'am': {}}
    for ids in ds_names:
        print(f'Get {ids}')
        
        if ids in ['access-am3-configs', 'am3-plus4k', 'am3-climaerosol', 'am3-climaerop4k']:
            # ids = 'am3-plus4k'
            istream = next((k for k, v in amvargroups.items() if ivar in v), None)
            if not istream is None:
                fl = sorted(glob.glob(f'cylc-run/{ids}/share/data/History_Data/netCDF/*a.p{istream}*.nc'))
                ds = xr.open_mfdataset(fl, preprocess=preprocess_amoutput, parallel=True).sel(time=slice(years, yeare))
                
                if ivar in ['ua']:
                    ds_data['ann'][ids] = interp_to_pressure_levels(
                        regrid(ds[ivar].rename({'rho85': 'theta85', 'lon_u': 'lon'}), ds['pa']),
                        ds['pa']/100,
                        plevs_hpa, theta='theta85').resample({'time': '1YE'}).map(time_weighted_mean).mean(dim='lon').compute()
                elif ivar in ['va']:
                    ds_data['ann'][ids] = interp_to_pressure_levels(
                        regrid(ds[ivar].rename({'rho85': 'theta85', 'lat_v': 'lat'}), ds['pa']),
                        ds['pa']/100,
                        plevs_hpa, theta='theta85').resample({'time': '1YE'}).map(time_weighted_mean).mean(dim='lon').compute()
                elif ivar in ['rsu', 'rsd', 'rsucs', 'rsdcs', 'rlu', 'rld', 'rlucs', 'rldcs']:
                    ds_data['ann'][ids] = interp_to_pressure_levels(ds[ivar].rename({'rho85': 'theta85'}), ds['pa']/100, plevs_hpa, theta='theta85').resample({'time': '1YE'}).map(time_weighted_mean).mean(dim='lon').compute()
                else:
                    ds_data['ann'][ids] = interp_to_pressure_levels(ds[ivar], ds['pa']/100, plevs_hpa, theta='theta85').resample({'time': '1YE'}).map(time_weighted_mean).mean(dim='lon').compute()
            
            if ivar in ['hus', 'qcf', 'qcl', 'qr', 'qs', 'qc', 'qt', 'clslw', 'qg']:
                ds_data['ann'][ids] *= 1000
            elif ivar in ['ta', 'theta', 'theta_e']:
                ds_data['ann'][ids] -= zerok
            elif ivar in ['ACF', 'BCF', 'TCF']:
                ds_data['ann'][ids] *= 100
            elif ivar in ['rsu', 'rsucs', 'rlu', 'rlucs', ]:
                ds_data['ann'][ids] *= (-1)
            # elif ivar in ['zg', ]:
            #     ds_data['ann'][ids] /= 1000
            del ds
        elif ids=='ERA5':
            # ids = 'ERA5'
            with open(f'data/sim/era5/mon/era5_pl_mon_alltime_{cmip6_era5_var[ivar]}.pkl', 'rb') as f:
                era5_pl_mon_alltime = pickle.load(f)
            ds_data['ann'][ids] = era5_pl_mon_alltime['ann'].sel(time=slice(years, yeare)).mean(dim='lon').rename({'level': 'pressure'}).compute()
            if ivar in ['qcf', 'qcl', 'qr']:
                ds_data['ann'][ids] *= 1000
            del era5_pl_mon_alltime
        
        ds_data['ann'][ids] = ds_data['ann'][ids].sortby(['lat']).transpose(..., 'pressure', 'lat')
        ds_data['am'][ids] = ds_data['ann'][ids].mean(dim='time').compute()
    
    cbar_label1 = f'{years}-{yeare} {era5_varlabels[cmip6_era5_var[ivar]]}'
    cbar_label2 = f'Difference in {era5_varlabels[cmip6_era5_var[ivar]]}'
    
    for plt_region in plt_regions:
        # plt_region = 'global'
        print(f'#---------------- {plt_region}')
        
        plt_org = {}
        plt_ann = {}
        for ids in ds_names:
            if plt_region == 'global':
                plt_org[ids] = ds_data['am'][ids]
                plt_ann[ids] = ds_data['ann'][ids]
        
        plt_diff = {}
        for ids in ds_names[1:]:
            # ids = ds_names[1]
            print(f'{ids} - {ds_names[0]}')
            plt_diff[ids] = plt_org[ids] - plt_org[ds_names[0]].interp(lat=plt_org[ids].lat, pressure=plt_org[ids].pressure)
            
            if ivar not in ['pr']:
                ttest_fdr_res = ttest_fdr_control(
                    plt_ann[ids],
                    plt_ann[ds_names[0]].interp(lat=plt_ann[ids].lat, pressure=plt_ann[ids].pressure))
                plt_diff[ids] = plt_diff[ids].where(ttest_fdr_res, np.nan)
        
        extend1 = 'both'
        if plt_region == 'global':
            if ivar in ['hus', 'qt']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=0, cm_max=20, cm_interval1=1, cm_interval2=2, cmap='Blues_r')
                extend1 = 'max'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='BrBG_r')
                # pltlevel = np.array([0, 0.1, 0.2, 0.5, 1, 2, 4, 8, 12, 16, 20])
                # pltticks = np.array([0, 0.1, 0.2, 0.5, 1, 2, 4, 8, 12, 16, 20])
                # pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
                # pltcmp = plt.get_cmap('Blues', len(pltlevel)-1)
                # extend1 = 'max'
                # pltlevel2 = np.array([-1.5, -1, -0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5, 1, 1.5])
                # pltticks2 = np.array([-1.5, -1, -0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5, 1, 1.5])
                # pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
                # pltcmp2 = plt.get_cmap('BrBG', len(pltlevel2)-1)
            elif ivar == 'ta':
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=-60, cm_max=24, cm_interval1=4, cm_interval2=12, cmap='RdBu', asymmetric=True)
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-10, cm_max=10, cm_interval1=1, cm_interval2=2, cmap='BrBG')
            elif ivar == 'ua':
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=-8, cm_max=32, cm_interval1=2, cm_interval2=4, cmap='PuOr', asymmetric=True)
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-4, cm_max=4, cm_interval1=1, cm_interval2=1, cmap='BrBG')
            elif ivar == 'va':
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=-6, cm_max=6, cm_interval1=1, cm_interval2=2, cmap='PuOr')
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-1, cm_max=1, cm_interval1=0.2, cm_interval2=0.4, cmap='BrBG')
            elif ivar == 'wap':
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=-0.04, cm_max=0.04, cm_interval1=0.01, cm_interval2=0.01, cmap='PuOr')
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-0.01, cm_max=0.01, cm_interval1=0.002, cm_interval2=0.004, cmap='BrBG')
            elif ivar == 'zg':
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=0, cm_max=20000, cm_interval1=2000, cm_interval2=4000, cmap='viridis_r')
                extend1 = 'max'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-500, cm_max=500, cm_interval1=100, cm_interval2=200, cmap='BrBG')
            elif ivar == 'hur':
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=30, cm_max=100, cm_interval1=5, cm_interval2=10, cmap='Blues_r')
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-12, cm_max=12, cm_interval1=2, cm_interval2=4, cmap='BrBG_r')
            elif ivar in ['ACF', 'BCF', 'TCF']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=0, cm_max=40, cm_interval1=5, cm_interval2=5, cmap='Blues_r')
                extend1 = 'max'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-15, cm_max=15, cm_interval1=3, cm_interval2=3, cmap='BrBG_r')
            elif ivar == 'theta':
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=20, cm_max=60, cm_interval1=2, cm_interval2=8, cmap='Oranges_r')
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-10, cm_max=10, cm_interval1=1, cm_interval2=2, cmap='BrBG')
            elif ivar == 'theta_e':
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=50, cm_max=80, cm_interval1=2.5, cm_interval2=5, cmap='Oranges_r')
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-6, cm_max=6, cm_interval1=1, cm_interval2=2, cmap='BrBG')
            elif ivar in ['qr']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=0, cm_max=0.01, cm_interval1=0.001, cm_interval2=0.002, cmap='Blues_r')
                extend1 = 'max'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-0.005, cm_max=0.005, cm_interval1=0.001, cm_interval2=0.002, cmap='BrBG_r')
            elif ivar in ['qcl', 'qc']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=0, cm_max=0.08, cm_interval1=0.01, cm_interval2=0.01, cmap='Blues_r')
                extend1 = 'max'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-0.02, cm_max=0.02, cm_interval1=0.005, cm_interval2=0.005, cmap='BrBG_r')
            elif ivar in ['qcf']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=0, cm_max=0.03, cm_interval1=0.005, cm_interval2=0.005, cmap='Blues_r')
                extend1 = 'max'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-0.012, cm_max=0.012, cm_interval1=0.002, cm_interval2=0.004, cmap='BrBG_r')
            elif ivar in ['clslw', 'qcf', 'qs', 'qg']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=0, cm_max=0.04, cm_interval1=0.005, cm_interval2=0.005, cmap='Blues_r')
                extend1 = 'max'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-0.012, cm_max=0.012, cm_interval1=0.002, cm_interval2=0.004, cmap='BrBG_r')
            elif ivar in ['rld', 'rldcs']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=20, cm_max=420, cm_interval1=20, cm_interval2=40,
                    cmap='Greens_r',)
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10,
                    cmap='BrBG')
            elif ivar in ['rlu', 'rlucs']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=-480, cm_max=-120, cm_interval1=20, cm_interval2=40,
                    cmap='Greens',)
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10,
                    cmap='BrBG')
            elif ivar in ['rsd', 'rsdcs']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=80, cm_max=400, cm_interval1=20, cm_interval2=40,
                    cmap='Greens_r',)
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10,
                    cmap='BrBG')
            elif ivar in ['rsu', 'rsucs']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=-120, cm_max=0, cm_interval1=10, cm_interval2=20,
                    cmap='Greens',)
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10,
                    cmap='BrBG')
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
                fig, axs = plt.subplots(
                    nrow, ncol,
                    figsize=np.array([6.6*ncol+2, 5.5*nrow+4])/2.54,
                    sharey=True, sharex=True,
                    gridspec_kw={'hspace': 0.01, 'wspace': 0.05},)
                fm_bottom = 3 / (5.5*nrow+4)
                fm_top = 1 - 1/(5.5*nrow+4)
                fm_left = 2 / (6.6*ncol+2)
                
                plt_colnames = [f'{am3_label[ds_names[0]]}']
                if plt_mode in ['original']:
                    plt_colnames += [f'{am3_label[ids]}' for ids in ds_names[1:]]
                elif plt_mode in ['difference']:
                    plt_colnames += [f'{am3_label[ids]} - {am3_label[ds_names[0]]}' for ids in ds_names[1:]]
                
                for irow in range(nrow):
                    for jcol in range(ncol):
                        if nrow == 1:
                            if ncol == 1:
                                axs.text(
                                    0, -0.02, plt_colnames[jcol],
                                    ha='left', va='top',
                                    transform=axs.transAxes)
                                plt_mesh1 = axs.pcolormesh(
                                    plt_org[ds_names[0]].lat,
                                    plt_org[ds_names[0]].pressure,
                                    plt_org[ds_names[0]],
                                    norm=pltnorm1, cmap=pltcmp1, zorder=1)
                                
                                axs.invert_yaxis()
                                axs.set_ylim(1000, ptop)
                                axs.yaxis.set_minor_locator(AutoMinorLocator(2))
                                
                                axs.set_xlim(-90, 90)
                                axs.set_xticks(np.arange(-60, 61, 60))
                                axs.xaxis.set_minor_locator(AutoMinorLocator(2))
                                axs.xaxis.set_major_formatter(LatitudeFormatter(degree_symbol='° '))
                                
                                axs.grid(True, which='both', lw=0.5, c='gray', alpha=0.5, ls='--')
                                axs.set_ylabel(r'Pressure [$hPa$]')
                            else:
                                axs[jcol].text(
                                    0, 1.02,
                                    f'({string.ascii_lowercase[jcol]}) {plt_colnames[jcol]}',
                                    ha='left',va='bottom',
                                    transform=axs[jcol].transAxes)
                                if jcol==0:
                                    plt_mesh1 = axs[0].pcolormesh(
                                        plt_org[ds_names[0]].lat,
                                        plt_org[ds_names[0]].pressure,
                                        plt_org[ds_names[0]],
                                        norm=pltnorm1, cmap=pltcmp1, zorder=1)
                                    axs[0].set_ylabel(r'Pressure [$hPa$]')
                                
                                axs[jcol].invert_yaxis()
                                axs[jcol].set_ylim(1000, ptop)
                                axs[jcol].yaxis.set_minor_locator(AutoMinorLocator(2))
                                
                                axs[jcol].set_xlim(-90, 90)
                                axs[jcol].set_xticks(np.arange(-60, 61, 60))
                                axs[jcol].xaxis.set_minor_locator(AutoMinorLocator(2))
                                axs[jcol].xaxis.set_major_formatter(LatitudeFormatter(degree_symbol='° '))
                                
                                axs[jcol].grid(True, which='both', lw=0.5, c='gray', alpha=0.5, ls='--')
                        else:
                            axs[irow, jcol].text(
                                0, 1.02,
                                f'({string.ascii_lowercase[irow*ncol+jcol]}) {plt_colnames[irow*ncol+jcol]}',
                                ha='left',va='bottom',
                                transform=axs[irow, jcol].transAxes)
                            if (irow==0) & (jcol==0):
                                plt_mesh1 = axs[0, 0].pcolormesh(
                                    plt_org[ds_names[0]].lat,
                                    plt_org[ds_names[0]].pressure,
                                    plt_org[ds_names[0]],
                                    norm=pltnorm1, cmap=pltcmp1, zorder=1)
                            
                            if jcol==0:
                                axs[irow, 0].set_ylabel(r'Pressure [$hPa$]')
                            
                            axs[irow, jcol].invert_yaxis()
                            axs[irow, jcol].set_ylim(1000, ptop)
                            axs[irow, jcol].yaxis.set_minor_locator(AutoMinorLocator(2))
                            
                            axs[irow, jcol].set_xlim(-90, 90)
                            axs[irow, jcol].set_xticks(np.arange(-60, 61, 60))
                            axs[irow, jcol].xaxis.set_minor_locator(AutoMinorLocator(2))
                            axs[irow, jcol].xaxis.set_major_formatter(LatitudeFormatter(degree_symbol='° '))
                            
                            axs[irow, jcol].grid(True, which='both', lw=0.5, c='gray', alpha=0.5, ls='--')
                
                if plt_mode in ['original']:
                    if nrow == 1:
                        for jcol in range(ncol-1):
                            plt_mesh1 = axs[jcol+1].pcolormesh(
                                plt_org[ds_names[jcol+1]].lat,
                                plt_org[ds_names[jcol+1]].pressure,
                                plt_org[ds_names[jcol+1]],
                                norm=pltnorm1, cmap=pltcmp1, zorder=1)
                    else:
                        for irow in range(nrow):
                            for jcol in range(ncol):
                                if ((irow != 0) | (jcol != 0)):
                                    plt_mesh1 = axs[irow, jcol].pcolormesh(
                                        plt_org[ds_names[irow*ncol+jcol]].lat,
                                        plt_org[ds_names[irow*ncol+jcol]].pressure,
                                        plt_org[ds_names[irow*ncol+jcol]],
                                        norm=pltnorm1, cmap=pltcmp1,zorder=1)
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
                            cax=fig.add_axes([0.26, fm_bottom*0.55, 0.48, fm_bottom / 6]))
                        cbar1.ax.set_xlabel(cbar_label1)
                elif plt_mode in ['difference']:
                    if nrow == 1:
                        for jcol in range(ncol-1):
                            plt_mesh2 = axs[jcol+1].pcolormesh(
                                plt_diff[ds_names[jcol+1]].lat,
                                plt_diff[ds_names[jcol+1]].pressure,
                                plt_diff[ds_names[jcol+1]],
                                norm=pltnorm2, cmap=pltcmp2,zorder=1)
                    else:
                        for irow in range(nrow):
                            for jcol in range(ncol):
                                if ((irow != 0) | (jcol != 0)):
                                    plt_mesh2 = axs[irow, jcol].pcolormesh(
                                        plt_diff[ds_names[irow*ncol+jcol]].lat,
                                        plt_diff[ds_names[irow*ncol+jcol]].pressure,
                                        plt_diff[ds_names[irow*ncol+jcol]],
                                        norm=pltnorm2, cmap=pltcmp2,zorder=1)
                    cbar1 = fig.colorbar(
                        plt_mesh1,#cm.ScalarMappable(norm=pltnorm1, cmap=pltcmp1),#
                        format=remove_trailing_zero_pos,
                        orientation="horizontal", ticks=pltticks1, extend=extend1,
                        cax=fig.add_axes([0.01, fm_bottom*0.55, 0.48, fm_bottom / 6]))
                    cbar1.ax.set_xlabel(cbar_label1)
                    cbar2 = fig.colorbar(
                        plt_mesh2,#cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2),#
                        format=remove_trailing_zero_pos,
                        orientation="horizontal", ticks=pltticks2, extend=extend2,
                        cax=fig.add_axes([0.51, fm_bottom*0.55, 0.48, fm_bottom / 6]))
                    cbar2.ax.set_xlabel(cbar_label2)
            
            opng = f'figures/4_um/4.2_access_am3/4.2.0_sim_obs/4.2.0.1 {ivar} {', '.join(ds_names)} {plt_region} {plt_mode} {years}-{yeare}.png'
            fig.subplots_adjust(left=fm_left, right=0.995, bottom=fm_bottom, top=fm_top)
            fig.savefig(opng, dpi=600)







'''
set(amvar2stash.keys()) - set([
    'pa', 'theta',
    'ua', 'va', 'hus', 'qcf', 'wa', 'qcl', 'qr', 'rsu', 'rsd', 'rsucs', 'rsdcs', 'rlu', 'rld', 'rlucs', 'rldcs', 'TCF', 'clslw', 'rain_evap', 'updraught_mf', 'downdraught_mf', 'deepc_mf', 'congestusc_mf', 'shallowc_mf', 'midc_mf', 'ta', 'zg', 'qc', 'qt', 'wap', 'hur', 'DMS',
    'cosp_cc_cf', 'cosp_c_ca', 'cosp_c_c', 'cosp_c_cfl', 'cosp_c_cfi', 'cosp_c_cfu',
    'cosp_isccp_ctp_tau',
    'meridional_hf', 'meridional_mf',
    'ncloud',
    'seaice', 'sst', 'rsn_trop', 'rsu_trop', 'rln_trop', 'rld_trop', 'cosp_isccp_Tb', 'cosp_isccp_Tbcs', 'GPP', 'PNPP', 'zg_freeze', 'p_freeze', 'p_trop', 't_trop', 'h_trop', 'ts', 'blh', 'rlds', 'ps', 'rsns', 'rsdt', 'rsutcs', 'rsdscs', 'rsuscs', 'rsds', 'rlns', 'rlutcs', 'rldscs', 'uas', 'vas', 'sfcWind', 'tas', 'das', 'psl', 'prw', 'rlut', 'rsut', 'hfss', 'hfls', 'pr', 'clh', 'clm', 'cll', 'clt', 'clwvi', 'clivi', 'rlu_t_s', 'rss_dir', 'rss_dif', 'cosp_isccp_albedo', 'cosp_isccp_tau', 'cosp_isccp_ctp', 'cosp_isccp_tcc', 'cosp_c_lcc', 'cosp_c_mcc', 'cosp_c_hcc', 'cosp_c_tcc', 'mlh', 'mlentrain', 'blentrain', 'wind_gust', 'lsrf', 'lssf', 'crf', 'csf', 'rain', 'snow', 'deep_pr', 'clvl', 'CAPE', 'CIN', 'dmvi', 'wmvi', 'fog2m', 'qt2m', 'hfms', 'huss', 'hurs',
    ])
'''
# endregion


# region plot obs and sim am zm hl

# options
years = '1983'; yeare = '1987'
vars = [
    'cosp_cc_cf', 'cosp_c_ca', 'cosp_c_c', 'cosp_c_cfl', 'cosp_c_cfi', 'cosp_c_cfu',
    ]
ds_names = ['access-am3-configs', 'am3-plus4k', 'am3-climaerosol', 'am3-climaerop4k'] #
plt_regions = ['global']
plt_modes = ['original', 'difference']
nrow = 1 # 2 #
ncol = len(ds_names) # 3 #
if ncol<=2:
    mpl.rc('font', family='Times New Roman', size=10)
elif ncol==3:
    mpl.rc('font', family='Times New Roman', size=12)
elif ncol>=4:
    mpl.rc('font', family='Times New Roman', size=14)

# settings
extend2 = 'both'

for ivar in vars:
    # ivar = 'cosp_c_ca'
    print(f'#-------------------------------- {ivar}')
    
    ds_data = {'ann': {}, 'am': {}}
    for ids in ds_names:
        print(f'Get {ids}')
        
        if ids in ['access-am3-configs', 'am3-plus4k', 'am3-climaerosol', 'am3-climaerop4k']:
            # ids = 'am3-plus4k'
            istream = next((k for k, v in amvargroups.items() if ivar in v), None)
            if not istream is None:
                fl = sorted(glob.glob(f'cylc-run/{ids}/share/data/History_Data/netCDF/*a.p{istream}*.nc'))
                ds = xr.open_mfdataset(fl, preprocess=preprocess_amoutput, parallel=True).sel(time=slice(years, yeare))
                
                ds_data['ann'][ids] = ds[ivar].resample({'time': '1YE'}).map(time_weighted_mean).mean(dim='lon').compute()
                # print(np.nanmax(ds_data['ann'][ids]))
                # print(np.nanmin(ds_data['ann'][ids]))
            
            if ivar in ['cosp_cc_cf', 'cosp_c_ca', 'cosp_c_c', 'cosp_c_cfl', 'cosp_c_cfi', 'cosp_c_cfu']:
                ds_data['ann'][ids] *= 100
            del ds
        
        ds_data['ann'][ids] = ds_data['ann'][ids].sortby(['lat']).transpose(..., 'height', 'lat')
        ds_data['am'][ids] = ds_data['ann'][ids].mean(dim='time').compute()
    
    cbar_label1 = f'{years}-{yeare} {era5_varlabels[cmip6_era5_var[ivar]]}'
    cbar_label2 = f'Difference in {era5_varlabels[cmip6_era5_var[ivar]]}'
    
    for plt_region in plt_regions:
        # plt_region = 'global'
        print(f'#---------------- {plt_region}')
        
        plt_org = {}
        plt_ann = {}
        for ids in ds_names:
            if plt_region == 'global':
                plt_org[ids] = ds_data['am'][ids]
                plt_ann[ids] = ds_data['ann'][ids]
        
        plt_diff = {}
        for ids in ds_names[1:]:
            # ids = ds_names[1]
            print(f'{ids} - {ds_names[0]}')
            plt_diff[ids] = plt_org[ids] - plt_org[ds_names[0]].interp(lat=plt_org[ids].lat, height=plt_org[ids].height)
            
            if ivar not in ['pr']:
                ttest_fdr_res = ttest_fdr_control(
                    plt_ann[ids],
                    plt_ann[ds_names[0]].interp(lat=plt_ann[ids].lat, height=plt_ann[ids].height))
                plt_diff[ids] = plt_diff[ids].where(ttest_fdr_res, np.nan)
        
        extend1 = 'both'
        if plt_region == 'global':
            if ivar in ['cosp_cc_cf', 'cosp_c_ca', 'cosp_c_c', 'cosp_c_cfl', 'cosp_c_cfi', 'cosp_c_cfu']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=0, cm_max=40, cm_interval1=5, cm_interval2=5, cmap='Blues_r')
                extend1 = 'max'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-15, cm_max=15, cm_interval1=3, cm_interval2=3, cmap='BrBG_r')
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
                fig, axs = plt.subplots(
                    nrow, ncol,
                    figsize=np.array([6.6*ncol+4, 5.5*nrow+4])/2.54,
                    sharey=True, sharex=True,
                    gridspec_kw={'hspace': 0.01, 'wspace': 0.05},)
                fm_bottom = 3 / (5.5*nrow+4)
                fm_top = 1 - 1/(5.5*nrow+4)
                fm_left = 2 / (6.6*ncol+4)
                fm_right = 1 - 2 / (6.6*ncol+4)
                
                plt_colnames = [f'{am3_label[ds_names[0]]}']
                if plt_mode in ['original']:
                    plt_colnames += [f'{am3_label[ids]}' for ids in ds_names[1:]]
                elif plt_mode in ['difference']:
                    plt_colnames += [f'{am3_label[ids]} - {am3_label[ds_names[0]]}' for ids in ds_names[1:]]
                
                for irow in range(nrow):
                    for jcol in range(ncol):
                        if nrow == 1:
                            if ncol == 1:
                                axs.text(
                                    0, -0.02, plt_colnames[jcol],
                                    ha='left', va='top',
                                    transform=axs.transAxes)
                                plt_mesh1 = axs.pcolormesh(
                                    plt_org[ds_names[0]].lat,
                                    plt_org[ds_names[0]].height/1000,
                                    plt_org[ds_names[0]],
                                    norm=pltnorm1, cmap=pltcmp1, zorder=1)
                                
                                axs.set_ylim(0, 19)
                                axs.yaxis.set_minor_locator(AutoMinorLocator(2))
                                
                                ax2 = axs.twinx()
                                ax2.set_ylim(0, 19)
                                ax2.set_yticks(pressure_to_height_std(np.array([1000,  800,  600,  400,  200, 100]) * units.hPa).magnitude)
                                ax2.set_yticklabels(np.array([1000,  800,  600,  400,  200, 100]), c = 'gray')
                                ax2.set_ylabel('Pressure [$hPa$]', c = 'gray')
                                
                                axs.set_xlim(-90, 90)
                                axs.set_xticks(np.arange(-60, 61, 60))
                                axs.xaxis.set_minor_locator(AutoMinorLocator(2))
                                axs.xaxis.set_major_formatter(LatitudeFormatter(degree_symbol='° '))
                                
                                axs.grid(True, which='both', lw=0.5, c='gray', alpha=0.5, ls='--')
                                axs.set_ylabel(r'Height [$km$]')
                            else:
                                axs[jcol].text(
                                    0, 1.02,
                                    f'({string.ascii_lowercase[jcol]}) {plt_colnames[jcol]}',
                                    ha='left',va='bottom',
                                    transform=axs[jcol].transAxes)
                                if jcol==0:
                                    plt_mesh1 = axs[0].pcolormesh(
                                        plt_org[ds_names[0]].lat,
                                        plt_org[ds_names[0]].height/1000,
                                        plt_org[ds_names[0]],
                                        norm=pltnorm1, cmap=pltcmp1, zorder=1)
                                    axs[0].set_ylabel(r'Height [$km$]')
                                    
                                    axs[0].set_ylim(0, 19)
                                    axs[0].yaxis.set_minor_locator(AutoMinorLocator(2))
                                    
                                if jcol==(ncol-1):
                                    ax2 = axs[jcol].twinx()
                                    ax2.set_ylim(0, 19)
                                    ax2.set_yticks(pressure_to_height_std(np.array([1000,  800,  600,  400,  200, 100]) * units.hPa).magnitude)
                                    ax2.set_yticklabels(np.array([1000,  800,  600,  400,  200, 100]), c = 'gray')
                                    ax2.set_ylabel('Pressure [$hPa$]', c = 'gray')
                                
                                axs[jcol].set_xlim(-90, 90)
                                axs[jcol].set_xticks(np.arange(-60, 61, 60))
                                axs[jcol].xaxis.set_minor_locator(AutoMinorLocator(2))
                                axs[jcol].xaxis.set_major_formatter(LatitudeFormatter(degree_symbol='° '))
                                
                                axs[jcol].grid(True, which='both', lw=0.5, c='gray', alpha=0.5, ls='--')
                        else:
                            axs[irow, jcol].text(
                                0, 1.02,
                                f'({string.ascii_lowercase[irow*ncol+jcol]}) {plt_colnames[irow*ncol+jcol]}',
                                ha='left',va='bottom',
                                transform=axs[irow, jcol].transAxes)
                            if (irow==0) & (jcol==0):
                                plt_mesh1 = axs[0, 0].pcolormesh(
                                    plt_org[ds_names[0]].lat,
                                    plt_org[ds_names[0]].height/1000,
                                    plt_org[ds_names[0]],
                                    norm=pltnorm1, cmap=pltcmp1, zorder=1)
                            
                            if jcol==0:
                                axs[irow, 0].set_ylabel(r'Height [$km$]')
                                
                                axs[irow, 0].set_ylim(0, 19)
                                axs[irow, 0].yaxis.set_minor_locator(AutoMinorLocator(2))
                                
                            if jcol==(ncol-1):
                                ax2 = axs[irow, jcol].twinx()
                                ax2.set_ylim(0, 19)
                                ax2.set_yticks(pressure_to_height_std(np.array([1000,  800,  600,  400,  200, 100]) * units.hPa).magnitude)
                                ax2.set_yticklabels(np.array([1000,  800,  600,  400,  200, 100]), c = 'gray')
                                ax2.set_ylabel('Pressure [$hPa$]', c = 'gray')
                            
                            axs[irow, jcol].set_xlim(-90, 90)
                            axs[irow, jcol].set_xticks(np.arange(-60, 61, 60))
                            axs[irow, jcol].xaxis.set_minor_locator(AutoMinorLocator(2))
                            axs[irow, jcol].xaxis.set_major_formatter(LatitudeFormatter(degree_symbol='° '))
                            
                            axs[irow, jcol].grid(True, which='both', lw=0.5, c='gray', alpha=0.5, ls='--')
                
                if plt_mode in ['original']:
                    if nrow == 1:
                        for jcol in range(ncol-1):
                            plt_mesh1 = axs[jcol+1].pcolormesh(
                                plt_org[ds_names[jcol+1]].lat,
                                plt_org[ds_names[jcol+1]].height/1000,
                                plt_org[ds_names[jcol+1]],
                                norm=pltnorm1, cmap=pltcmp1, zorder=1)
                    else:
                        for irow in range(nrow):
                            for jcol in range(ncol):
                                if ((irow != 0) | (jcol != 0)):
                                    plt_mesh1 = axs[irow, jcol].pcolormesh(
                                        plt_org[ds_names[irow*ncol+jcol]].lat,
                                        plt_org[ds_names[irow*ncol+jcol]].height/1000,
                                        plt_org[ds_names[irow*ncol+jcol]],
                                        norm=pltnorm1, cmap=pltcmp1,zorder=1)
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
                            cax=fig.add_axes([0.26, fm_bottom*0.55, 0.48, fm_bottom / 6]))
                        cbar1.ax.set_xlabel(cbar_label1)
                elif plt_mode in ['difference']:
                    if nrow == 1:
                        for jcol in range(ncol-1):
                            plt_mesh2 = axs[jcol+1].pcolormesh(
                                plt_diff[ds_names[jcol+1]].lat,
                                plt_diff[ds_names[jcol+1]].height/1000,
                                plt_diff[ds_names[jcol+1]],
                                norm=pltnorm2, cmap=pltcmp2,zorder=1)
                    else:
                        for irow in range(nrow):
                            for jcol in range(ncol):
                                if ((irow != 0) | (jcol != 0)):
                                    plt_mesh2 = axs[irow, jcol].pcolormesh(
                                        plt_diff[ds_names[irow*ncol+jcol]].lat,
                                        plt_diff[ds_names[irow*ncol+jcol]].height/1000,
                                        plt_diff[ds_names[irow*ncol+jcol]],
                                        norm=pltnorm2, cmap=pltcmp2,zorder=1)
                    cbar1 = fig.colorbar(
                        plt_mesh1,#cm.ScalarMappable(norm=pltnorm1, cmap=pltcmp1),#
                        format=remove_trailing_zero_pos,
                        orientation="horizontal", ticks=pltticks1, extend=extend1,
                        cax=fig.add_axes([0.01, fm_bottom*0.55, 0.48, fm_bottom / 6]))
                    cbar1.ax.set_xlabel(cbar_label1)
                    cbar2 = fig.colorbar(
                        plt_mesh2,#cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2),#
                        format=remove_trailing_zero_pos,
                        orientation="horizontal", ticks=pltticks2, extend=extend2,
                        cax=fig.add_axes([0.51, fm_bottom*0.55, 0.48, fm_bottom / 6]))
                    cbar2.ax.set_xlabel(cbar_label2)
            
            opng = f'figures/4_um/4.2_access_am3/4.2.0_sim_obs/4.2.0.2 {ivar} {', '.join(ds_names)} {plt_region} {plt_mode} {years}-{yeare}.png'
            fig.subplots_adjust(left=fm_left, right=fm_right, bottom=fm_bottom, top=fm_top)
            fig.savefig(opng, dpi=600)




'''

'''
# endregion


# region plot obs and sim am cosp_isccp_ctp_tau

# options
years = '1983'; yeare = '1987'
vars = [
    'cosp_isccp_ctp_tau',
    ]
ds_names = ['access-am3-configs', 'am3-plus4k', 'am3-climaerosol', 'am3-climaerop4k'] #
plt_regions = ['global']
plt_modes = ['original', 'difference']
nrow = 1 # 2 #
ncol = len(ds_names) # 3 #
if ncol<=2:
    mpl.rc('font', family='Times New Roman', size=10)
elif ncol==3:
    mpl.rc('font', family='Times New Roman', size=12)
elif ncol>=4:
    mpl.rc('font', family='Times New Roman', size=14)

# settings
extend2 = 'both'

for ivar in vars:
    # ivar = 'cosp_isccp_ctp_tau'
    print(f'#-------------------------------- {ivar}')
    
    ds_data = {'ann': {}, 'am': {}}
    for ids in ds_names:
        print(f'Get {ids}')
        
        if ids in ['access-am3-configs', 'am3-plus4k', 'am3-climaerosol', 'am3-climaerop4k']:
            # ids = 'am3-plus4k'
            istream = next((k for k, v in amvargroups.items() if ivar in v), None)
            if not istream is None:
                fl = sorted(glob.glob(f'cylc-run/{ids}/share/data/History_Data/netCDF/*a.p{istream}*.nc'))
                ds = xr.open_mfdataset(fl, preprocess=preprocess_amoutput, parallel=True).sel(time=slice(years, yeare))
                
                ds_data['ann'][ids] = ds[ivar].rename({'pseudo_level': 'tau'}).resample({'time': '1YE'}).map(time_weighted_mean).mean(dim='lon').weighted(np.cos(np.deg2rad(ds.lat))).mean(dim='lat').compute()
                
                # print(np.nanmax(ds_data['ann'][ids]))
                # print(np.nanmin(ds_data['ann'][ids]))
            
            if ivar in ['cosp_isccp_ctp_tau']:
                ds_data['ann'][ids] *= 100
            del ds
        
        ds_data['ann'][ids] = ds_data['ann'][ids].transpose(..., 'pressure', 'tau')
        ds_data['am'][ids] = ds_data['ann'][ids].mean(dim='time').compute()
    
    cbar_label1 = f'{years}-{yeare} {era5_varlabels[cmip6_era5_var[ivar]]}'
    cbar_label2 = f'Difference in {era5_varlabels[cmip6_era5_var[ivar]]}'
    
    for plt_region in plt_regions:
        # plt_region = 'global'
        print(f'#---------------- {plt_region}')
        
        plt_org = {}
        plt_ann = {}
        for ids in ds_names:
            if plt_region == 'global':
                plt_org[ids] = ds_data['am'][ids]
                plt_ann[ids] = ds_data['ann'][ids]
        
        plt_diff = {}
        for ids in ds_names[1:]:
            # ids = ds_names[1]
            print(f'{ids} - {ds_names[0]}')
            plt_diff[ids] = plt_org[ids] - plt_org[ds_names[0]].interp(pressure=plt_org[ids].pressure, tau=plt_org[ids].tau)
            
            if ivar not in ['pr']:
                ttest_fdr_res = ttest_fdr_control(
                    plt_ann[ids],
                    plt_ann[ds_names[0]].interp(pressure=plt_ann[ids].pressure, tau=plt_ann[ids].tau))
                plt_diff[ids] = plt_diff[ids].where(ttest_fdr_res, np.nan)
        
        extend1 = 'both'
        if plt_region == 'global':
            if ivar in ['cosp_isccp_ctp_tau']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=0, cm_max=4, cm_interval1=0.25, cm_interval2=0.5, cmap='Blues_r')
                extend1 = 'max'
                pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                    cm_min=-0.5, cm_max=0.5, cm_interval1=0.05, cm_interval2=0.1, cmap='BrBG_r')
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
                fig, axs = plt.subplots(
                    nrow, ncol,
                    figsize=np.array([6.6*ncol+2.5, 6*nrow+4.5])/2.54,
                    sharey=True, sharex=True,
                    gridspec_kw={'hspace': 0.01, 'wspace': 0.12},)
                fm_bottom = 3.6 / (6*nrow+4.5)
                fm_top = 1 - 0.9/(6*nrow+4.5)
                fm_left = 2 / (6.6*ncol+2.5)
                fm_right = 1 - 0.5 / (6.6*ncol+2.5)
                
                plt_colnames = [f'{am3_label[ds_names[0]]}']
                if plt_mode in ['original']:
                    plt_colnames += [f'{am3_label[ids]}' for ids in ds_names[1:]]
                elif plt_mode in ['difference']:
                    plt_colnames += [f'{am3_label[ids]} - {am3_label[ds_names[0]]}' for ids in ds_names[1:]]
                
                for irow in range(nrow):
                    for jcol in range(ncol):
                        if nrow == 1:
                            if ncol == 1:
                                axs.text(
                                    0, -0.02, plt_colnames[jcol],
                                    ha='left', va='top',
                                    transform=axs.transAxes)
                                plt_mesh1 = axs.pcolormesh(
                                    np.arange(0.5, 7, 1),
                                    np.arange(0.5, 7, 1),
                                    plt_org[ds_names[0]],
                                    norm=pltnorm1, cmap=pltcmp1, zorder=1)
                                
                                axs.set_xlim(0, 7)
                                axs.set_xticks(np.arange(0, 8, 1))
                                axs.set_xlabel(r'$\tau$ [$-$]')
                                axs.set_xticklabels(['0', '0.3', '1.3', '3.6', '9.4', '23', '60', '380'])
                                
                                axs.set_ylim(0, 7)
                                axs.set_yticks(np.arange(0, 8, 1))
                                axs.set_ylabel(r'CTP [$hPa$]')
                                axs.set_yticklabels(['1000', '800', '680', '560', '440', '310', '180', '50'])
                                
                                axs.grid(True, which='both', lw=0.5, c='gray', alpha=0.5, ls='--')
                                
                            else:
                                axs[jcol].text(
                                    0, 1.02,
                                    f'({string.ascii_lowercase[jcol]}) {plt_colnames[jcol]}',
                                    ha='left',va='bottom',
                                    transform=axs[jcol].transAxes)
                                if jcol==0:
                                    plt_mesh1 = axs[0].pcolormesh(
                                        np.arange(0.5, 7, 1),
                                        np.arange(0.5, 7, 1),
                                        plt_org[ds_names[0]],
                                        norm=pltnorm1, cmap=pltcmp1, zorder=1)
                                    
                                    axs[0].set_ylim(0, 7)
                                    axs[0].set_yticks(np.arange(0, 8, 1))
                                    axs[0].set_ylabel(r'CTP [$hPa$]')
                                    axs[0].set_yticklabels(['1000', '800', '680', '560', '440', '310', '180', '50'])
                                
                                axs[jcol].set_xlim(0, 7)
                                axs[jcol].set_xticks(np.arange(0, 8, 1))
                                axs[jcol].set_xlabel(r'$\tau$ [$-$]')
                                axs[jcol].set_xticklabels(['0', '0.3', '1.3', '3.6', '9.4', '23', '60', '380'])
                                
                                axs[jcol].grid(True, which='both', lw=0.5, c='gray', alpha=0.5, ls='--')
                                
                        else:
                            axs[irow, jcol].text(
                                0, 1.02,
                                f'({string.ascii_lowercase[irow*ncol+jcol]}) {plt_colnames[irow*ncol+jcol]}',
                                ha='left',va='bottom',
                                transform=axs[irow, jcol].transAxes)
                            if (irow==0) & (jcol==0):
                                plt_mesh1 = axs[0, 0].pcolormesh(
                                    np.arange(0.5, 7, 1),
                                    np.arange(0.5, 7, 1),
                                    plt_org[ds_names[0]],
                                    norm=pltnorm1, cmap=pltcmp1, zorder=1)
                            
                            if jcol==0:
                                axs[irow, 0].set_ylim(0, 7)
                                axs[irow, 0].set_yticks(np.arange(0, 8, 1))
                                axs[irow, 0].set_ylabel(r'CTP [$hPa$]')
                                axs[irow, 0].set_yticklabels(['1000', '800', '680', '560', '440', '310', '180', '50'])
                            
                            axs[irow, jcol].set_xlim(0, 7)
                            axs[irow, jcol].set_xticks(np.arange(0, 8, 1))
                            axs[irow, jcol].set_xlabel(r'$\tau$ [$-$]')
                            axs[irow, jcol].set_xticklabels(['0', '0.3', '1.3', '3.6', '9.4', '23', '60', '380'])
                            
                            axs[irow, jcol].grid(True, which='both', lw=0.5, c='gray', alpha=0.5, ls='--')
                
                if plt_mode in ['original']:
                    if nrow == 1:
                        for jcol in range(ncol-1):
                            plt_mesh1 = axs[jcol+1].pcolormesh(
                                np.arange(0.5, 7, 1),
                                np.arange(0.5, 7, 1),
                                plt_org[ds_names[jcol+1]],
                                norm=pltnorm1, cmap=pltcmp1, zorder=1)
                    else:
                        for irow in range(nrow):
                            for jcol in range(ncol):
                                if ((irow != 0) | (jcol != 0)):
                                    plt_mesh1 = axs[irow, jcol].pcolormesh(
                                        np.arange(0.5, 7, 1),
                                        np.arange(0.5, 7, 1),
                                        plt_org[ds_names[irow*ncol+jcol]],
                                        norm=pltnorm1, cmap=pltcmp1,zorder=1)
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
                            cax=fig.add_axes([0.26, fm_bottom*0.44, 0.48, fm_bottom / 6]))
                        cbar1.ax.set_xlabel(cbar_label1)
                elif plt_mode in ['difference']:
                    if nrow == 1:
                        for jcol in range(ncol-1):
                            plt_mesh2 = axs[jcol+1].pcolormesh(
                                np.arange(0.5, 7, 1),
                                np.arange(0.5, 7, 1),
                                plt_diff[ds_names[jcol+1]],
                                norm=pltnorm2, cmap=pltcmp2,zorder=1)
                    else:
                        for irow in range(nrow):
                            for jcol in range(ncol):
                                if ((irow != 0) | (jcol != 0)):
                                    plt_mesh2 = axs[irow, jcol].pcolormesh(
                                        np.arange(0.5, 7, 1),
                                        np.arange(0.5, 7, 1),
                                        plt_diff[ds_names[irow*ncol+jcol]],
                                        norm=pltnorm2, cmap=pltcmp2,zorder=1)
                    cbar1 = fig.colorbar(
                        plt_mesh1,#cm.ScalarMappable(norm=pltnorm1, cmap=pltcmp1),#
                        format=remove_trailing_zero_pos,
                        orientation="horizontal", ticks=pltticks1, extend=extend1,
                        cax=fig.add_axes([0.01, fm_bottom*0.44, 0.48, fm_bottom / 6]))
                    cbar1.ax.set_xlabel(cbar_label1)
                    cbar2 = fig.colorbar(
                        plt_mesh2,#cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2),#
                        format=remove_trailing_zero_pos,
                        orientation="horizontal", ticks=pltticks2, extend=extend2,
                        cax=fig.add_axes([0.51, fm_bottom*0.44, 0.48, fm_bottom / 6]))
                    cbar2.ax.set_xlabel(cbar_label2)
            
            opng = f'figures/4_um/4.2_access_am3/4.2.0_sim_obs/4.2.0.3 {ivar} {', '.join(ds_names)} {plt_region} {plt_mode} {years}-{yeare}.png'
            fig.subplots_adjust(left=fm_left, right=fm_right, bottom=fm_bottom, top=fm_top)
            fig.savefig(opng, dpi=600)




'''
for ids in ds_names:
    print(f'{ids}: {np.round(plt_org[ids].sum().values, 1)}')
'''
# endregion

