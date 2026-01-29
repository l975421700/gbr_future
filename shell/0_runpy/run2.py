

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


# region plot obs and sim am zm pl

# options
years = '1983'; yeare = '1987'
vars = [
    'TCF', 'qcf', 'qcl', 'qr', 'hus', 'hur', 'ta', 'ua', 'va', 'wap', 'zg'
    
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


