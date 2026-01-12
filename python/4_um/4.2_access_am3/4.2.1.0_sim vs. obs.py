

# qsub -I -q copyq -P v46 -l walltime=6:00:00,ncpus=1,mem=48GB,jobfs=10GB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22+gdata/py18+gdata/gx60+gdata/xp65+gdata/qx55+gdata/rv74+gdata/al33+gdata/rr3+gdata/hr22+scratch/gx60+scratch/gb02+gdata/gb02


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


# region plot obs and sim am

# options
years = '1983'; yeare = '1987'
vars = ['sst', ] # 'rsut', 'rlut', 'pr'
ds_names = ['access-am3-configs', 'am3-plus4k']
plt_regions = ['global']
plt_modes = ['original', 'difference']
nrow = 1 # 2 #
ncol = len(ds_names) # 3 #

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

for ivar in vars:
    # ivar = 'sst'
    print(f'#-------------------------------- {ivar}')
    
    if ivar in ['sst']:
        iregion = 'ocean'
    else:
        iregion = 'global'
    
    ds_data = {'ann': {}, 'am': {}}
    for ids in ds_names:
        print(f'Get {ids}')
        
        if ids in ['access-am3-configs', 'am3-plus4k']:
            # ids = 'am3-plus4k'
            istream = next((k for k, v in amvargroups.items() if ivar in v), None)
            if not istream is None:
                fl = sorted(glob.glob(f'scratch/cylc-run/{ids}/share/data/History_Data/netCDF/*a.p{istream}*.nc'))
                ds = xr.open_mfdataset(fl, preprocess=preprocess_amoutput, parallel=True)
                ds = ds[ivar].sel(time=slice(years, yeare))
                
                if ivar in ['sst']:
                    ds -= zerok
                
                ds_data['ann'][ids] = ds.resample({'time': '1YE'}).map(time_weighted_mean).compute()
        
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
            elif ivar in ['CDNC']:
                pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
                    cm_min=0, cm_max=150, cm_interval1=10, cm_interval2=20,
                    cmap='viridis_r',)
                extend1 = 'max'
        
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
            print(f'get diff: {ids} - {ds_names[0]}')
            plt_diff[ids] = regrid(plt_org[ids], plt_org[ds_names[0]]) - plt_org[ds_names[0]]
            plt_rmsd[ids] = global_land_ocean_rmsd(plt_diff[ids], iregion)
            plt_md[ids] = global_land_ocean_mean(plt_diff[ids], iregion)
            
            if ivar not in ['pr']:
                ttest_fdr_res = ttest_fdr_control(
                    # xe.Regridder(plt_ann[ids], plt_ann[ds_names[0]], 'bilinear')(plt_ann[ids]),
                    regrid(plt_ann[ids], plt_ann[ds_names[0]]),
                    plt_ann[ds_names[0]])
                plt_diff[ids] = plt_diff[ids].where(ttest_fdr_res, np.nan)
        
        for plt_mode in plt_modes:
            print(f'#-------- {plt_mode}')
            
            if plt_region == 'global':
                # plt_region = 'global'
                fig, axs = plt.subplots(
                    nrow, ncol,
                    figsize=np.array([8.8*ncol, 4.4*nrow+2.4]) / 2.54,
                    subplot_kw={'projection': ccrs.Mollweide(central_longitude=180)},
                    gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)
                fm_bottom = 1.8 / (4.4*nrow+2.4)
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
            
            if ivar in ['pr']:
                digit = 2
            else:
                digit = 1
            
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
                        0, 0, plt_text[jcol], ha='left', va='top',
                        transform=axs.transAxes, size=8)
                else:
                    for jcol in range(ncol):
                        axs[jcol].text(
                            0, 1,
                            f'({string.ascii_lowercase[jcol]}) {plt_colnames[jcol]}',
                            ha='left',va='bottom',transform=axs[jcol].transAxes)
                        axs[jcol].text(
                            0, 0, plt_text[jcol], ha='left', va='top',
                            transform=axs[jcol].transAxes,)
            else:
                for irow in range(nrow):
                    for jcol in range(ncol):
                        axs[irow, jcol].text(
                            0, 1,
                            f'({string.ascii_lowercase[irow * ncol + jcol]}) {plt_colnames[irow * ncol + jcol]}',
                            ha='left',va='bottom',
                            transform=axs[irow, jcol].transAxes)
                        axs[irow, jcol].text(
                            0, 0, plt_text[irow * ncol + jcol],
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
                        cax=fig.add_axes([0.26, fm_bottom*0.7, 0.48, fm_bottom / 6]))
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
                    cax=fig.add_axes([0.01, fm_bottom*0.7, 0.48, fm_bottom / 6]))
                cbar1.ax.set_xlabel(cbar_label1)
                cbar2 = fig.colorbar(
                    plt_mesh2,#cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2),#
                    format=remove_trailing_zero_pos,
                    orientation="horizontal", ticks=pltticks2, extend=extend2,
                    cax=fig.add_axes([0.51, fm_bottom*0.7, 0.48, fm_bottom / 6]))
                cbar2.ax.set_xlabel(cbar_label2)
            
            opng = f'figures/4_um/4.2_access_am3/4.2.0_sim_obs/4.2.0.0 {ivar} {', '.join(ds_names)} {plt_region} {plt_mode} {years}-{yeare}.png'
            fig.subplots_adjust(left=0.005, right=0.995, bottom=fm_bottom, top=0.96)
            fig.savefig(opng, dpi=600)






# endregion

