

# qsub -I -q copyq -l walltime=10:00:00,ncpus=1,mem=192GB,storage=gdata/v46+gdata/rt52+gdata/ob53+gdata/zv2


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
from xmip.preprocessing import rename_cmip6, broadcast_lonlat, correct_lon, promote_empty_dims, replace_x_y_nominal_lat_lon, correct_units, correct_coordinates, parse_lon_lat_bounds, maybe_convert_bounds_to_vertex, maybe_convert_vertex_to_bounds, combined_preprocessing

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
import re
import glob

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
    cdo_regrid,
    time_weighted_mean)

from statistics0 import (
    ttest_fdr_control,)

# endregion


# region plot AGCD, ERA5, BARRA-R2, BARRA-C2, historical, amip, am sm

plt_colnames = ['Annual mean', 'DJF', 'MAM', 'JJA', 'SON']
plt_rownames = ['AGCD', 'ERA5 - AGCD', 'BARRA-R2 - AGCD', 'BARRA-C2 - AGCD', r'$historical$ - AGCD', r'$amip$ - AGCD']
min_lon, max_lon, min_lat, max_lat = [110.58, 157.34, -43.69, -7.01]

for var2 in ['pr']:
    # var2='pr'
    var1 = cmip6_era5_var[var2]
    print(f'#-------------------------------- {var1} and {var2}')
    
    with open(f'data/obs/agcd/agcd_alltime_{var2}.pkl','rb') as f:
        agcd_alltime = pickle.load(f)
    with open(f'data/obs/era5/mon/era5_sl_mon_alltime_{var1}.pkl', 'rb') as f:
        era5_sl_mon_alltime = pickle.load(f)
    with open(f'data/sim/um/barra_r2/barra_r2_mon_alltime_{var2}.pkl','rb') as f:
        barra_r2_mon_alltime = pickle.load(f)
    with open(f'data/sim/um/barra_c2/barra_c2_mon_alltime_{var2}.pkl','rb') as f:
        barra_c2_mon_alltime = pickle.load(f)
    with open(f'/home/563/qg8515/scratch/data/sim/cmip6/historical_Amon_{var2}_regridded_alltime_ens.pkl', 'rb') as f:
        historical_regridded_alltime_ens = pickle.load(f)
    with open(f'/home/563/qg8515/scratch/data/sim/cmip6/amip_Amon_{var2}_regridded_alltime_ens.pkl', 'rb') as f:
        amip_regridded_alltime_ens = pickle.load(f)
    
    plt_data = {}
    for irow in plt_rownames: plt_data[irow] = {}
    
    plt_data['AGCD']['Annual mean'] = agcd_alltime['ann'].sel(time=slice('2001', '2014')).mean(dim='time')
    agcd_ann = regrid(agcd_alltime['ann'].sel(time=slice('2001', '2014')), ds_out=era5_sl_mon_alltime['am'].pipe(regrid).pipe(replace_x_y_nominal_lat_lon).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon)))
    agcd_am = agcd_ann.mean(dim='time').compute()
    
    plt_data['ERA5 - AGCD']['Annual mean'] = (regrid(era5_sl_mon_alltime['ann'].sel(time=slice('2001', '2014')).mean(dim='time'), ds_out=agcd_am) - agcd_am).compute()
    # ttest_fdr_res = ttest_fdr_control(
    #     agcd_ann,
    #     regrid(era5_sl_mon_alltime['ann'].sel(time=slice('2001', '2014')), ds_out=agcd_am))
    # plt_data['ERA5 - AGCD']['Annual mean'] = plt_data['ERA5 - AGCD']['Annual mean'].where(ttest_fdr_res, np.nan)
    
    plt_data['BARRA-R2 - AGCD']['Annual mean'] = (regrid(barra_r2_mon_alltime['ann'].sel(time=slice('2001', '2014')).mean(dim='time'), ds_out=agcd_am) - agcd_am).compute()
    # ttest_fdr_res = ttest_fdr_control(
    #     agcd_ann,
    #     regrid(barra_r2_mon_alltime['ann'].sel(time=slice('2001', '2014')), ds_out=agcd_am)
    #     )
    # plt_data['BARRA-R2 - AGCD']['Annual mean'] = plt_data['BARRA-R2 - AGCD']['Annual mean'].where(ttest_fdr_res, np.nan)
    
    plt_data['BARRA-C2 - AGCD']['Annual mean'] = (regrid(barra_c2_mon_alltime['ann'].sel(time=slice('2001', '2014')).mean(dim='time'), ds_out=agcd_am) - agcd_am).compute()
    # ttest_fdr_res = ttest_fdr_control(
    #     agcd_ann,
    #     regrid(barra_c2_mon_alltime['ann'].sel(time=slice('2001', '2014')), ds_out=agcd_am)
    #     )
    # plt_data['BARRA-C2 - AGCD']['Annual mean'] = plt_data['BARRA-C2 - AGCD']['Annual mean'].where(ttest_fdr_res, np.nan)
    
    plt_data[r'$historical$ - AGCD']['Annual mean'] = (historical_regridded_alltime_ens['ann'].sel(time=slice('2001', '2014')).mean(dim=['source_id', 'time']).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon)) - agcd_am).compute()
    # ttest_fdr_res = ttest_fdr_control(
    #     agcd_ann,
    #     historical_regridded_alltime_ens['ann'].sel(time=slice('2001', '2014')).mean(dim='source_id').sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon))
    #     )
    # plt_data[r'$historical$ - AGCD']['Annual mean'] = plt_data[r'$historical$ - AGCD']['Annual mean'].where(ttest_fdr_res, np.nan)
    
    plt_data[r'$amip$ - AGCD']['Annual mean'] = (amip_regridded_alltime_ens['ann'].sel(time=slice('2001', '2014')).mean(dim=['source_id', 'time']).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon)) - agcd_am).compute()
    # ttest_fdr_res = ttest_fdr_control(
    #     agcd_ann,
    #     amip_regridded_alltime_ens['ann'].sel(time=slice('2001', '2014')).mean(dim='source_id').sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon))
    #     )
    # plt_data[r'$amip$ - AGCD']['Annual mean'] = plt_data[r'$amip$ - AGCD']['Annual mean'].where(ttest_fdr_res, np.nan)
    
    for jcolnames in plt_colnames[1:]:
        # jcolnames='DJF'
        print(f'#---------------- {jcolnames}')
        
        plt_data['AGCD'][jcolnames] = agcd_alltime['sea'][agcd_alltime['sea'].time.dt.season==jcolnames].sel(time=slice('2001', '2014')).mean(dim='time')
        agcd_sea = regrid(agcd_alltime['sea'][agcd_alltime['sea'].time.dt.season==jcolnames].sel(time=slice('2001', '2014')), ds_out=era5_sl_mon_alltime['am'].pipe(regrid).pipe(replace_x_y_nominal_lat_lon).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon)))
        agcd_sm = agcd_sea.mean(dim='time').compute()
        
        plt_data['ERA5 - AGCD'][jcolnames] = (regrid(era5_sl_mon_alltime['sea'][era5_sl_mon_alltime['sea'].time.dt.season==jcolnames].sel(time=slice('2001', '2014')).mean(dim='time'), ds_out=agcd_sm) - agcd_sm).compute()
        # ttest_fdr_res = ttest_fdr_control(
        #     ceres_sea,
        #     era5_sl_mon_alltime['mon'].sel(time=slice('2001', '2014')).resample({'time': 'QE-FEB'}).map(time_weighted_mean)[1:-1][era5_sl_mon_alltime['mon'].sel(time=slice('2001', '2014')).resample({'time': 'QE-FEB'}).map(time_weighted_mean)[1:-1].time.dt.season==jcolnames].pipe(regrid).pipe(replace_x_y_nominal_lat_lon).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon))
        #     )
        # plt_data['ERA5 - CERES'][jcolnames] = plt_data['ERA5 - CERES'][jcolnames].where(ttest_fdr_res, np.nan)
        
        plt_data['BARRA-R2 - AGCD'][jcolnames] = (regrid(barra_r2_mon_alltime['sea'][barra_r2_mon_alltime['sea'].time.dt.season==jcolnames].sel(time=slice('2001', '2014')).mean(dim='time'), ds_out=agcd_sm) - agcd_sm).compute()
        # ttest_fdr_res = ttest_fdr_control(
        #     ceres_sea,
        #     regrid(barra_r2_mon_alltime['mon'].sel(time=slice('2001', '2014')).resample({'time': 'QE-FEB'}).map(time_weighted_mean)[1:-1][barra_r2_mon_alltime['mon'].sel(time=slice('2001', '2014')).resample({'time': 'QE-FEB'}).map(time_weighted_mean)[1:-1].time.dt.season==jcolnames], ds_out=plt_data['CERES'][jcolnames])
        #     )
        # plt_data['BARRA-R2 - CERES'][jcolnames] = plt_data['BARRA-R2 - CERES'][jcolnames].where(ttest_fdr_res, np.nan)
        
        plt_data['BARRA-C2 - AGCD'][jcolnames] = (regrid(barra_c2_mon_alltime['sea'][barra_c2_mon_alltime['sea'].time.dt.season==jcolnames].sel(time=slice('2001', '2014')).mean(dim='time'), ds_out=agcd_sm) - agcd_sm).compute()
        # ttest_fdr_res = ttest_fdr_control(
        #     ceres_sea,
        #     regrid(barra_c2_mon_alltime['mon'].sel(time=slice('2001', '2014')).resample({'time': 'QE-FEB'}).map(time_weighted_mean)[1:-1][barra_c2_mon_alltime['mon'].sel(time=slice('2001', '2014')).resample({'time': 'QE-FEB'}).map(time_weighted_mean)[1:-1].time.dt.season==jcolnames], ds_out=plt_data['CERES'][jcolnames])
        #     )
        # plt_data['BARRA-C2 - CERES'][jcolnames] = plt_data['BARRA-C2 - CERES'][jcolnames].where(ttest_fdr_res, np.nan)
        
        plt_data[r'$historical$ - AGCD'][jcolnames] = (historical_regridded_alltime_ens['sea'][:, historical_regridded_alltime_ens['sea'].time.dt.season==jcolnames].sel(time=slice('2001', '2014')).mean(dim=['source_id', 'time']).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon)) - agcd_sm).compute()
        # ttest_fdr_res = ttest_fdr_control(
        #     ceres_sea,
        #     historical_regridded_alltime_ens['mon'].sel(time=slice('2001', '2014')).mean(dim='source_id').resample({'time': 'QE-FEB'}).map(time_weighted_mean)[1:-1][historical_regridded_alltime_ens['mon'].sel(time=slice('2001', '2014')).mean(dim='source_id').resample({'time': 'QE-FEB'}).map(time_weighted_mean)[1:-1].time.dt.season==jcolnames].sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon))
        #     )
        # plt_data[r'$historical$ - CERES'][jcolnames] = plt_data[r'$historical$ - CERES'][jcolnames].where(ttest_fdr_res, np.nan)
        
        plt_data[r'$amip$ - AGCD'][jcolnames] = (amip_regridded_alltime_ens['sea'][:, amip_regridded_alltime_ens['sea'].time.dt.season==jcolnames].sel(time=slice('2001', '2014')).mean(dim=['source_id', 'time']).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon)) - agcd_sm).compute()
        # ttest_fdr_res = ttest_fdr_control(
        #     ceres_sea,
        #     amip_regridded_alltime_ens['mon'].sel(time=slice('2001', '2014')).mean(dim='source_id').resample({'time': 'QE-FEB'}).map(time_weighted_mean)[1:-1][amip_regridded_alltime_ens['mon'].sel(time=slice('2001', '2014')).mean(dim='source_id').resample({'time': 'QE-FEB'}).map(time_weighted_mean)[1:-1].time.dt.season==jcolnames].sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon))
        #     )
        # plt_data[r'$amip$ - CERES'][jcolnames] = plt_data[r'$amip$ - CERES'][jcolnames].where(ttest_fdr_res, np.nan)
    
    cbar_label1 = '2001-2014 ' + era5_varlabels[var1]
    cbar_label2 = 'Difference in 2001-2014 ' + era5_varlabels[var1]
    extend1 = 'max'
    extend2 = 'both'
    
    # print(stats.describe(np.concatenate([plt_data['AGCD'][colname].values for colname in plt_colnames]), axis=None, nan_policy='omit'))
    # print(stats.describe(np.concatenate([plt_data[rowname][colname].values for rowname in plt_rownames[1:] for colname in plt_colnames]), axis=None, nan_policy='omit'))
    
    if var1 in ['tp']:
        pltlevel1 = np.array([0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 6])
        pltticks1 = np.array([0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 6])
        pltnorm1 = BoundaryNorm(pltlevel1, ncolors=len(pltlevel1)-1, clip=True)
        pltcmp1 = plt.get_cmap('viridis_r', len(pltlevel1)-1)
        pltlevel2 = np.array([-3, -2, -1.5, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3])
        pltticks2 = np.array([-3, -2, -1.5, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3])
        pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
        pltcmp2 = plt.get_cmap('BrBG', len(pltlevel2)-1)
    
    nrow=len(plt_rownames)
    ncol=len(plt_colnames)
    fm_bottom=1.5/(4*nrow+1.5)
    
    fig, axs = plt.subplots(
        nrow, ncol, figsize=np.array([4.4*ncol, 4*nrow + 1.5]) / 2.54,
        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
        gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)
    
    for irow in range(nrow):
        axs[irow, 0].text(-0.05, 0.5, plt_rownames[irow], ha='right', va='center', rotation='vertical', transform=axs[irow, 0].transAxes)
        for jcol in range(ncol):
            axs[irow, jcol] = regional_plot(extent=[min_lon, max_lon, min_lat, max_lat], central_longitude=180, ax_org=axs[irow, jcol])
            axs[irow, jcol].add_feature(cfeature.OCEAN,color='white',zorder=2,edgecolor=None,lw=0)
            axs[irow, jcol].text(0, 1.02, f'({string.ascii_lowercase[irow]}{jcol+1})', ha='left', va='bottom', transform=axs[irow, jcol].transAxes,)
            if irow==0:
                axs[0, jcol].text(0.5, 1.02, plt_colnames[jcol], ha='center', va='bottom', transform=axs[0, jcol].transAxes)
    
    for jcol in range(ncol):
        plt_mesh1 = axs[0, jcol].pcolormesh(
            plt_data[plt_rownames[0]][plt_colnames[jcol]].lon,
            plt_data[plt_rownames[0]][plt_colnames[jcol]].lat,
            plt_data[plt_rownames[0]][plt_colnames[jcol]].values,
            norm=pltnorm1, cmap=pltcmp1, transform=ccrs.PlateCarree(),zorder=1)
    
    for irow in range(nrow-1):
        for jcol in range(ncol):
            plt_mesh2 = axs[irow+1, jcol].pcolormesh(
                plt_data[plt_rownames[irow+1]][plt_colnames[jcol]].lon,
                plt_data[plt_rownames[irow+1]][plt_colnames[jcol]].lat,
                plt_data[plt_rownames[irow+1]][plt_colnames[jcol]].values,
                norm=pltnorm2, cmap=pltcmp2,
                transform=ccrs.PlateCarree(),zorder=1)
    
    cbar1 = fig.colorbar(
        plt_mesh1, #cm.ScalarMappable(norm=pltnorm1, cmap=pltcmp1), #
        format=remove_trailing_zero_pos,
        orientation="horizontal", ticks=pltticks1, extend=extend1,
        cax=fig.add_axes([0.05, fm_bottom-0.01, 0.4, 0.015]))
    cbar1.ax.set_xlabel(cbar_label1)
    cbar2 = fig.colorbar(
        plt_mesh2, #cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), #
        format=remove_trailing_zero_pos,
        orientation="horizontal", ticks=pltticks2, extend=extend2,
        cax=fig.add_axes([0.55, fm_bottom-0.01, 0.4, 0.015]))
    cbar2.ax.set_xlabel(cbar_label2)
    
    fig.subplots_adjust(left=0.03, right=0.995, bottom=fm_bottom, top=0.98)
    fig.savefig(f'figures/4_um/4.0_barra/4.0.0_whole region/4.0.0.1 agcd vs. barra_c2, era5, and cmip6 am sm {var1}.png')
    
    del era5_sl_mon_alltime, barra_c2_mon_alltime, historical_regridded_alltime_ens, amip_regridded_alltime_ens


# endregion


# region plot AGCD, ERA5, BARRA-R2, BARRA-C2, historical, amip, am sm, pct

plt_colnames = ['Annual mean', 'DJF', 'MAM', 'JJA', 'SON']
plt_rownames = ['AGCD', 'ERA5/AGCD - 1', 'BARRA-R2/AGCD - 1', 'BARRA-C2/AGCD - 1', r'$historical$/AGCD - 1', r'$amip$/AGCD - 1']
min_lon, max_lon, min_lat, max_lat = [110.58, 157.34, -43.69, -7.01]

for var2 in ['pr']:
    # var2='pr'
    var1 = cmip6_era5_var[var2]
    print(f'#-------------------------------- {var1} and {var2}')
    
    with open(f'data/obs/agcd/agcd_alltime_{var2}.pkl','rb') as f:
        agcd_alltime = pickle.load(f)
    with open(f'data/obs/era5/mon/era5_sl_mon_alltime_{var1}.pkl', 'rb') as f:
        era5_sl_mon_alltime = pickle.load(f)
    with open(f'data/sim/um/barra_r2/barra_r2_mon_alltime_{var2}.pkl','rb') as f:
        barra_r2_mon_alltime = pickle.load(f)
    with open(f'data/sim/um/barra_c2/barra_c2_mon_alltime_{var2}.pkl','rb') as f:
        barra_c2_mon_alltime = pickle.load(f)
    with open(f'/home/563/qg8515/scratch/data/sim/cmip6/historical_Amon_{var2}_regridded_alltime_ens.pkl', 'rb') as f:
        historical_regridded_alltime_ens = pickle.load(f)
    with open(f'/home/563/qg8515/scratch/data/sim/cmip6/amip_Amon_{var2}_regridded_alltime_ens.pkl', 'rb') as f:
        amip_regridded_alltime_ens = pickle.load(f)
    
    plt_data = {}
    for irow in plt_rownames: plt_data[irow] = {}
    
    plt_data['AGCD']['Annual mean'] = agcd_alltime['ann'].sel(time=slice('2001', '2014')).mean(dim='time')
    agcd_ann = regrid(agcd_alltime['ann'].sel(time=slice('2001', '2014')), ds_out=era5_sl_mon_alltime['am'].pipe(regrid).pipe(replace_x_y_nominal_lat_lon).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon)))
    agcd_am = agcd_ann.mean(dim='time').compute()
    
    plt_data['ERA5/AGCD - 1']['Annual mean'] = ((regrid(era5_sl_mon_alltime['ann'].sel(time=slice('2001', '2014')).mean(dim='time'), ds_out=agcd_am) / agcd_am - 1) * 100).compute()
    # ttest_fdr_res = ttest_fdr_control(
    #     agcd_ann,
    #     regrid(era5_sl_mon_alltime['ann'].sel(time=slice('2001', '2014')), ds_out=agcd_am))
    # plt_data['ERA5/AGCD - 1']['Annual mean'] = plt_data['ERA5/AGCD - 1']['Annual mean'].where(ttest_fdr_res, np.nan)
    
    plt_data['BARRA-R2/AGCD - 1']['Annual mean'] = ((regrid(barra_r2_mon_alltime['ann'].sel(time=slice('2001', '2014')).mean(dim='time'), ds_out=agcd_am) / agcd_am - 1) * 100).compute()
    # ttest_fdr_res = ttest_fdr_control(
    #     agcd_ann,
    #     regrid(barra_r2_mon_alltime['ann'].sel(time=slice('2001', '2014')), ds_out=agcd_am)
    #     )
    # plt_data['BARRA-R2/AGCD - 1']['Annual mean'] = plt_data['BARRA-R2/AGCD - 1']['Annual mean'].where(ttest_fdr_res, np.nan)
    
    plt_data['BARRA-C2/AGCD - 1']['Annual mean'] = ((regrid(barra_c2_mon_alltime['ann'].sel(time=slice('2001', '2014')).mean(dim='time'), ds_out=agcd_am) / agcd_am - 1) * 100).compute()
    # ttest_fdr_res = ttest_fdr_control(
    #     agcd_ann,
    #     regrid(barra_c2_mon_alltime['ann'].sel(time=slice('2001', '2014')), ds_out=agcd_am)
    #     )
    # plt_data['BARRA-C2/AGCD - 1']['Annual mean'] = plt_data['BARRA-C2/AGCD - 1']['Annual mean'].where(ttest_fdr_res, np.nan)
    
    plt_data[r'$historical$/AGCD - 1']['Annual mean'] = ((historical_regridded_alltime_ens['ann'].sel(time=slice('2001', '2014')).mean(dim=['source_id', 'time']).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon)) / agcd_am - 1) * 100).compute()
    # ttest_fdr_res = ttest_fdr_control(
    #     agcd_ann,
    #     historical_regridded_alltime_ens['ann'].sel(time=slice('2001', '2014')).mean(dim='source_id').sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon))
    #     )
    # plt_data[r'$historical$/AGCD - 1']['Annual mean'] = plt_data[r'$historical$/AGCD - 1']['Annual mean'].where(ttest_fdr_res, np.nan)
    
    plt_data[r'$amip$/AGCD - 1']['Annual mean'] = ((amip_regridded_alltime_ens['ann'].sel(time=slice('2001', '2014')).mean(dim=['source_id', 'time']).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon)) / agcd_am - 1) * 100).compute()
    # ttest_fdr_res = ttest_fdr_control(
    #     agcd_ann,
    #     amip_regridded_alltime_ens['ann'].sel(time=slice('2001', '2014')).mean(dim='source_id').sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon))
    #     )
    # plt_data[r'$amip$/AGCD - 1']['Annual mean'] = plt_data[r'$amip$/AGCD - 1']['Annual mean'].where(ttest_fdr_res, np.nan)
    
    for jcolnames in plt_colnames[1:]:
        # jcolnames='DJF'
        print(f'#---------------- {jcolnames}')
        
        plt_data['AGCD'][jcolnames] = agcd_alltime['sea'][agcd_alltime['sea'].time.dt.season==jcolnames].sel(time=slice('2001', '2014')).mean(dim='time')
        agcd_sea = regrid(agcd_alltime['sea'][agcd_alltime['sea'].time.dt.season==jcolnames].sel(time=slice('2001', '2014')), ds_out=era5_sl_mon_alltime['am'].pipe(regrid).pipe(replace_x_y_nominal_lat_lon).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon)))
        agcd_sm = agcd_sea.mean(dim='time').compute()
        
        plt_data['ERA5/AGCD - 1'][jcolnames] = ((regrid(era5_sl_mon_alltime['sea'][era5_sl_mon_alltime['sea'].time.dt.season==jcolnames].sel(time=slice('2001', '2014')).mean(dim='time'), ds_out=agcd_sm) / agcd_sm - 1) * 100).compute()
        
        plt_data['BARRA-R2/AGCD - 1'][jcolnames] = ((regrid(barra_r2_mon_alltime['sea'][barra_r2_mon_alltime['sea'].time.dt.season==jcolnames].sel(time=slice('2001', '2014')).mean(dim='time'), ds_out=agcd_sm) / agcd_sm - 1) * 100).compute()
        
        plt_data['BARRA-C2/AGCD - 1'][jcolnames] = ((regrid(barra_c2_mon_alltime['sea'][barra_c2_mon_alltime['sea'].time.dt.season==jcolnames].sel(time=slice('2001', '2014')).mean(dim='time'), ds_out=agcd_sm) / agcd_sm - 1) * 100).compute()
        
        plt_data[r'$historical$/AGCD - 1'][jcolnames] = ((historical_regridded_alltime_ens['sea'][:, historical_regridded_alltime_ens['sea'].time.dt.season==jcolnames].sel(time=slice('2001', '2014')).mean(dim=['source_id', 'time']).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon)) / agcd_sm - 1) * 100).compute()
        
        plt_data[r'$amip$/AGCD - 1'][jcolnames] = ((amip_regridded_alltime_ens['sea'][:, amip_regridded_alltime_ens['sea'].time.dt.season==jcolnames].sel(time=slice('2001', '2014')).mean(dim=['source_id', 'time']).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon)) / agcd_sm - 1) * 100).compute()
    
    cbar_label1 = '2001-2014 ' + era5_varlabels[var1]
    cbar_label2 = 'Difference in 2001-2014 ' + re.sub(r'\[.*?\]', r'[$\%$]', era5_varlabels[var1])
    extend1 = 'max'
    extend2 = 'max'
    
    # print(stats.describe(np.concatenate([plt_data['AGCD'][colname].values for colname in plt_colnames]), axis=None, nan_policy='omit'))
    # print(stats.describe(np.concatenate([plt_data[rowname][colname].values for rowname in plt_rownames[1:] for colname in plt_colnames]), axis=None, nan_policy='omit'))
    
    if var1 in ['tp']:
        pltlevel1 = np.array([0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 6])
        pltticks1 = np.array([0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 6])
        pltnorm1 = BoundaryNorm(pltlevel1, ncolors=len(pltlevel1)-1, clip=True)
        pltcmp1 = plt.get_cmap('viridis_r', len(pltlevel1)-1)
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-100, cm_max=100, cm_interval1=10, cm_interval2=20, cmap='BrBG_r')
    
    nrow=len(plt_rownames)
    ncol=len(plt_colnames)
    fm_bottom=1.5/(4*nrow+1.5)
    
    fig, axs = plt.subplots(
        nrow, ncol, figsize=np.array([4.4*ncol, 4*nrow + 1.5]) / 2.54,
        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
        gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)
    
    for irow in range(nrow):
        axs[irow, 0].text(-0.05, 0.5, plt_rownames[irow], ha='right', va='center', rotation='vertical', transform=axs[irow, 0].transAxes)
        for jcol in range(ncol):
            axs[irow, jcol] = regional_plot(extent=[min_lon, max_lon, min_lat, max_lat], central_longitude=180, ax_org=axs[irow, jcol])
            axs[irow, jcol].add_feature(cfeature.OCEAN,color='white',zorder=2,edgecolor=None,lw=0)
            axs[irow, jcol].text(0, 1.02, f'({string.ascii_lowercase[irow]}{jcol+1})', ha='left', va='bottom', transform=axs[irow, jcol].transAxes,)
            if irow==0:
                axs[0, jcol].text(0.5, 1.02, plt_colnames[jcol], ha='center', va='bottom', transform=axs[0, jcol].transAxes)
    
    for jcol in range(ncol):
        plt_mesh1 = axs[0, jcol].pcolormesh(
            plt_data[plt_rownames[0]][plt_colnames[jcol]].lon,
            plt_data[plt_rownames[0]][plt_colnames[jcol]].lat,
            plt_data[plt_rownames[0]][plt_colnames[jcol]].values,
            norm=pltnorm1, cmap=pltcmp1, transform=ccrs.PlateCarree(),zorder=1)
    
    for irow in range(nrow-1):
        for jcol in range(ncol):
            plt_mesh2 = axs[irow+1, jcol].pcolormesh(
                plt_data[plt_rownames[irow+1]][plt_colnames[jcol]].lon,
                plt_data[plt_rownames[irow+1]][plt_colnames[jcol]].lat,
                plt_data[plt_rownames[irow+1]][plt_colnames[jcol]].values,
                norm=pltnorm2, cmap=pltcmp2,
                transform=ccrs.PlateCarree(),zorder=1)
    
    cbar1 = fig.colorbar(
        plt_mesh1, #cm.ScalarMappable(norm=pltnorm1, cmap=pltcmp1), #
        format=remove_trailing_zero_pos,
        orientation="horizontal", ticks=pltticks1, extend=extend1,
        cax=fig.add_axes([0.05, fm_bottom-0.01, 0.4, 0.015]))
    cbar1.ax.set_xlabel(cbar_label1)
    cbar2 = fig.colorbar(
        plt_mesh2, #cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), #
        format=remove_trailing_zero_pos,
        orientation="horizontal", ticks=pltticks2, extend=extend2,
        cax=fig.add_axes([0.55, fm_bottom-0.01, 0.4, 0.015]))
    cbar2.ax.set_xlabel(cbar_label2)
    
    fig.subplots_adjust(left=0.03, right=0.995, bottom=fm_bottom, top=0.98)
    fig.savefig(f'figures/4_um/4.0_barra/4.0.0_whole region/4.0.0.1 agcd vs. barra_c2, era5, and cmip6 am sm {var1} pct.png')
    
    del era5_sl_mon_alltime, barra_c2_mon_alltime, historical_regridded_alltime_ens, amip_regridded_alltime_ens


# endregion


# region plot AGCD, ERA5, BARRA-R2, BARRA-C2, historical, amip, am

mpl.rc('font', family='Times New Roman', size=8)
plt_colnames = ['AGCD', 'ERA5 - AGCD', 'BARRA-R2 - AGCD', 'BARRA-C2 - AGCD', r'$historical$ - AGCD', r'$amip$ - AGCD']
min_lon, max_lon, min_lat, max_lat = [110.58, 157.34, -43.69, -7.01]

for var2 in ['pr']:
    # var2='pr'
    var1 = cmip6_era5_var[var2]
    print(f'#-------------------------------- {var1} and {var2}')
    
    with open(f'data/obs/agcd/agcd_alltime_{var2}.pkl','rb') as f:
        agcd_alltime = pickle.load(f)
    with open(f'data/obs/era5/mon/era5_sl_mon_alltime_{var1}.pkl', 'rb') as f:
        era5_sl_mon_alltime = pickle.load(f)
    with open(f'data/sim/um/barra_r2/barra_r2_mon_alltime_{var2}.pkl','rb') as f:
        barra_r2_mon_alltime = pickle.load(f)
    with open(f'data/sim/um/barra_c2/barra_c2_mon_alltime_{var2}.pkl','rb') as f:
        barra_c2_mon_alltime = pickle.load(f)
    with open(f'/home/563/qg8515/scratch/data/sim/cmip6/historical_Amon_{var2}_regridded_alltime_ens.pkl', 'rb') as f:
        historical_regridded_alltime_ens = pickle.load(f)
    with open(f'/home/563/qg8515/scratch/data/sim/cmip6/amip_Amon_{var2}_regridded_alltime_ens.pkl', 'rb') as f:
        amip_regridded_alltime_ens = pickle.load(f)
    
    plt_data = {}
    
    plt_data['AGCD'] = agcd_alltime['ann'].sel(time=slice('2001', '2014')).mean(dim='time')
    agcd_ann = regrid(agcd_alltime['ann'].sel(time=slice('2001', '2014')), ds_out=era5_sl_mon_alltime['am'].pipe(regrid).pipe(replace_x_y_nominal_lat_lon).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon)))
    agcd_am = agcd_ann.mean(dim='time')
    
    plt_data['ERA5 - AGCD'] = (regrid(era5_sl_mon_alltime['ann'].sel(time=slice('2001', '2014')).mean(dim='time'), ds_out=agcd_am) - agcd_am).compute()
    ttest_fdr_res = ttest_fdr_control(
        agcd_ann,
        regrid(era5_sl_mon_alltime['ann'].sel(time=slice('2001', '2014')), ds_out=agcd_am)
        )
    plt_data['ERA5 - AGCD'] = plt_data['ERA5 - AGCD'].where(ttest_fdr_res, np.nan)
    
    plt_data['BARRA-R2 - AGCD'] = (regrid(barra_r2_mon_alltime['ann'].sel(time=slice('2001', '2014')).mean(dim='time'), ds_out=agcd_am) - agcd_am).compute()
    ttest_fdr_res = ttest_fdr_control(
        agcd_ann,
        regrid(barra_r2_mon_alltime['ann'].sel(time=slice('2001', '2014')), ds_out=agcd_am)
        )
    plt_data['BARRA-R2 - AGCD'] = plt_data['BARRA-R2 - AGCD'].where(ttest_fdr_res, np.nan)
    
    plt_data['BARRA-C2 - AGCD'] = (regrid(barra_c2_mon_alltime['ann'].sel(time=slice('2001', '2014')).mean(dim='time'), ds_out=agcd_am) - agcd_am).compute()
    ttest_fdr_res = ttest_fdr_control(
        agcd_ann,
        regrid(barra_c2_mon_alltime['ann'].sel(time=slice('2001', '2014')), ds_out=agcd_am)
        )
    plt_data['BARRA-C2 - AGCD'] = plt_data['BARRA-C2 - AGCD'].where(ttest_fdr_res, np.nan)
    
    plt_data[r'$historical$ - AGCD'] = (historical_regridded_alltime_ens['ann'].sel(time=slice('2001', '2014')).mean(dim=['source_id', 'time']).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon)) - agcd_am).compute()
    ttest_fdr_res = ttest_fdr_control(
        agcd_ann,
        historical_regridded_alltime_ens['ann'].sel(time=slice('2001', '2014')).mean(dim='source_id').sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon))
        )
    plt_data[r'$historical$ - AGCD'] = plt_data[r'$historical$ - AGCD'].where(ttest_fdr_res, np.nan)
    
    plt_data[r'$amip$ - AGCD'] = (amip_regridded_alltime_ens['ann'].sel(time=slice('2001', '2014')).mean(dim=['source_id', 'time']).sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon)) - agcd_am).compute()
    ttest_fdr_res = ttest_fdr_control(
        agcd_ann,
        amip_regridded_alltime_ens['ann'].sel(time=slice('2001', '2014')).mean(dim='source_id').sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon))
        )
    plt_data[r'$amip$ - AGCD'] = plt_data[r'$amip$ - AGCD'].where(ttest_fdr_res, np.nan)
    
    print(stats.describe(plt_data['AGCD'].values, axis=None, nan_policy='omit'))
    print(stats.describe(np.concatenate([plt_data[colname].values for colname in plt_colnames[1:]]), axis=None, nan_policy='omit'))
    
    cbar_label1 = '2001-2014 ' + era5_varlabels[var1]
    cbar_label2 = 'Difference in 2001-2014 ' + era5_varlabels[var1]
    extend1 = 'max'
    extend2 = 'both'
    
    if var1 in ['tp']:
        pltlevel1 = np.array([0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 6])
        pltticks1 = np.array([0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 6])
        pltnorm1 = BoundaryNorm(pltlevel1, ncolors=len(pltlevel1)-1, clip=True)
        pltcmp1 = plt.get_cmap('viridis_r', len(pltlevel1)-1)
        pltlevel2 = np.array([-3, -2, -1.5, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3])
        pltticks2 = np.array([-3, -2, -1.5, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3])
        pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
        pltcmp2 = plt.get_cmap('BrBG', len(pltlevel2)-1)
    
    nrow=1
    ncol=len(plt_colnames)
    fm_bottom=1.5/(4*nrow+1.5)
    
    fig, axs = plt.subplots(
        nrow, ncol, figsize=np.array([4.4*ncol, 4*nrow + 1.5]) / 2.54,
        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
        gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)
    
    for jcol in range(ncol):
        axs[jcol] = regional_plot(extent=[min_lon, max_lon, min_lat, max_lat], central_longitude=180, ax_org=axs[jcol])
        axs[jcol].add_feature(cfeature.OCEAN,color='white',zorder=2,edgecolor=None,lw=0)
        plt_text = f'({string.ascii_lowercase[jcol]}) {plt_colnames[jcol]}'
        axs[jcol].text(0, 1.02, plt_text, ha='left', va='bottom', transform=axs[jcol].transAxes, size=9)
    
    plt_mesh1 = axs[0].pcolormesh(
            plt_data[plt_colnames[0]].lon,
            plt_data[plt_colnames[0]].lat,
            plt_data[plt_colnames[0]].values,
            norm=pltnorm1, cmap=pltcmp1, transform=ccrs.PlateCarree(),zorder=1)
    for jcol in range(ncol-1):
        plt_mesh2 = axs[jcol+1].pcolormesh(
            plt_data[plt_colnames[jcol+1]].lon,
            plt_data[plt_colnames[jcol+1]].lat,
            plt_data[plt_colnames[jcol+1]].values,
            norm=pltnorm2, cmap=pltcmp2, transform=ccrs.PlateCarree(),zorder=1)
    
    cbar1 = fig.colorbar(
        plt_mesh1, #cm.ScalarMappable(norm=pltnorm1, cmap=pltcmp1), #
        format=remove_trailing_zero_pos,
        orientation="horizontal", ticks=pltticks1, extend=extend1,
        cax=fig.add_axes([0.05, fm_bottom-0.05, 0.4, 0.05]))
    cbar1.ax.set_xlabel(cbar_label1)
    cbar2 = fig.colorbar(
        plt_mesh2, #cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), #
        format=remove_trailing_zero_pos,
        orientation="horizontal", ticks=pltticks2, extend=extend2,
        cax=fig.add_axes([0.55, fm_bottom-0.05, 0.4, 0.05]))
    cbar2.ax.set_xlabel(cbar_label2)
    
    fig.subplots_adjust(left=0.005, right=0.995, bottom=fm_bottom, top=0.95)
    fig.savefig(f'figures/4_um/4.0_barra/4.0.0_whole region/4.0.0.1 agcd vs. barra_c2, era5, and cmip6 am {var1}.png')
    
    del agcd_alltime, era5_sl_mon_alltime, barra_r2_mon_alltime, barra_c2_mon_alltime, historical_regridded_alltime_ens, amip_regridded_alltime_ens




'''
agcd_pre = xr.open_mfdataset(sorted(glob.glob('/g/data/zv2/agcd/v2-0-2/precip/total/r001/01month/*')))
agcd_pre.precip.sel(time=slice('1979', '2023'))
'''
# endregion

