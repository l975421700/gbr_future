

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


# region get and plot data

suite_res = {
    'u-dq700': ['d12km', 'd4p4km'],
}
year, month, day, hour = 2020, 6, 2, 4
ntime = pd.Timestamp(year,month,day,hour) + pd.Timedelta('1h')
year1, month1, day1, hour1 = ntime.year, ntime.month, ntime.day, ntime.hour

min_lon1, max_lon1, min_lat1, max_lat1 = 80, 220, -70, 20
min_lon, max_lon, min_lat, max_lat = 110.58, 157.34, -43.69, -7.01
pwidth  = 6.6
pheight = 6.6 * (max_lat1 - min_lat1) / (max_lon1 - min_lon1)
nrow = 1
ncol = 3
fm_bottom = 1.6/(pheight*nrow+2.1)
fm_top = 1 - 0.5/(pheight*nrow+2.1)

stash_var = {
    'cll':      'STASH_m01s09i203',
    'clm':      'STASH_m01s09i204',
    'clh':      'STASH_m01s09i205',
    'clt':      'STASH_m01s09i216',
    'prw':      'STASH_m01s30i461',
    # 'clwvi':?
    # 'clivi':?
    # 'pr':   'STASH_m01s00i348',
    'skt':      'STASH_m01s00i024',
    'tas':      'STASH_m01s03i236',
    'huss':     'STASH_m01s03i237',
    'hurs':     'STASH_m01s03i245',
    'hus'       'STASH_m01s00i010',
    'ta':       'STASH_m01s16i004',
    'ua':       'STASH_m01s00i002',
    'va':       'STASH_m01s00i003',
    'wa':       'STASH_m01s00i150',
    'wap':      'STASH_m01s30i008',
    'theta':    'STASH_m01s00i004',
    'qcf':      'STASH_m01s00i012',
    'qcl':      'STASH_m01s00i254',
    'qc':       'STASH_m01s16i206',
    'qs':       'STASH_m01s00i271',
    'qr':       'STASH_m01s00i272',
    'qg':       'STASH_m01s00i273',
    'qt':       'STASH_m01s16i207',
    'mv':       'STASH_m01s00i391',
    'mcl':      'STASH_m01s00i392',
    'mcf':      'STASH_m01s00i393',
    'mr':       'STASH_m01s00i394',
    'mg':       'STASH_m01s00i395',
    'mcf2':     'STASH_m01s00i396',
    'pa':       'STASH_m01s00i408',
    'hfls':     'STASH_m01s03i234',
    'hfss':     'STASH_m01s03i217',
    'rlut':     'STASH_m01s02i205',
    'rlds':     'STASH_m01s00i238',
    'rsut':     'STASH_m01s01i205',
    'rsdt':     'STASH_m01s01i207',
    'rsutcs':   'STASH_m01s01i209',
    'rsdscs':   'STASH_m01s01i210',
    'rsds':     'STASH_m01s01i235',
    'rlns':     'STASH_m01s02i201',
    'rlutcs':   'STASH_m01s02i206',
    'rlds':     'STASH_m01s02i207',
    'rldscs':   'STASH_m01s02i208',
    'psl':      'STASH_m01s16i222',
}

regridder = {}
for isuite in suite_res.keys():
    print(f'#-------------------------------- {isuite}')
    
    for var2 in ['prw', 'clm', 'clh', 'clt', 'hfls', 'hfss', 'tas', 'huss', 'hurs', 'rlut']:
        # var2 = 'cll'
        # ['cll', 'rsut']
        # ['prw', 'cll', 'clm', 'clh', 'clt', 'clwvi', 'clivi', 'pr', 'hfls', 'hfss', 'tas', 'huss', 'hurs', 'rsut', 'rlut']
        var1 = cmip6_era5_var[var2]
        print(f'#---------------- {var1} vs. {var2}')
        
        ds = {}
        if var1=='mtuwswrf':
            era5_mtnswrf = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/mtnswrf/{year1}/mtnswrf_era5_oper_sfc_{year1}{month1:02d}01-{year1}{month1:02d}{calendar.monthrange(year1, month1)[1]}.nc')['mtnswrf'].sel(time=pd.Timestamp(year1,month1,day1,hour1))
            era5_mtdwswrf = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/mtdwswrf/{year1}/mtdwswrf_era5_oper_sfc_{year1}{month1:02d}01-{year1}{month1:02d}{calendar.monthrange(year1, month1)[1]}.nc')['mtdwswrf'].sel(time=pd.Timestamp(year1,month1,day1,hour1))
            ds['ERA5'] = era5_mtnswrf - era5_mtdwswrf
        elif var1 in ['msl', 'tcwv', 'hcc', 'mcc', 'lcc', 'tcc', 'tclw', 'tciw']:
            ds['ERA5'] = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/{var1}/{year}/{var1}_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')[var1].sel(time=pd.Timestamp(year,month,day,hour))
        ds['ERA5']['longitude'] = ds['ERA5']['longitude'] % 360
        ds['ERA5'] = ds['ERA5'].sortby(['longitude', 'latitude']).rename({'latitude': 'lat', 'longitude': 'lon'}).sel(lon=slice(min_lon1, max_lon1), lat=slice(min_lat1, max_lat1))
        
        if var1 in ['tp', 'e', 'cp', 'lsp', 'pev']:
            ds['ERA5'] *= 24000 / 24
        elif var1 in ['msl']:
            ds['ERA5'] /= 100
        elif var1 in ['sst', 't2m', 'd2m', 'skt']:
            ds['ERA5'] -= zerok
        elif var1 in ['hcc', 'mcc', 'lcc', 'tcc']:
            ds['ERA5'] *= 100
        elif var1 in ['z']:
            ds['ERA5'] /= 9.80665
        elif var1 in ['mper']:
            ds['ERA5'] *= seconds_per_d / 24
        
        for ires in suite_res[isuite]:
            # ires = 'd4p4km'
            print(f'#-------- {ires}')
            
            ds[ires] = xr.open_dataset(sorted(glob.glob(f'/home/563/qg8515/cylc-run/{isuite}/share/cycle/{year}{month:02d}{day:02d}T0000Z/Australia/{ires}/*/um/umnsaa_pa000.nc'))[0])[stash_var[var2]].sel(T1HR_MN=pd.Timestamp(year,month,day,hour)).rename({'grid_latitude_t': 'lat', 'grid_longitude_t': 'lon'})
            if var2 in ['pr', 'evspsbl', 'evspsblpot']:
                ds[ires] *= seconds_per_d / 24
            elif var2 in ['tas', 'ts']:
                ds[ires] -= zerok
            elif var2 in ['rlus', 'rluscs', 'rlut', 'rlutcs', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'hfls', 'hfss']:
                ds[ires] *= (-1)
            elif var2 in ['psl']:
                ds[ires] /= 100
            elif var2 in ['huss']:
                ds[ires] *= 1000
            elif var2 in ['cll', 'clm', 'clh', 'clt']:
                ds[ires] *= 100
        
        if var2 == 'prw':
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=80, cm_interval1=5, cm_interval2=10, cmap='Blues_r')
            extend = 'max'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-10,cm_max=10,cm_interval1=1,cm_interval2=2,cmap='BrBG_r')
            extend2 = 'both'
        elif var2 in ['cll', 'clm', 'clh', 'clt']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=100, cm_interval1=5, cm_interval2=10, cmap='Blues_r')
            extend = 'neither'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-100,cm_max=100,cm_interval1=10,cm_interval2=20,cmap='BrBG_r')
            extend2 = 'neither'
        elif var2 in ['clwvi']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=0.6, cm_interval1=0.05, cm_interval2=0.1, cmap='Blues_r')
            extend = 'max'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-0.6,cm_max=0.6,cm_interval1=0.1,cm_interval2=0.1,cmap='BrBG_r')
            extend2 = 'both'
        elif var2 in ['clivi']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=1, cm_interval1=0.1, cm_interval2=0.2, cmap='Blues_r')
            extend = 'max'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-1,cm_max=1,cm_interval1=0.1,cm_interval2=0.2,cmap='BrBG_r')
            extend2 = 'both'
        elif var2=='pr':
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=8, cm_interval1=1, cm_interval2=1, cmap='Blues_r')
            extend = 'max'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-4,cm_max=4,cm_interval1=1,cm_interval2=1,cmap='BrBG_r')
            extend2 = 'both'
        elif var2 in ['evspsbl', 'evspsblpot']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=1, cm_interval1=0.1, cm_interval2=0.2, cmap='Blues_r')
            extend = 'max'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-1,cm_max=1,cm_interval1=0.1,cm_interval2=0.2,cmap='BrBG_r')
            extend2 = 'both'
        elif var2 in ['hfls']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=-600, cm_max=300, cm_interval1=50, cm_interval2=100, cmap='PRGn', asymmetric=True)
            extend = 'both'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-30,cm_max=30,cm_interval1=5,cm_interval2=10,cmap='BrBG')
            extend2 = 'both'
        elif var2 in ['hfss']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=-400, cm_max=400, cm_interval1=50, cm_interval2=100, cmap='PRGn')
            extend = 'both'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-30,cm_max=30,cm_interval1=5,cm_interval2=10,cmap='BrBG')
            extend2 = 'both'
        elif var2 in ['sfcWind']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=24, cm_interval1=2, cm_interval2=4, cmap='Purples_r')
            extend = 'max'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-4,cm_max=4,cm_interval1=1,cm_interval2=1,cmap='BrBG_r')
            extend2 = 'both'
        elif var2 in ['tas']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=-20, cm_max=35, cm_interval1=2.5, cm_interval2=5, cmap='PuOr', asymmetric=True)
            extend = 'both'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-4,cm_max=4,cm_interval1=1,cm_interval2=1,cmap='BrBG')
            extend2 = 'both'
        elif var2 in ['huss']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=24, cm_interval1=2, cm_interval2=4, cmap='Blues_r')
            extend = 'max'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-8,cm_max=8,cm_interval1=1,cm_interval2=2,cmap='BrBG_r')
            extend2 = 'both'
        elif var2 in ['hurs']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=20, cmap='Blues_r')
            extend = 'neither'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-10,cm_max=10,cm_interval1=1,cm_interval2=2,cmap='BrBG_r')
            extend2 = 'both'
        elif var2 in ['rsut']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=-800, cm_max=0, cm_interval1=50, cm_interval2=100, cmap='Greens')
            extend = 'min'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-400,cm_max=400,cm_interval1=50,cm_interval2=100,cmap='BrBG')
            extend2 = 'both'
        elif var2 in ['rlut']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=-360, cm_max=0, cm_interval1=20, cm_interval2=40, cmap='Greens')
            extend = 'min'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-180,cm_max=180,cm_interval1=20,cm_interval2=40,cmap='BrBG')
            extend2 = 'both'
        
        for imode in ['org', 'diff']:
            # imode = 'org'
            print(f'#-------- {imode}')
            
            if imode=='org':
                plt_colnames = list(ds.keys())
            elif imode=='diff':
                plt_colnames = ['ERA5'] + [f'{suite_res[isuite][0]} - ERA5'] + [f'{suite_res[isuite][1]} - {suite_res[isuite][0]}']
            
            opng = f'figures/4_um/4.1_access_ram3/4.1.0_{year}-{month}-{day}-{hour} {var2} in {isuite} {', '.join(suite_res[isuite])}, and ERA5, {imode} {min_lon1}_{max_lon1}_{min_lat1}_{max_lat1}.png'
            
            fig, axs = plt.subplots(
                nrow, ncol,
                figsize=np.array([pwidth*ncol, pheight*nrow+2.1])/2.54,
                subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
                gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)
            
            for jcol in range(ncol):
                axs[jcol] = regional_plot(extent=[min_lon1, max_lon1, min_lat1, max_lat1], central_longitude=180, ax_org=axs[jcol], lw=0.1)
                axs[jcol].text(0, 1.02, f'({string.ascii_lowercase[jcol]}) {plt_colnames[jcol]}', ha='left', va='bottom', transform=axs[jcol].transAxes)
                axs[jcol].add_patch(Rectangle(
                    (min_lon, min_lat), max_lon-min_lon, max_lat-min_lat,
                    ec='red', color='None', lw=0.5,
                    transform=ccrs.PlateCarree(), zorder=2))
                axs[jcol].add_patch(Rectangle(
                    (ds[suite_res[isuite][0]].lon[0], ds[suite_res[isuite][0]].lat[0]),
                    ds[suite_res[isuite][0]].lon[-1] - ds[suite_res[isuite][0]].lon[0],
                    ds[suite_res[isuite][0]].lat[-1] - ds[suite_res[isuite][0]].lat[0],
                    ec='red', color='None', lw=0.5, linestyle='--',
                    transform=ccrs.PlateCarree(), zorder=2))
                axs[jcol].add_patch(Rectangle(
                    (ds[suite_res[isuite][1]].lon[0], ds[suite_res[isuite][1]].lat[0]),
                    ds[suite_res[isuite][1]].lon[-1] - ds[suite_res[isuite][1]].lon[0],
                    ds[suite_res[isuite][1]].lat[-1] - ds[suite_res[isuite][1]].lat[0],
                    ec='red', color='None', lw=0.5, linestyle=':',
                    transform=ccrs.PlateCarree(), zorder=2))
            
            if imode=='org':
                for jcol, ids in enumerate(ds.keys()):
                    print(f'#---- {jcol} {ids}')
                    plt_mesh = axs[jcol].pcolormesh(
                        ds[ids].lon,
                        ds[ids].lat,
                        ds[ids],
                        norm=pltnorm, cmap=pltcmp,
                        transform=ccrs.PlateCarree(), zorder=1)
                    cbar = fig.colorbar(
                        plt_mesh, #cm.ScalarMappable(norm=pltnorm1, cmap=pltcmp1), #
                        format=remove_trailing_zero_pos,
                        orientation="horizontal", ticks=pltticks, extend=extend,
                        cax=fig.add_axes([1/3, fm_bottom-0.115, 1/3, 0.03]))
                    cbar.ax.set_xlabel(era5_varlabels[var1].replace('day^{-1}', 'hour^{-1}'), fontsize=9, labelpad=1)
                    cbar.ax.tick_params(labelsize=9, pad=1)
            elif imode=='diff':
                plt_mesh = axs[0].pcolormesh(
                    ds['ERA5'].lon, ds['ERA5'].lat, ds['ERA5'],
                    norm=pltnorm, cmap=pltcmp,
                    transform=ccrs.PlateCarree(), zorder=1)
                for jcol, ids1, ids2 in zip(range(1, ncol), list(ds.keys())[1:], list(ds.keys())[:-1]):
                    print(f'#-------- {jcol} {ids1} {ids2}')
                    if not f'{ids1} - {ids2}' in regridder.keys():
                        regridder[f'{ids1} - {ids2}'] = xe.Regridder(
                            ds[ids1],
                            ds[ids2].sel(lon=slice(ds[ids1].lon[0], ds[ids1].lon[-1]), lat=slice(ds[ids1].lat[0], ds[ids1].lat[-1])),
                            method='bilinear')
                    plt_data = regridder[f'{ids1} - {ids2}'](ds[ids1]) - ds[ids2].sel(lon=slice(ds[ids1].lon[0], ds[ids1].lon[-1]), lat=slice(ds[ids1].lat[0], ds[ids1].lat[-1]))
                    plt_mesh2 = axs[jcol].pcolormesh(
                        plt_data.lon, plt_data.lat, plt_data,
                        norm=pltnorm2, cmap=pltcmp2,
                        transform=ccrs.PlateCarree(), zorder=1)
                cbar = fig.colorbar(
                    plt_mesh, #cm.ScalarMappable(norm=pltnorm1, cmap=pltcmp1), #
                    format=remove_trailing_zero_pos,
                    orientation="horizontal", ticks=pltticks, extend=extend,
                    cax=fig.add_axes([0.05, fm_bottom-0.115, 0.4, 0.03]))
                cbar.ax.set_xlabel(era5_varlabels[var1].replace('day^{-1}', 'hour^{-1}'), fontsize=9, labelpad=1)
                cbar.ax.tick_params(labelsize=9, pad=1)
                cbar2 = fig.colorbar(
                    plt_mesh2, #cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), #
                    format=remove_trailing_zero_pos,
                    orientation="horizontal", ticks=pltticks2, extend=extend2,
                    cax=fig.add_axes([0.55, fm_bottom-0.115, 0.4, 0.03]))
                cbar2.ax.set_xlabel(f'Difference in {era5_varlabels[var1].replace('day^{-1}', 'hour^{-1}')}',
                                    fontsize=9, labelpad=1)
                cbar2.ax.tick_params(labelsize=9, pad=1)
            
            fig.text(0.5, fm_bottom-0.02, f'{year}-{month:02d}-{day:02d} {hour:02d}:00 UTC', ha='center', va='top')
            fig.subplots_adjust(left=0.005, right=0.995, bottom=fm_bottom, top=fm_top)
            fig.savefig(opng)



# endregion


# region check data

year, month, day, hour = 2020, 6, 2, 4
ds = {}
isuite = 'u-dq700'
# ires = 'd4p4km'
ires = 'd12km'

ds[ires] = xr.open_dataset(sorted(glob.glob(f'/home/563/qg8515/cylc-run/{isuite}/share/cycle/{year}{month:02d}{day:02d}T0000Z/Australia/{ires}/*/um/umnsaa_pa000.nc'))[0])

ds[ires]['STASH_m01s03i513'][0]

ds[ires]['STASH_m01s04i210'][0, :, :5, :5]
# 5242880.

# endregion
