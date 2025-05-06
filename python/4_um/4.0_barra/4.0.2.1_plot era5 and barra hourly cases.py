

# qsub -I -q normal -l walltime=4:00:00,ncpus=1,mem=192GB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22


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
from metpy.calc import pressure_to_height_std, geopotential_to_height, specific_humidity_from_dewpoint, relative_humidity_from_dewpoint
from metpy.units import units
import metpy.calc as mpcalc
import pickle
import calendar
from xmip.preprocessing import correct_lon

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
    month_jan,
    month_num,
    monthini,
    seasons,
    seconds_per_d,
    zerok,
    panel_labels,
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
    mon_sea_ann,
    regrid,
    cdo_regrid,)

from statistics0 import (
    ttest_fdr_control,)

# endregion


# region plot hourly data

dss = ['ERA5', 'BARRA-R2', 'BARRA-C2']


year, month, day, hour, minute = 2020, 6, 2, 4, 0
min_lon1, max_lon1, min_lat1, max_lat1 = 80, 220, -70, 20
min_lon, max_lon, min_lat, max_lat = 110.58, 157.34, -43.69, -7.01

pwidth  = 6.6
pheight = 6.6 * (max_lat1 - min_lat1) / (max_lon1 - min_lon1)

nrow = 1
ncol = len(dss)
fm_bottom = 1.6/(pheight*nrow+2.1)
fm_top = 1 - 0.5/(pheight*nrow+2.1)


ds = {}
for ids in dss: ds[ids] = {}

for imode in ['org']:
    # imode = 'org'
    # ['org', 'diff']
    print(f'#-------------------------------- {imode}')
    
    for vars in [['psl', 'uas', 'vas', 'rlut']]:
        # ['prw', 'clwvi', 'clivi', 'cll', 'clm', 'clh', 'clt', 'pr', 'evspsbl', 'evspsblpot', 'hfls', 'hfss', 'tas', 'huss', 'hurs']
        print(f'#---------------- {vars}')
        print(f'#---------------- {[cmip6_era5_var[var] for var in vars]}')
        
        opng = f'figures/4_um/4.0_barra/4.0.5_case_studies/4.0.5.0_{year}-{month}-{day}-{hour} {', '.join(vars)} in {', '.join(dss)} {imode} {min_lon1}_{max_lon1}_{min_lat1}_{max_lat1}.png'
        
        # get data
        for var2 in vars:
            # var2 = 'psl'
            var1 = cmip6_era5_var[var2]
            print(f'#-------- {var2} {var1}')
            
            if var1 in ['t2m', 'si10', 'd2m', 'u10', 'v10', 'u100', 'v100']:
                if var1 == 't2m': vart = '2t'
                if var1 == 'si10': vart = '10si'
                if var1 == 'd2m': vart = '2d'
                if var1 == 'u10': vart = '10u'
                if var1 == 'v10': vart = '10v'
                if var1 == 'u100': vart = '100u'
                if var1 == 'v100': vart = '100v'
                ds['ERA5'][var2] = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/{vart}/{year}/{vart}_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')[var1].sel(time=pd.Timestamp(year,month,day,hour))
            elif var1=='rh2m':
                era5_t2m = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/2t/{year}/2t_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['t2m'].sel(time=pd.Timestamp(year,month,day,hour))
                era5_d2m = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/2d/{year}/2d_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['d2m'].sel(time=pd.Timestamp(year,month,day,hour))
                ds['ERA5'][var2] = relative_humidity_from_dewpoint(era5_t2m * units.K, era5_d2m * units.K) * 100
            elif var1=='q2m':
                era5_sp = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/sp/{year}/sp_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['sp'].sel(time=pd.Timestamp(year,month,day,hour))
                era5_d2m = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/2d/{year}/2d_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['d2m'].sel(time=pd.Timestamp(year,month,day,hour))
                ds['ERA5'][var2] = specific_humidity_from_dewpoint(era5_sp * units.Pa, era5_d2m * units.K) * 1000
            else:
                ds['ERA5'][var2] = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/{var1}/{year}/{var1}_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')[var1].sel(time=pd.Timestamp(year,month,day,hour))
            ds['ERA5'][var2] = ds['ERA5'][var2].rename({'latitude': 'lat', 'longitude': 'lon'}).sel(lat=slice(max_lat1, min_lat1))
            
            ds['BARRA-R2'][var2] = xr.open_dataset(f'/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/{var2}/latest/{var2}_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc')[var2].sel(time=pd.Timestamp(year,month,day,hour), method='nearest')
            ds['BARRA-C2'][var2] = xr.open_dataset(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/{var2}/latest/{var2}_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc')[var2].sel(time=pd.Timestamp(year,month,day,hour), method='nearest')
        
        
        fig, axs = plt.subplots(
            nrow, ncol, figsize=np.array([pwidth*ncol, pheight*nrow+2.1])/2.54,
            subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
            gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)
        
        for jcol in range(ncol):
            axs[jcol] = regional_plot(extent=[min_lon1, max_lon1, min_lat1, max_lat1], central_longitude=180, ax_org=axs[jcol], lw=0.1)
            axs[jcol].text(0, 1.02, f'({string.ascii_lowercase[jcol]}) {dss[jcol]}', ha='left', va='bottom', transform=axs[jcol].transAxes)
            axs[jcol].add_patch(Rectangle(
                (min_lon, min_lat), max_lon-min_lon, max_lat-min_lat,
                ec='red', color='None', lw=0.5,
                transform=ccrs.PlateCarree(), zorder=2))
            axs[jcol].add_patch(Rectangle(
                (ds['BARRA-R2'][var2].lon[0], ds['BARRA-R2'][var2].lat[0]),
                ds['BARRA-R2'][var2].lon[-1] - ds['BARRA-R2'][var2].lon[0],
                ds['BARRA-R2'][var2].lat[-1] - ds['BARRA-R2'][var2].lat[0],
                ec='red', color='None', lw=0.5, linestyle='--',
                transform=ccrs.PlateCarree(), zorder=2))
            axs[jcol].add_patch(Rectangle(
                (ds['BARRA-C2'][var2].lon[0], ds['BARRA-C2'][var2].lat[0]),
                ds['BARRA-C2'][var2].lon[-1] - ds['BARRA-C2'][var2].lon[0],
                ds['BARRA-C2'][var2].lat[-1] - ds['BARRA-C2'][var2].lat[0],
                ec='red', color='None', lw=0.5, linestyle=':',
                transform=ccrs.PlateCarree(), zorder=2))
        
        # plot pressure
        psl_intervals = np.arange(800, 1200+1e-4, 4)
        psl_labels = np.arange(800, 1200+1e-4, 8)
        for jcol, ids in enumerate(dss):
            # ids = 'ERA5'
            print(f'#---- {jcol} {ids}')
            plt_pres = axs[jcol].contour(
                ds[ids]['psl'].lon, ds[ids]['psl'].lat, ds[ids]['psl']/100,
                levels=psl_intervals, colors='tab:orange',
                linewidths=0.15,transform=ccrs.PlateCarree())
            if jcol==0:
                ax_clabel = axs[jcol].clabel(
                    plt_pres, inline=1, colors='tab:orange', fmt='%d',
                    levels=psl_labels, inline_spacing=10, fontsize=8)
            
            nsizes = {'ERA5': 300, 'BARRA-R2': 600, 'BARRA-C2': 1200}
            plot_maxmin_points(
                ds[ids]['psl'].lon, ds[ids]['psl'].lat, ds[ids]['psl']/100,
                axs[jcol], 'max', nsizes[ids], symbol='H', color='tab:orange',
                transform=ccrs.PlateCarree())
            plot_maxmin_points(
                ds[ids]['psl'].lon, ds[ids]['psl'].lat, ds[ids]['psl']/100,
                axs[jcol], 'min', nsizes[ids], symbol='L', color='r',
                transform=ccrs.PlateCarree())
        
        cax = fig.add_axes([0, fm_bottom - 0.02, 1/3, 0.05])
        cax.axis('off')
        cax.legend(
            [plt_pres.legend_elements()[0][0]],
            [r'Mean sea level pressure [$hPa$]'],
            loc='upper center', frameon=False, handlelength=1, columnspacing=1)
        
        # plot wind
        iarrows = {'ERA5': 8, 'BARRA-R2': 20, 'BARRA-C2': 50}
        for jcol, ids in enumerate(dss):
            # ids = 'ERA5'
            print(f'#---- {jcol} {ids}')
            plt_quiver = axs[jcol].quiver(
                ds[ids]['psl'].lon[::iarrows[ids]],
                ds[ids]['psl'].lat[::iarrows[ids]],
                ds[ids]['uas'][::iarrows[ids], ::iarrows[ids]].values,
                ds[ids]['vas'][::iarrows[ids], ::iarrows[ids]].values,
                color='gray', units='height', scale=500,
                width=0.002, headwidth=3, headlength=5, alpha=1,
                transform=ccrs.PlateCarree(), zorder=2)
        
        axs[-1].quiverkey(
            plt_quiver, X=0.15, Y=-0.08, U=10, coordinates='axes',
            label=r'10 $m$ wind speed  [$10\;m\;s^{-1}$]', labelpos='E')
        
        # plot var
        var2 = list(set(vars) - set(['psl', 'uas', 'vas']))[0]
        var1 = cmip6_era5_var[var2]
        if var2 == 'prw':
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=80, cm_interval1=5, cm_interval2=10, cmap='viridis')
            extend = 'max'
        elif var2 in ['cll', 'clm', 'clh', 'clt']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=20, cmap='Blues_r')
            extend = 'neither'
        elif var2 in ['clwvi']:
            pltlevel = np.array([0, 0.01, 0.02, 0.04, 0.06, 0.1, 0.2, 0.3, 0.4])
            pltticks = np.array([0, 0.01, 0.02, 0.04, 0.06, 0.1, 0.2, 0.3, 0.4])
            pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
            pltcmp = plt.get_cmap('Blues', len(pltlevel)-1)
            extend = 'max'
        elif var2 in ['clivi']:
            pltlevel = np.array([0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1])
            pltticks = np.array([0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1])
            pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
            pltcmp = plt.get_cmap('Blues', len(pltlevel)-1)
            extend = 'max'
        elif var2=='pr':
            pltlevel = np.array([0, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20,])
            pltticks = np.array([0, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20,])
            pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
            pltcmp = plt.get_cmap('Blues', len(pltlevel)-1)
            extend = 'max'
        elif var2 in ['evspsbl', 'evspsblpot']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=16, cm_interval1=1, cm_interval2=2, cmap='Blues_r')
            extend = 'max'
        elif var2 in ['hfls']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=-600, cm_max=300, cm_interval1=50, cm_interval2=100, cmap='PRGn', asymmetric=True)
            extend = 'both'
        elif var2 in ['hfss']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=-400, cm_max=400, cm_interval1=50, cm_interval2=100, cmap='PRGn')
            extend = 'both'
        elif var2 in ['sfcWind']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=1, cm_max=11, cm_interval1=1, cm_interval2=2, cmap='viridis_r')
            extend = 'both'
        elif var2 in ['tas']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=-20, cm_max=35, cm_interval1=2.5, cm_interval2=5, cmap='PuOr', asymmetric=True)
            extend = 'both'
        elif var2 in ['huss']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=24, cm_interval1=2, cm_interval2=4, cmap='Blues_r')
            extend = 'max'
        elif var2 in ['hurs']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=20, cmap='Blues_r')
            extend = 'neither'
        elif var2 in ['rsut']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=-500, cm_max=0, cm_interval1=25, cm_interval2=100, cmap='viridis')
            extend = 'min'
        elif var2 in ['rlut']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=-500, cm_max=0, cm_interval1=25, cm_interval2=100, cmap='viridis')
            extend = 'min'
        
        if var1 in ['tp', 'e', 'cp', 'lsp', 'pev']:
            ds['ERA5'][var2] *= 24000
        elif var1 in ['msl']:
            ds['ERA5'][var2] /= 100
        elif var1 in ['sst', 't2m', 'd2m', 'skt']:
            ds['ERA5'][var2] -= zerok
        elif var1 in ['hcc', 'mcc', 'lcc', 'tcc']:
            ds['ERA5'][var2] *= 100
        elif var1 in ['z']:
            ds['ERA5'][var2] /= 9.80665
        elif var1 in ['mper']:
            ds['ERA5'][var2] *= seconds_per_d
        
        if var1 in ['e', 'pev', 'mper']:
            ds['ERA5'][var2] *= (-1)
        
        if var2 in ['pr', 'evspsbl', 'evspsblpot']:
            ds['BARRA-R2'][var2] *= seconds_per_d
            ds['BARRA-C2'][var2] *= seconds_per_d
        elif var2 in ['tas', 'ts']:
            ds['BARRA-R2'][var2] -= zerok
            ds['BARRA-C2'][var2] -= zerok
        elif var2 in ['rlus', 'rluscs', 'rlut', 'rlutcs', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'hfls', 'hfss']:
            ds['BARRA-R2'][var2] *= (-1)
            ds['BARRA-C2'][var2] *= (-1)
        elif var2 in ['psl']:
            ds['BARRA-R2'][var2] /= 100
            ds['BARRA-C2'][var2] /= 100
        elif var2 in ['huss']:
            ds['BARRA-R2'][var2] *= 1000
            ds['BARRA-C2'][var2] *= 1000
        
        for jcol, ids in enumerate(dss):
            # ids = 'ERA5'
            print(f'#---- {jcol} {ids}')
            plt_mesh = axs[jcol].pcolormesh(
                ds[ids][var2].lon,
                ds[ids][var2].lat,
                ds[ids][var2],
                norm=pltnorm, cmap=pltcmp,
                transform=ccrs.PlateCarree(), zorder=1)
        
        cbar = fig.colorbar(
            plt_mesh, #cm.ScalarMappable(norm=pltnorm1, cmap=pltcmp1), #
            format=remove_trailing_zero_pos,
            orientation="horizontal", ticks=pltticks, extend=extend,
            cax=fig.add_axes([1/3, fm_bottom-0.115, 1/3, 0.03]))
        cbar.ax.set_xlabel(era5_varlabels[var1], fontsize=9, labelpad=1)
        cbar.ax.tick_params(labelsize=9, pad=1)
        
        fig.text(0.5, fm_bottom-0.02, f'{year}-{month:02d}-{day:02d} {hour:02d}:00 UTC', ha='center', va='top')
        fig.subplots_adjust(left=0.005, right=0.995, bottom=fm_bottom, top=fm_top)
        fig.savefig(opng)



'''
instanteneous:
BARRA: uas/vas, psl
ERA5: 10u, 10v

Time mean:
BARRA: cll, clm, clh, clt

https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation

ds = xr.open_dataset('/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/mon/cll/latest/cll_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1_mon_202109-202109.nc')

'''
# endregion


