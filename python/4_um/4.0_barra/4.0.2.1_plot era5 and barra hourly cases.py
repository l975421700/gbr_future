

# qsub -I -q normal -P nf33 -l walltime=3:00:00,ncpus=1,mem=10GB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22


# region import packages

# data analysis
import numpy as np
import xarray as xr
import pandas as pd
from metpy.calc import specific_humidity_from_dewpoint, relative_humidity_from_dewpoint, vertical_velocity_pressure, mixing_ratio_from_specific_humidity
from metpy.units import units
import calendar
import xesmf as xe

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib import cm
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=10)
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})
mpl.use('Agg')
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as ticker
from matplotlib.colors import BoundaryNorm

# management
import os
import sys  # print(sys.path)
sys.path.append(os.getcwd() + '/code/gbr_future/module')
import psutil
process = psutil.Process()
# print(process.memory_info().rss / 2**30)
import string
import time
import glob

# self defined
from mapplot import (
    regional_plot,
    plot_maxmin_points,
    remove_trailing_zero_pos)

from namelist import (
    seconds_per_d,
    zerok,
    era5_varlabels,
    cmip6_era5_var)

from component_plot import (
    plt_mesh_pars)

# endregion


# region animate hourly data
# Memory Used: 6.19GB; Walltime Used: 01:17:26

imode = 'diff' #'org' #
vars = ['cll'] #['psl', 'cll'] #['psl', 'pr'] #['psl', 'huss'] #['psl', 'prw'] #['psl', 'clm'] #['psl', 'clh'] #['psl', 'clt'] #['psl', 'clwvi'] #['psl', 'clivi'] #['psl', 'evspsbl'] #['psl', 'hfls'] #['psl', 'hfss'] #['psl', 'sfcWind'] #['psl', 'tas'] #['psl', 'hurs'] #['psl', 'rsut'] #['psl', 'rlut'] #['psl', 'uas', 'vas'] #

year, month = 2020, 6
start_day = pd.Timestamp(year, month, 1, 0, 0)
end_day = start_day + pd.Timedelta(days=calendar.monthrange(year, month)[1])
time_series = pd.date_range(start=start_day, end=end_day, freq='1h')[:-1]

dss = ['ERA5', 'BARRA-R2', 'BARRA-C2']
regridder = {}
min_lon1, max_lon1, min_lat1, max_lat1 = 80, 220, -70, 20
min_lon, max_lon, min_lat, max_lat = 110.58, 157.34, -43.69, -7.01
pwidth  = 6.6
pheight = 6.6 * (max_lat1 - min_lat1) / (max_lon1 - min_lon1)
nrow = 1
ncol = len(dss)
fm_bottom = 1.6/(pheight*nrow+2.1)
fm_top = 1 - 0.5/(pheight*nrow+2.1)
psl_intervals = np.arange(800, 1200+1e-4, 4)
psl_labels = np.arange(800, 1200+1e-4, 8)
nsizes = {'ERA5': 300, 'BARRA-R2': 600, 'BARRA-C2': 1200}
iarrows = {'ERA5': 8, 'BARRA-R2': 20, 'BARRA-C2': 50}

if (len(set(vars) - set(['psl', 'uas', 'vas']))==1):
    var2 = list(set(vars) - set(['psl', 'uas', 'vas']))[0]
    var1 = cmip6_era5_var[var2]
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


omp4 = f'figures/4_um/4.0_barra/4.0.5_case_studies/4.0.5.1_{year}-{month} {', '.join(vars)} in {', '.join(dss)} {imode} {min_lon1}_{max_lon1}_{min_lat1}_{max_lat1}.mp4'
if imode=='org':
    plt_colnames = dss
elif imode=='diff':
    plt_colnames = [dss[0]] + [f'{ids1} - {ids2}' for ids1, ids2 in zip(dss[1:], dss[:-1])]

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([pwidth*ncol, pheight*nrow+2.1])/2.54,
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
        (88.48, -57.97), 207.39 - 88.48, 12.98 - -57.97,
        ec='red', color='None', lw=0.5, linestyle='--',
        transform=ccrs.PlateCarree(), zorder=2))
    axs[jcol].add_patch(Rectangle(
        (108.02, -45.69), 159.9 - 108.02, -5.01 - (-45.69),
        ec='red', color='None', lw=0.5, linestyle=':',
        transform=ccrs.PlateCarree(), zorder=2))

plt_objs = []
def update_frames(itime):
    # itime = 0
    time1 = time.perf_counter()
    global plt_objs
    for plt_obj in plt_objs:
        try:
            plt_obj.remove()
        except ValueError:
            pass
    plt_objs = []
    
    day = time_series[itime].day
    hour = time_series[itime].hour
    ntime = pd.Timestamp(year,month,day,hour) + pd.Timedelta('1h')
    year1, month1, day1, hour1 = ntime.year, ntime.month, ntime.day, ntime.hour
    print(f'#-------------------------------- {day:02d} {hour:02d}')
    
    ds = {}
    for ids in dss: ds[ids] = {}
    for var2 in vars:
        # var2 = 'hurs'
        var1 = cmip6_era5_var[var2]
        # print(f'#-------- {var2} {var1}')
        
        if var1 in ['t2m', 'd2m', 'u10', 'v10', 'u100', 'v100']:
            if var1 == 't2m': vart = '2t'
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
            del era5_t2m, era5_d2m
        elif var1=='q2m':
            era5_sp = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/sp/{year}/sp_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['sp'].sel(time=pd.Timestamp(year,month,day,hour))
            era5_d2m = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/2d/{year}/2d_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['d2m'].sel(time=pd.Timestamp(year,month,day,hour))
            ds['ERA5'][var2] = specific_humidity_from_dewpoint(era5_sp * units.Pa, era5_d2m * units.K) * 1000
            del era5_sp, era5_d2m
        elif var1=='mtuwswrf':
            era5_mtnswrf = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/mtnswrf/{year1}/mtnswrf_era5_oper_sfc_{year1}{month1:02d}01-{year1}{month1:02d}{calendar.monthrange(year1, month1)[1]}.nc')['mtnswrf'].sel(time=pd.Timestamp(year1,month1,day1,hour1))
            era5_mtdwswrf = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/mtdwswrf/{year1}/mtdwswrf_era5_oper_sfc_{year1}{month1:02d}01-{year1}{month1:02d}{calendar.monthrange(year1, month1)[1]}.nc')['mtdwswrf'].sel(time=pd.Timestamp(year1,month1,day1,hour1))
            ds['ERA5'][var2] = era5_mtnswrf - era5_mtdwswrf
            del era5_mtnswrf, era5_mtdwswrf
        elif var1=='si10':
            era5_u10 = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/10u/{year}/10u_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['u10'].sel(time=pd.Timestamp(year,month,day,hour))
            era5_v10 = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/10v/{year}/10v_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['v10'].sel(time=pd.Timestamp(year,month,day,hour))
            ds['ERA5'][var2] = (era5_u10**2 + era5_v10**2)**0.5
            del era5_u10, era5_v10
        elif var1 in ['tp', 'e', 'pev', 'mslhf', 'msshf', 'mtnlwrf']:
            ds['ERA5'][var2] = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/{var1}/{year1}/{var1}_era5_oper_sfc_{year1}{month1:02d}01-{year1}{month1:02d}{calendar.monthrange(year1, month1)[1]}.nc')[var1].sel(time=pd.Timestamp(year1,month1,day1,hour1))
        elif var1 in ['msl', 'tcwv', 'hcc','mcc','lcc','tcc', 'tclw','tciw']:
            ds['ERA5'][var2] = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/{var1}/{year}/{var1}_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')[var1].sel(time=pd.Timestamp(year,month,day,hour))
        ds['ERA5'][var2]['longitude'] = ds['ERA5'][var2]['longitude'] % 360
        ds['ERA5'][var2] = ds['ERA5'][var2].sortby(['longitude', 'latitude']).rename({'latitude': 'lat', 'longitude': 'lon'}).sel(lon=slice(min_lon1, max_lon1), lat=slice(min_lat1, max_lat1))
        
        if var2 in ['psl', 'uas', 'vas', 'prw', 'clwvi', 'clivi', 'sfcWind', 'tas', 'huss', 'hurs']:
            ds['BARRA-R2'][var2] = xr.open_dataset(f'/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/{var2}/latest/{var2}_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc')[var2].sel(time=pd.Timestamp(year,month,day,hour))
            ds['BARRA-C2'][var2] = xr.open_dataset(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/{var2}/latest/{var2}_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc')[var2].sel(time=pd.Timestamp(year,month,day,hour))
        elif var2 in ['cll', 'clm', 'clh', 'clt', 'pr', 'evspsbl', 'evspsblpot', 'hfls', 'hfss', 'rsut', 'rlut']:
            ds['BARRA-R2'][var2] = xr.open_dataset(f'/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/{var2}/latest/{var2}_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc')[var2].sel(time=pd.Timestamp(year,month,day,hour) + pd.Timedelta('30min'))
            ds['BARRA-C2'][var2] = xr.open_dataset(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/{var2}/latest/{var2}_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc')[var2].sel(time=pd.Timestamp(year,month,day,hour) + pd.Timedelta('30min'))
        
        if var1 in ['tp', 'e', 'cp', 'lsp', 'pev']:
            ds['ERA5'][var2] *= 24000 / 24
        elif var1 in ['msl']:
            ds['ERA5'][var2] /= 100
        elif var1 in ['sst', 't2m', 'd2m', 'skt']:
            ds['ERA5'][var2] -= zerok
        elif var1 in ['hcc', 'mcc', 'lcc', 'tcc']:
            ds['ERA5'][var2] *= 100
        elif var1 in ['z']:
            ds['ERA5'][var2] /= 9.80665
        elif var1 in ['mper']:
            ds['ERA5'][var2] *= seconds_per_d / 24
        
        if var1 in ['e', 'pev', 'mper']:
            ds['ERA5'][var2] *= (-1)
        
        if var2 in ['pr', 'evspsbl', 'evspsblpot']:
            ds['BARRA-R2'][var2] *= seconds_per_d / 24
            ds['BARRA-C2'][var2] *= seconds_per_d / 24
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
    
    if 'psl' in vars:
        plt_pres = []
        for jcol, ids in enumerate(dss):
            # jcol=0; ids = 'ERA5'
            # print(f'#---- {jcol} {ids}')
            plt_pres.append(axs[jcol].contour(
                ds[ids]['psl'].lon,ds[ids]['psl'].lat,ds[ids]['psl'],
                levels=psl_intervals, colors='tab:orange',
                linewidths=0.15,transform=ccrs.PlateCarree()))
            # plt_pres[jcol].__class__ = mpl.contour.QuadContourSet
            if jcol==0:
                ax_clabel = axs[jcol].clabel(
                    plt_pres[0], inline=1, colors='tab:orange', fmt='%d',
                    levels=psl_labels, inline_spacing=10, fontsize=8)
                plt_H = plot_maxmin_points(
                    ds[ids]['psl'].lon, ds[ids]['psl'].lat, ds[ids]['psl'],
                    axs[jcol], 'max', nsizes[ids], symbol='H',
                    color='tab:orange', transform=ccrs.PlateCarree())
                plt_L = plot_maxmin_points(
                    ds[ids]['psl'].lon, ds[ids]['psl'].lat, ds[ids]['psl'],
                    axs[jcol], 'min', nsizes[ids], symbol='L',
                    color='r', transform=ccrs.PlateCarree())
        
        if itime==0:
            cax = fig.add_axes([0, fm_bottom - 0.02, 1/3, 0.05])
            cax.axis('off')
            cax.legend(
                [plt_pres[0].legend_elements()[0][0]],
                [r'Mean sea level pressure [$hPa$]'],
                loc='upper center', frameon=False, handlelength=1)
        
        plt_objs += plt_pres + ax_clabel + plt_H + plt_L
    
    if all(ivar in vars for ivar in ['vas', 'uas']):
        plt_quiver = []
        for jcol, ids in enumerate(dss):
            # ids = 'ERA5'
            # print(f'#---- {jcol} {ids}')
            plt_quiver.append(axs[jcol].quiver(
                ds[ids]['uas'].lon[::iarrows[ids]],
                ds[ids]['uas'].lat[::iarrows[ids]],
                ds[ids]['uas'][::iarrows[ids], ::iarrows[ids]].values,
                ds[ids]['vas'][::iarrows[ids], ::iarrows[ids]].values,
                color='gray', units='height', scale=500,
                width=0.002, headwidth=3, headlength=5, alpha=1,
                transform=ccrs.PlateCarree(), zorder=2))
        
        plt_quiverkey = axs[-1].quiverkey(
            plt_quiver[-1], X=0.25, Y=-0.08, U=10, coordinates='axes',
            label=r'10 $m$ wind [$10\;m\;s^{-1}$]', labelpos='E')
        plt_objs += plt_quiver + [plt_quiverkey]
    
    if (len(set(vars) - set(['psl', 'uas', 'vas']))==1):
        var2 = list(set(vars) - set(['psl', 'uas', 'vas']))[0]
        var1 = cmip6_era5_var[var2]
        plt_mesh = []
        if imode=='org':
            for jcol, ids in enumerate(dss):
                # ids = 'ERA5'
                # print(f'#---- {jcol} {ids}')
                plt_mesh.append(axs[jcol].pcolormesh(
                    ds[ids][var2].lon, ds[ids][var2].lat, ds[ids][var2],
                    norm=pltnorm, cmap=pltcmp,
                    transform=ccrs.PlateCarree(), zorder=1))
        elif imode=='diff':
            plt_mesh.append(axs[0].pcolormesh(
                ds[dss[0]][var2].lon, ds[dss[0]][var2].lat, ds[dss[0]][var2],
                norm=pltnorm, cmap=pltcmp,
                transform=ccrs.PlateCarree(), zorder=1))
            for jcol, ids1, ids2 in zip(range(1, len(dss)), dss[1:], dss[:-1]):
                # jcol=1; ids1='BARRA-R2'; ids2='ERA5'
                # print(f'#-------- {jcol} {ids1} {ids2}')
                if not f'{ids1} - {ids2}' in regridder.keys():
                    regridder[f'{ids1} - {ids2}'] = xe.Regridder(
                        ds[ids1][var2],
                        ds[ids2][var2].sel(lon=slice(ds[ids1][var2].lon[0], ds[ids1][var2].lon[-1]), lat=slice(ds[ids1][var2].lat[0], ds[ids1][var2].lat[-1])),
                        method='bilinear')
                plt_data = regridder[f'{ids1} - {ids2}'](ds[ids1][var2]) - ds[ids2][var2].sel(lon=slice(ds[ids1][var2].lon[0], ds[ids1][var2].lon[-1]), lat=slice(ds[ids1][var2].lat[0], ds[ids1][var2].lat[-1]))
                plt_mesh.append(axs[jcol].pcolormesh(
                    plt_data.lon, plt_data.lat, plt_data,
                    norm=pltnorm2, cmap=pltcmp2,
                    transform=ccrs.PlateCarree(), zorder=1))
        
        if itime==0:
            if imode=='org':
                cbar = fig.colorbar(
                    plt_mesh[0], #cm.ScalarMappable(norm=pltnorm1, cmap=pltcmp1), #
                    format=remove_trailing_zero_pos,
                    orientation="horizontal", ticks=pltticks, extend=extend,
                    cax=fig.add_axes([1/3, fm_bottom-0.115, 1/3, 0.03]))
                cbar.ax.set_xlabel(era5_varlabels[var1].replace('day^{-1}', 'hour^{-1}'), fontsize=9, labelpad=1)
                cbar.ax.tick_params(labelsize=9, pad=1)
            elif imode=='diff':
                cbar = fig.colorbar(
                    plt_mesh[0], #cm.ScalarMappable(norm=pltnorm1, cmap=pltcmp1), #
                    format=remove_trailing_zero_pos,
                    orientation="horizontal", ticks=pltticks, extend=extend,
                    cax=fig.add_axes([0.05, fm_bottom-0.115, 0.4, 0.03]))
                cbar.ax.set_xlabel(era5_varlabels[var1].replace('day^{-1}', 'hour^{-1}'), fontsize=9, labelpad=1)
                cbar.ax.tick_params(labelsize=9, pad=1)
                cbar2 = fig.colorbar(
                    plt_mesh[1], #cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), #
                    format=remove_trailing_zero_pos,
                    orientation="horizontal", ticks=pltticks2, extend=extend2,
                    cax=fig.add_axes([0.55, fm_bottom-0.115, 0.4, 0.03]))
                cbar2.ax.set_xlabel(f'Difference in {era5_varlabels[var1].replace('day^{-1}', 'hour^{-1}')}',
                                    fontsize=9, labelpad=1)
                cbar2.ax.tick_params(labelsize=9, pad=1)
        
        plt_objs += plt_mesh
    
    plt_text = fig.text(0.5, fm_bottom-0.02, f'{year}-{month:02d}-{day:02d} {hour:02d}:00 UTC', ha='center', va='top')
    plt_objs += [plt_text]
    time2 = time.perf_counter()
    del ds
    print(f'Execution time: {time2 - time1:.1f} s')
    print(f'Memory usage: {np.round(process.memory_info().rss/2**30, 3)} GB')
    return(plt_objs)

fig.subplots_adjust(left=0.005, right=0.995, bottom=fm_bottom, top=fm_top)
ani = animation.FuncAnimation(
    fig, update_frames, frames=len(time_series), interval=500, blit=False)
if os.path.exists(omp4): os.remove(omp4)
ani.save(omp4,progress_callback=lambda iframe,n:print(f'Frame {iframe}/{n-1}'))



# endregion


# region plot hourly data

dss = ['ERA5', 'BARRA-R2', 'BARRA-C2']
regridder = {}

year, month, day, hour = 2020, 6, 2, 4
ntime = pd.Timestamp(year,month,day,hour) + pd.Timedelta('1h')
year1, month1, day1, hour1 = ntime.year, ntime.month, ntime.day, ntime.hour

min_lon1, max_lon1, min_lat1, max_lat1 = 80, 220, -70, 20
min_lon, max_lon, min_lat, max_lat = 110.58, 157.34, -43.69, -7.01

pwidth  = 6.6
pheight = 6.6 * (max_lat1 - min_lat1) / (max_lon1 - min_lon1)

nrow = 1
ncol = len(dss)
fm_bottom = 1.6/(pheight*nrow+2.1)
fm_top = 1 - 0.5/(pheight*nrow+2.1)


for imode in ['org']:
    # imode = 'org'
    # ['org', 'diff']
    print(f'#-------------------------------- {imode}')
    
    for vars in [['psl', 'uas', 'vas']]:
        # [['psl', 'uas', 'vas']] + [['psl', ivar] for ivar in ['prw', 'cll', 'clm', 'clh', 'clt', 'clwvi', 'clivi', 'pr', 'evspsbl', 'evspsblpot', 'hfls', 'hfss', 'sfcWind', 'tas', 'huss', 'hurs', 'rsut', 'rlut']]
        # ['prw', 'cll', 'clm', 'clh', 'clt', 'clwvi', 'clivi', 'pr', 'evspsbl', 'evspsblpot', 'hfls', 'hfss', 'sfcWind', 'tas', 'huss', 'hurs', 'rsut', 'rlut']
        print(f'#---------------- {vars}')
        print(f'#---------------- {[cmip6_era5_var[var] for var in vars]}')
        time1 = time.perf_counter()
        
        opng = f'figures/4_um/4.0_barra/4.0.5_case_studies/4.0.5.0_{year}-{month}-{day}-{hour} {', '.join(vars)} in {', '.join(dss)} {imode} {min_lon1}_{max_lon1}_{min_lat1}_{max_lat1}.png'
        
        ds = {}
        for ids in dss: ds[ids] = {}
        for var2 in vars:
            # var2 = 'hurs'
            var1 = cmip6_era5_var[var2]
            # print(f'#-------- {var2} {var1}')
            
            if var1 in ['t2m', 'd2m', 'u10', 'v10', 'u100', 'v100']:
                if var1 == 't2m': vart = '2t'
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
            elif var1=='mtuwswrf':
                era5_mtnswrf = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/mtnswrf/{year1}/mtnswrf_era5_oper_sfc_{year1}{month1:02d}01-{year1}{month1:02d}{calendar.monthrange(year1, month1)[1]}.nc')['mtnswrf'].sel(time=pd.Timestamp(year1,month1,day1,hour1))
                era5_mtdwswrf = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/mtdwswrf/{year1}/mtdwswrf_era5_oper_sfc_{year1}{month1:02d}01-{year1}{month1:02d}{calendar.monthrange(year1, month1)[1]}.nc')['mtdwswrf'].sel(time=pd.Timestamp(year1,month1,day1,hour1))
                ds['ERA5'][var2] = era5_mtnswrf - era5_mtdwswrf
            elif var1=='si10':
                era5_u10 = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/10u/{year}/10u_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['u10'].sel(time=pd.Timestamp(year,month,day,hour))
                era5_v10 = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/10v/{year}/10v_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['v10'].sel(time=pd.Timestamp(year,month,day,hour))
                ds['ERA5'][var2] = (era5_u10**2 + era5_v10**2)**0.5
            elif var1 in ['tp', 'e', 'pev', 'mslhf', 'msshf', 'mtnlwrf']:
                ds['ERA5'][var2] = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/{var1}/{year1}/{var1}_era5_oper_sfc_{year1}{month1:02d}01-{year1}{month1:02d}{calendar.monthrange(year1, month1)[1]}.nc')[var1].sel(time=pd.Timestamp(year1,month1,day1,hour1))
            elif var1 in ['msl', 'tcwv', 'hcc', 'mcc', 'lcc', 'tcc', 'tclw', 'tciw']:
                ds['ERA5'][var2] = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/{var1}/{year}/{var1}_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')[var1].sel(time=pd.Timestamp(year,month,day,hour))
            ds['ERA5'][var2]['longitude'] = ds['ERA5'][var2]['longitude'] % 360
            ds['ERA5'][var2] = ds['ERA5'][var2].sortby(['longitude', 'latitude']).rename({'latitude': 'lat', 'longitude': 'lon'}).sel(lon=slice(min_lon1, max_lon1), lat=slice(min_lat1, max_lat1))
            
            if var1 in ['tp', 'e', 'cp', 'lsp', 'pev']:
                ds['ERA5'][var2] *= 24000 / 24
            elif var1 in ['msl']:
                ds['ERA5'][var2] /= 100
            elif var1 in ['sst', 't2m', 'd2m', 'skt']:
                ds['ERA5'][var2] -= zerok
            elif var1 in ['hcc', 'mcc', 'lcc', 'tcc']:
                ds['ERA5'][var2] *= 100
            elif var1 in ['z']:
                ds['ERA5'][var2] /= 9.80665
            elif var1 in ['mper']:
                ds['ERA5'][var2] *= seconds_per_d / 24
            
            if var1 in ['e', 'pev', 'mper']:
                ds['ERA5'][var2] *= (-1)
            
            if var2 in ['psl', 'uas', 'vas', 'prw', 'clwvi', 'clivi', 'sfcWind', 'tas', 'huss', 'hurs']:
                ds['BARRA-R2'][var2] = xr.open_dataset(f'/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/{var2}/latest/{var2}_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc')[var2].sel(time=pd.Timestamp(year,month,day,hour))
                ds['BARRA-C2'][var2] = xr.open_dataset(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/{var2}/latest/{var2}_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc')[var2].sel(time=pd.Timestamp(year,month,day,hour))
            elif var2 in ['cll', 'clm', 'clh', 'clt', 'pr', 'evspsbl', 'evspsblpot', 'hfls', 'hfss', 'rsut', 'rlut']:
                ds['BARRA-R2'][var2] = xr.open_dataset(f'/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/{var2}/latest/{var2}_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc')[var2].sel(time=pd.Timestamp(year,month,day,hour) + pd.Timedelta('30min'))
                ds['BARRA-C2'][var2] = xr.open_dataset(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/{var2}/latest/{var2}_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc')[var2].sel(time=pd.Timestamp(year,month,day,hour) + pd.Timedelta('30min'))
            
            if var2 in ['pr', 'evspsbl', 'evspsblpot']:
                ds['BARRA-R2'][var2] *= seconds_per_d / 24
                ds['BARRA-C2'][var2] *= seconds_per_d / 24
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
        
        if imode=='org':
            plt_colnames = dss
        elif imode=='diff':
            plt_colnames = [dss[0]] + [f'{ids1} - {ids2}' for ids1, ids2 in zip(dss[1:], dss[:-1])]
        
        fig, axs = plt.subplots(
            nrow, ncol, figsize=np.array([pwidth*ncol, pheight*nrow+2.1])/2.54,
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
        
        if 'psl' in vars:
            psl_intervals = np.arange(800, 1200+1e-4, 4)
            psl_labels = np.arange(800, 1200+1e-4, 8)
            nsizes = {'ERA5': 300, 'BARRA-R2': 600, 'BARRA-C2': 1200}
            for jcol, ids in enumerate(dss):
                # ids = 'ERA5'
                # print(f'#---- {jcol} {ids}')
                plt_pres = axs[jcol].contour(
                    ds[ids]['psl'].lon,ds[ids]['psl'].lat,ds[ids]['psl'],
                    levels=psl_intervals, colors='tab:orange',
                    linewidths=0.15,transform=ccrs.PlateCarree())
                if jcol==0:
                    ax_clabel = axs[jcol].clabel(
                        plt_pres, inline=1, colors='tab:orange', fmt='%d',
                        levels=psl_labels, inline_spacing=10, fontsize=8)
                    plot_maxmin_points(
                        ds[ids]['psl'].lon, ds[ids]['psl'].lat, ds[ids]['psl'],
                        axs[jcol], 'max', nsizes[ids], symbol='H',
                        color='tab:orange', transform=ccrs.PlateCarree())
                    plot_maxmin_points(
                        ds[ids]['psl'].lon, ds[ids]['psl'].lat, ds[ids]['psl'],
                        axs[jcol], 'min', nsizes[ids], symbol='L',
                        color='r', transform=ccrs.PlateCarree())
            
            cax = fig.add_axes([0, fm_bottom - 0.02, 1/3, 0.05])
            cax.axis('off')
            cax.legend(
                [plt_pres.legend_elements()[0][0]],
                [r'Mean sea level pressure [$hPa$]'],
                loc='upper center', frameon=False, handlelength=1)
        
        if all(ivar in vars for ivar in ['vas', 'uas']):
            iarrows = {'ERA5': 8, 'BARRA-R2': 20, 'BARRA-C2': 50}
            for jcol, ids in enumerate(dss):
                # ids = 'ERA5'
                # print(f'#---- {jcol} {ids}')
                plt_quiver = axs[jcol].quiver(
                    ds[ids]['uas'].lon[::iarrows[ids]],
                    ds[ids]['uas'].lat[::iarrows[ids]],
                    ds[ids]['uas'][::iarrows[ids], ::iarrows[ids]].values,
                    ds[ids]['vas'][::iarrows[ids], ::iarrows[ids]].values,
                    color='gray', units='height', scale=500,
                    width=0.002, headwidth=3, headlength=5, alpha=1,
                    transform=ccrs.PlateCarree(), zorder=2)
            axs[-1].quiverkey(
                plt_quiver, X=0.15, Y=-0.08, U=10, coordinates='axes',
                label=r'10 $m$ wind [$10\;m\;s^{-1}$]', labelpos='E')
        
        # plot var
        if (len(set(vars) - set(['psl', 'uas', 'vas']))==1):
            var2 = list(set(vars) - set(['psl', 'uas', 'vas']))[0]
            var1 = cmip6_era5_var[var2]
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
            
            if imode=='org':
                for jcol, ids in enumerate(dss):
                    # ids = 'ERA5'
                    # print(f'#---- {jcol} {ids}')
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
                cbar.ax.set_xlabel(era5_varlabels[var1].replace('day^{-1}', 'hour^{-1}'), fontsize=9, labelpad=1)
                cbar.ax.tick_params(labelsize=9, pad=1)
            elif imode=='diff':
                plt_mesh = axs[0].pcolormesh(
                    ds[dss[0]][var2].lon, ds[dss[0]][var2].lat, ds[dss[0]][var2],
                    norm=pltnorm, cmap=pltcmp,
                    transform=ccrs.PlateCarree(), zorder=1)
                for jcol, ids1, ids2 in zip(range(1, len(dss)), dss[1:], dss[:-1]):
                    # jcol=1; ids1='BARRA-R2'; ids2='ERA5'
                    # print(f'#-------- {jcol} {ids1} {ids2}')
                    if not f'{ids1} - {ids2}' in regridder.keys():
                        regridder[f'{ids1} - {ids2}'] = xe.Regridder(
                            ds[ids1][var2],
                            ds[ids2][var2].sel(lon=slice(ds[ids1][var2].lon[0], ds[ids1][var2].lon[-1]), lat=slice(ds[ids1][var2].lat[0], ds[ids1][var2].lat[-1])),
                            method='bilinear')
                    plt_data = regridder[f'{ids1} - {ids2}'](ds[ids1][var2]) - ds[ids2][var2].sel(lon=slice(ds[ids1][var2].lon[0], ds[ids1][var2].lon[-1]), lat=slice(ds[ids1][var2].lat[0], ds[ids1][var2].lat[-1]))
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
        
        time2 = time.perf_counter()
        print(f'Execution time: {time2 - time1:.1f} s')



'''
Instanteneous or time mean?

var2        var1        BARRA       ERA5
psl         msl         i           i
uas/vas     u10/v10     i           i
prw         tcwv        i           i
cll/m/h/t   l/m/h/tcc   m           i
clwvi       tclw        i           i
clivi       tciw        i           i
pr          tp          m           m
evspsbl     e           m           m
evspsblpot  pev         m           m
hfls        mslhf       m           m
hfss        msshf       m           m
sfcWind     si10        i           i
tas         t2m         i           i
huss        q2m         i           i
hurs        rh2m        i           i
rsut        mtuwswrf    m           m
rlut        mtnlwrf     m           m



for var2 in ['psl', 'uas', 'vas', 'prw', 'cll', 'clm', 'clh', 'clt', 'clwvi', 'clivi', 'pr', 'evspsbl', 'evspsblpot', 'hfls', 'hfss', 'sfcWind', 'tas', 'huss', 'hurs', 'rsut', 'rlut']:
    # var2 = 'psl'
    var1 = cmip6_era5_var[var2]
    print(f'#-------- {var2} {var1}')

# check BARRA var instanteneous or mean?
for var2 in ['psl', 'uas', 'vas', 'prw', 'cll', 'clm', 'clh', 'clt', 'clwvi', 'clivi', 'pr', 'evspsbl', 'evspsblpot', 'hfls', 'hfss', 'sfcWind', 'tas', 'huss', 'hurs', 'rsut', 'rlut']:
    print(f'#-------------------------------- {var2}')
    print(xr.open_dataset(f'/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/{var2}/latest/{var2}_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc')[var2])
    # print(xr.open_dataset(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/{var2}/latest/{var2}_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc')[var2])

https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation

ds = xr.open_dataset('/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/mon/cll/latest/cll_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1_mon_202109-202109.nc')

# year, month, day, hour = 2020, 6, 30, 23
print(f'{year}-{month}-{day} {hour}')
print(f'{year1}-{month1}-{day1} {hour1}')

'''
# endregion


# region plot am/sea/mon data



# endregion


# region plot hourly cross section

dss = ['ERA5', 'BARRA-R2', 'BARRA-C2']
year, month, day, hour = 2020, 6, 1, 3

min_lon1, max_lon1, min_lat1, max_lat1 = 80, 220, -70, 20
min_lon, max_lon, min_lat, max_lat = 110.58, 157.34, -43.69, -7.01
clon = 150

pwidth  = 6.6
pheight = 5

nrow = 1
ncol = len(dss)
fm_bottom = 2.2/(pheight*nrow+2.7)
fm_top = 1 - 0.5/(pheight*nrow+2.7)


for var2 in ['hus', 'ta', 'ua', 'va', 'wap']:
    # var2 = 'hus'
    # ['hus', 'ta', 'ua', 'va', 'wap', 'zg']
    var1 = cmip6_era5_var[var2]
    print(f'#-------------------------------- {var1} in ERA5 vs. {var2} in BARRA-R2/C2')
    
    ds = {}
    ds['ERA5'] = xr.open_dataset(f'/g/data/rt52/era5/pressure-levels/reanalysis/{var1}/{year}/{var1}_era5_oper_pl_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')[var1].sel(time=pd.Timestamp(year, month, day, hour), level=slice(200, 1000), latitude=slice(max_lat1, min_lat1)).sel(longitude=clon, method='nearest').rename({'latitude': 'lat', 'level': 'pressure'}).sortby('lat')
    if var1 in ['q']:
        ds['ERA5'] *= 1000
    elif var1 in ['t']:
        ds['ERA5'] -= zerok
    elif var1 in ['z']:
        ds['ERA5'] /= 9.80665
    
    def std_func(ds_in, var=var2):
        ds = ds_in.expand_dims(dim='pressure', axis=1)
        varname = [varname for varname in ds.data_vars if varname.startswith(var)][0]
        ds = ds.rename({varname: var})
        ds = ds.chunk(chunks={'time': 1, 'pressure': 1, 'lat': len(ds.lat), 'lon': len(ds.lon)})
        ds = ds.astype('float32')
        if var == 'hus':
            ds = ds * 1000
        elif var == 'ta':
            ds = ds - zerok
        return(ds)
    
    if var2 == 'wap':
        barra_r2_hus = xr.open_mfdataset(sorted(glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/hus[0-9]*[!m]/latest/hus[0-9]*_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc')), parallel=True, preprocess=lambda ds: std_func(ds, var='hus'))['hus'].sel(time=pd.Timestamp(year, month, day, hour), pressure=slice(200, 1000)).sel(lon=clon, method='nearest')
        barra_r2_wa = xr.open_mfdataset(sorted(glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/wa[0-9]*[!m]/latest/wa[0-9]*_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc')), parallel=True, preprocess=lambda ds: std_func(ds, var='wa'))['wa'].sel(time=pd.Timestamp(year, month, day, hour), pressure=slice(200, 1000)).sel(lon=clon, method='nearest')
        barra_r2_ta = xr.open_mfdataset(sorted(glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/ta[0-9]*[!m]/latest/ta[0-9]*_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc')), parallel=True, preprocess=lambda ds: std_func(ds, var='ta'))['ta'].sel(time=pd.Timestamp(year, month, day, hour), pressure=slice(200, 1000)).sel(lon=clon, method='nearest')
        barra_r2_mixr = mixing_ratio_from_specific_humidity(barra_r2_hus.sel(pressure=barra_r2_wa.pressure) * units('g/kg')).compute()
        ds['BARRA-R2'] = vertical_velocity_pressure(
            barra_r2_wa * units('m/s'),
            barra_r2_wa.pressure * units.hPa,
            barra_r2_ta.sel(pressure=barra_r2_wa.pressure) * units.degC,
            barra_r2_mixr).compute()
        del barra_r2_hus, barra_r2_wa, barra_r2_ta, barra_r2_mixr
    else:
        ds['BARRA-R2'] = xr.open_mfdataset(sorted(glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/{var2}[0-9]*[!m]/latest/{var2}[0-9]*_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc')), parallel=True, preprocess=std_func)[var2].sel(time=pd.Timestamp(year, month, day, hour), pressure=slice(200, 1000)).sel(lon=clon, method='nearest')
    
    ds['BARRA-C2'] = xr.open_mfdataset(sorted(glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/{var2}[0-9]*[!m]/latest/{var2}[0-9]*_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc')), parallel=True, preprocess=std_func)[var2].sel(time=pd.Timestamp(year, month, day, hour), pressure=slice(200, 1000)).sel(lon=clon, method='nearest')
    
    if var1 == 'q':
        pltlevel = np.array([0, 0.1, 0.2, 0.5, 1, 2, 4, 8, 12, 16, 20])
        pltticks = np.array([0, 0.1, 0.2, 0.5, 1, 2, 4, 8, 12, 16, 20])
        pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
        pltcmp = plt.get_cmap('viridis_r', len(pltlevel)-1)
        extend = 'max'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-100, cm_max=100, cm_interval1=20, cm_interval2=20, cmap='BrBG_r')
    elif var1 == 't':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-48, cm_max=32, cm_interval1=4, cm_interval2=8, cmap='PuOr', asymmetric=True)
        extend = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-3, cm_max=3, cm_interval1=0.5, cm_interval2=1, cmap='BrBG', asymmetric=True)
    elif var1 == 'w':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-2, cm_max=2, cm_interval1=0.2, cm_interval2=0.4, cmap='PuOr')
        extend = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-2, cm_max=2, cm_interval1=0.2, cm_interval2=0.4, cmap='BrBG')
    elif var1 == 'u':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-20, cm_max=50, cm_interval1=2.5, cm_interval2=10, cmap='PuOr', asymmetric=True)
        extend = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-10, cm_max=10, cm_interval1=1, cm_interval2=2, cmap='BrBG')
    elif var1 == 'v':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=-40, cm_max=50, cm_interval1=2.5, cm_interval2=10, cmap='PuOr', asymmetric=True)
        extend = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-10, cm_max=10, cm_interval1=1, cm_interval2=2, cmap='BrBG')
    elif var1 == 'z':
        pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            cm_min=0, cm_max=12000, cm_interval1=500, cm_interval2=2000, cmap='viridis_r')
        extend = 'max'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-20, cm_max=20, cm_interval1=2, cm_interval2=4, cmap='BrBG')
    
    extend2 = 'both'
    
    for imode in ['org', 'diff']:
        # imode = 'org'
        # ['org', 'diff']
        print(f'#---------------- {imode}')
        time1 = time.perf_counter()
        
        opng = f'figures/4_um/4.0_barra/4.0.5_case_studies/4.0.5.2_{year}-{month}-{day}-{hour} {var2} in {', '.join(dss)} {imode} {clon} {min_lat1}_{max_lat1}.png'
        
        if imode=='org':
            plt_colnames = dss
        elif imode=='diff':
            plt_colnames = [dss[0]] + [f'{ids1} - {ids2}' for ids1, ids2 in zip(dss[1:], dss[:-1])]
        
        fig, axs = plt.subplots(nrow, ncol, figsize=np.array([pwidth*ncol, pheight*nrow+2.7])/2.54, sharey=True, gridspec_kw={'hspace':0.01, 'wspace':0.05})
        
        for jcol in range(ncol):
            axs[jcol].invert_yaxis()
            axs[jcol].set_ylim(1000, 200)
            axs[jcol].set_yticks(np.arange(1000, 200 - 1e-4, -200))
            
            axs[jcol].set_xticks(np.arange(-90, 90+1e-4, 30))
            axs[jcol].xaxis.set_minor_locator(ticker.AutoMinorLocator(3))
            axs[jcol].set_xlim(min_lat1, max_lat1)
            axs[jcol].xaxis.set_major_formatter(LatitudeFormatter(degree_symbol='° '))
            
            axs[jcol].axvline(min_lat, c='red', lw=0.5)
            axs[jcol].axvline(max_lat, c='red', lw=0.5)
            axs[jcol].axvline(ds['BARRA-R2'].lat[0], c='red', lw=0.5, ls='--')
            axs[jcol].axvline(ds['BARRA-R2'].lat[-1], c='red', lw=0.5, ls='--')
            axs[jcol].axvline(ds['BARRA-C2'].lat[0], c='red', lw=0.5, ls=':')
            axs[jcol].axvline(ds['BARRA-C2'].lat[-1], c='red', lw=0.5, ls=':')
            
            axs[jcol].grid(True, which='both', lw=0.5, c='gray', alpha=0.5, linestyle='--')
            
            axs[jcol].text(0, 1.02, f'({string.ascii_lowercase[jcol]}) {plt_colnames[jcol]}', ha='left', va='bottom', transform=axs[jcol].transAxes)
        
        if imode == 'org':
            for jcol, ids in enumerate(dss):
                # print(f'#---- {jcol} {ids}')
                plt_mesh = axs[jcol].pcolormesh(
                    ds[ids].lat, ds[ids].pressure, ds[ids],
                    norm=pltnorm, cmap=pltcmp, zorder=1)
            cbar = fig.colorbar(
                plt_mesh, #cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), #
                format=remove_trailing_zero_pos,
                orientation="horizontal", ticks=pltticks, extend=extend,
                cax=fig.add_axes([0.25, fm_bottom-0.13, 0.5, 0.04]))
            cbar.ax.set_xlabel(f'{era5_varlabels[var1]}')
        elif imode == 'diff':
            plt_mesh = axs[0].pcolormesh(
                ds[dss[0]].lat, ds[dss[0]].pressure, ds[dss[0]],
                norm=pltnorm, cmap=pltcmp, zorder=1)
            for jcol, ids1, ids2 in zip(range(1, len(dss)), dss[1:], dss[:-1]):
                # print(f'#-------- {jcol} {ids1} {ids2}')
                plevels = np.intersect1d(ds[ids1].pressure.values, ds[ids2].pressure.values)
                if var2 != 'hus':
                    plt_data = ds[ids1].sel(pressure=plevels).interp(lat=ds[ids2].sel(lat=slice(ds[ids1].lat[0], ds[ids1].lat[-1])).lat) - ds[ids2].sel(pressure=plevels, lat=slice(ds[ids1].lat[0], ds[ids1].lat[-1])).values
                elif var2 == 'hus':
                    plt_data = (ds[ids1].sel(pressure=plevels).interp(lat=ds[ids2].sel(lat=slice(ds[ids1].lat[0], ds[ids1].lat[-1])).lat) - ds[ids2].sel(pressure=plevels, lat=slice(ds[ids1].lat[0], ds[ids1].lat[-1]))) / ds[ids2].sel(pressure=plevels, lat=slice(ds[ids1].lat[0], ds[ids1].lat[-1])) * 100
                plt_mesh2 = axs[jcol].pcolormesh(
                    plt_data.lat, plt_data.pressure, plt_data,
                    norm=pltnorm2, cmap=pltcmp2, zorder=1)
            
            cbar = fig.colorbar(
                plt_mesh, #cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), #
                format=remove_trailing_zero_pos,
                orientation="horizontal", ticks=pltticks, extend=extend,
                cax=fig.add_axes([0.05, fm_bottom-0.13, 0.4, 0.04]))
            cbar.ax.set_xlabel(f'{era5_varlabels[var1]}')
            cbar2 = fig.colorbar(
                plt_mesh2, #cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), #
                format=remove_trailing_zero_pos,
                orientation="horizontal", ticks=pltticks2, extend=extend2,
                cax=fig.add_axes([0.55, fm_bottom-0.13, 0.4, 0.04]))
            cbar2.ax.set_xlabel(f'Difference in {era5_varlabels[var1].replace(r'[$g \; kg^{-1}$]', r'[$\%$]')}')
            
            del plt_data
        
        axs[0].set_ylabel(r'Pressure [$hPa$]')
        fig.subplots_adjust(left=0.08, right=0.99, bottom=fm_bottom, top=fm_top)
        fig.savefig(opng)
        
        time2 = time.perf_counter()
        print(f'Execution time: {time2 - time1:.1f} s')
    
    del ds






'''
Instanteneous or time mean?

var2        var1        BARRA       ERA5
hus         q                       i
ta          t                       i
ua          u                       i
va          v                       i
wap         w                       i
zg          z                       i





'''
# endregion


# region animate hourly cross section
# Memory Used: 31.73GB; Walltime Used: 09:57:15

imode = 'diff' #'org' #
var2 = 'wap' #['hus', 'ta', 'ua', 'va', 'wap']
var1 = cmip6_era5_var[var2]

year, month = 2020, 6
start_day = pd.Timestamp(year, month, 1, 0, 0)
end_day = start_day + pd.Timedelta(days=calendar.monthrange(year, month)[1])
time_series = pd.date_range(start=start_day, end=end_day, freq='1h')[:-1]

dss = ['ERA5', 'BARRA-R2', 'BARRA-C2']
min_lon1, max_lon1, min_lat1, max_lat1 = 80, 220, -70, 20
min_lon, max_lon, min_lat, max_lat = 110.58, 157.34, -43.69, -7.01
clon = 150
pwidth  = 6.6
pheight = 5
nrow = 1
ncol = len(dss)
fm_bottom = 2.2/(pheight*nrow+2.7)
fm_top = 1 - 0.5/(pheight*nrow+2.7)

if var1 == 'q':
    pltlevel = np.array([0, 0.1, 0.2, 0.5, 1, 2, 4, 8, 12, 16, 20])
    pltticks = np.array([0, 0.1, 0.2, 0.5, 1, 2, 4, 8, 12, 16, 20])
    pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
    pltcmp = plt.get_cmap('viridis_r', len(pltlevel)-1)
    extend = 'max'
    pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
        cm_min=-100, cm_max=100, cm_interval1=20, cm_interval2=20, cmap='BrBG_r')
elif var1 == 't':
    pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
        cm_min=-48, cm_max=32, cm_interval1=4, cm_interval2=8, cmap='PuOr', asymmetric=True)
    extend = 'both'
    pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
        cm_min=-3, cm_max=3, cm_interval1=0.5, cm_interval2=1, cmap='BrBG', asymmetric=True)
elif var1 == 'w':
    pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
        cm_min=-2, cm_max=2, cm_interval1=0.2, cm_interval2=0.4, cmap='PuOr')
    extend = 'both'
    pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
        cm_min=-2, cm_max=2, cm_interval1=0.2, cm_interval2=0.4, cmap='BrBG')
elif var1 == 'u':
    pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
        cm_min=-20, cm_max=50, cm_interval1=2.5, cm_interval2=10, cmap='PuOr', asymmetric=True)
    extend = 'both'
    pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
        cm_min=-10, cm_max=10, cm_interval1=1, cm_interval2=2, cmap='BrBG')
elif var1 == 'v':
    pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
        cm_min=-40, cm_max=50, cm_interval1=2.5, cm_interval2=10, cmap='PuOr', asymmetric=True)
    extend = 'both'
    pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
        cm_min=-10, cm_max=10, cm_interval1=1, cm_interval2=2, cmap='BrBG')
elif var1 == 'z':
    pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
        cm_min=0, cm_max=12000, cm_interval1=500, cm_interval2=2000, cmap='viridis_r')
    extend = 'max'
    pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
        cm_min=-20, cm_max=20, cm_interval1=2, cm_interval2=4, cmap='BrBG')

extend2 = 'both'

omp4 = f'figures/4_um/4.0_barra/4.0.5_case_studies/4.0.5.2_{year}-{month} {var2} in {', '.join(dss)} {imode} {clon} {min_lat1}_{max_lat1}.mp4'
if imode=='org':
    plt_colnames = dss
elif imode=='diff':
    plt_colnames = [dss[0]] + [f'{ids1} - {ids2}' for ids1, ids2 in zip(dss[1:], dss[:-1])]

ds = {}
ds['ERA5'] = xr.open_dataset(f'/g/data/rt52/era5/pressure-levels/reanalysis/{var1}/{year}/{var1}_era5_oper_pl_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')[var1].sel(level=slice(200, 1000), latitude=slice(max_lat1, min_lat1)).sel(longitude=clon, method='nearest').rename({'latitude': 'lat', 'level': 'pressure'}).sortby('lat')
if var1 in ['q']:
    ds['ERA5'] *= 1000
elif var1 in ['t']:
    ds['ERA5'] -= zerok
elif var1 in ['z']:
    ds['ERA5'] /= 9.80665

def std_func(ds_in, var=var2):
    ds = ds_in.expand_dims(dim='pressure', axis=1)
    varname = [varname for varname in ds.data_vars if varname.startswith(var)][0]
    ds = ds.rename({varname: var})
    ds = ds.chunk(chunks={'time': 1, 'pressure': 1, 'lat': len(ds.lat), 'lon': len(ds.lon)})
    ds = ds.astype('float32')
    if var == 'hus':
        ds = ds * 1000
    elif var == 'ta':
        ds = ds - zerok
    return(ds)

if var2 == 'wap':
    barra_r2_hus = xr.open_mfdataset(sorted(glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/hus[0-9]*[!m]/latest/hus[0-9]*_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc')), parallel=True, preprocess=lambda ds: std_func(ds, var='hus'))['hus'].sel(pressure=slice(200, 1000)).sel(lon=clon, method='nearest')
    barra_r2_wa = xr.open_mfdataset(sorted(glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/wa[0-9]*[!m]/latest/wa[0-9]*_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc')), parallel=True, preprocess=lambda ds: std_func(ds, var='wa'))['wa'].sel(pressure=slice(200, 1000)).sel(lon=clon, method='nearest')
    barra_r2_ta = xr.open_mfdataset(sorted(glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/ta[0-9]*[!m]/latest/ta[0-9]*_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc')), parallel=True, preprocess=lambda ds: std_func(ds, var='ta'))['ta'].sel(pressure=slice(200, 1000)).sel(lon=clon, method='nearest')
    barra_r2_mixr = mixing_ratio_from_specific_humidity(barra_r2_hus.sel(pressure=barra_r2_wa.pressure) * units('g/kg'))
    ds['BARRA-R2'] = vertical_velocity_pressure(
            barra_r2_wa * units('m/s'),
            barra_r2_wa.pressure * units.hPa,
            barra_r2_ta.sel(pressure=barra_r2_wa.pressure) * units.degC,
            barra_r2_mixr).metpy.dequantify()
else:
    ds['BARRA-R2'] = xr.open_mfdataset(sorted(glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/{var2}[0-9]*[!m]/latest/{var2}[0-9]*_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc')), parallel=True, preprocess=std_func)[var2].sel(pressure=slice(200, 1000)).sel(lon=clon, method='nearest')

ds['BARRA-C2'] = xr.open_mfdataset(sorted(glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/{var2}[0-9]*[!m]/latest/{var2}[0-9]*_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc')), parallel=True, preprocess=std_func)[var2].sel(pressure=slice(200, 1000)).sel(lon=clon, method='nearest')


fig, axs = plt.subplots(nrow, ncol, figsize=np.array([pwidth*ncol, pheight*nrow+2.7])/2.54, sharey=True, gridspec_kw={'hspace':0.01, 'wspace':0.05})

for jcol in range(ncol):
    axs[jcol].invert_yaxis()
    axs[jcol].set_ylim(1000, 200)
    axs[jcol].set_yticks(np.arange(1000, 200 - 1e-4, -200))
    
    axs[jcol].set_xticks(np.arange(-90, 90+1e-4, 30))
    axs[jcol].xaxis.set_minor_locator(ticker.AutoMinorLocator(3))
    axs[jcol].set_xlim(min_lat1, max_lat1)
    axs[jcol].xaxis.set_major_formatter(LatitudeFormatter(degree_symbol='° '))
    
    axs[jcol].axvline(min_lat, c='red', lw=0.5)
    axs[jcol].axvline(max_lat, c='red', lw=0.5)
    axs[jcol].axvline(ds['BARRA-R2'].lat[0], c='red', lw=0.5, ls='--')
    axs[jcol].axvline(ds['BARRA-R2'].lat[-1], c='red', lw=0.5, ls='--')
    axs[jcol].axvline(ds['BARRA-C2'].lat[0], c='red', lw=0.5, ls=':')
    axs[jcol].axvline(ds['BARRA-C2'].lat[-1], c='red', lw=0.5, ls=':')
    
    axs[jcol].grid(True, which='both', lw=0.5, c='gray', alpha=0.5, linestyle='--')
    
    axs[jcol].text(0, 1.02, f'({string.ascii_lowercase[jcol]}) {plt_colnames[jcol]}', ha='left', va='bottom', transform=axs[jcol].transAxes)


plt_objs = []
def update_frames(itime):
    # itime = 0
    time1 = time.perf_counter()
    global plt_objs
    for plt_obj in plt_objs:
        try:
            plt_obj.remove()
        except ValueError:
            pass
    plt_objs = []
    
    day = time_series[itime].day
    hour = time_series[itime].hour
    print(f'#-------------------------------- {day:02d} {hour:02d}')
    
    plt_mesh = []
    if imode == 'org':
        for jcol, ids in enumerate(dss):
            # print(f'#---- {jcol} {ids}')
            plt_mesh.append(axs[jcol].pcolormesh(
                ds[ids].lat, ds[ids].pressure,
                ds[ids].sel(time=pd.Timestamp(year, month, day, hour)),
                norm=pltnorm, cmap=pltcmp, zorder=1))
        plt_text = fig.text(0.05, 0.02, f'{year}-{month:02d}-{day:02d} {hour:02d}:00 UTC', ha='left', va='bottom')
        cbar = fig.colorbar(
            plt_mesh[0], #cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), #
            format=remove_trailing_zero_pos,
            orientation="horizontal", ticks=pltticks, extend=extend,
            cax=fig.add_axes([0.25, fm_bottom-0.13, 0.5, 0.04]))
        cbar.ax.set_xlabel(f'{era5_varlabels[var1]}')
    elif imode == 'diff':
        plt_mesh.append(axs[0].pcolormesh(
            ds[dss[0]].lat, ds[dss[0]].pressure,
            ds[dss[0]].sel(time=pd.Timestamp(year, month, day, hour)),
            norm=pltnorm, cmap=pltcmp, zorder=1))
        for jcol, ids1, ids2 in zip(range(1, len(dss)), dss[1:], dss[:-1]):
            # jcol=1; ids1=dss[1]; ids2=dss[0]
            # print(f'#-------- {jcol} {ids1} {ids2}')
            plevels = np.intersect1d(ds[ids1].pressure.values, ds[ids2].pressure.values)
            if var2 != 'hus':
                plt_data = ds[ids1].sel(time=pd.Timestamp(year, month, day, hour)).sel(pressure=plevels).interp(lat=ds[ids2].sel(lat=slice(ds[ids1].lat[0], ds[ids1].lat[-1])).lat) - ds[ids2].sel(time=pd.Timestamp(year, month, day, hour)).sel(pressure=plevels, lat=slice(ds[ids1].lat[0], ds[ids1].lat[-1])).values
            elif var2 == 'hus':
                plt_data = (ds[ids1].sel(time=pd.Timestamp(year, month, day, hour)).sel(pressure=plevels).interp(lat=ds[ids2].sel(lat=slice(ds[ids1].lat[0], ds[ids1].lat[-1])).lat) - ds[ids2].sel(time=pd.Timestamp(year, month, day, hour)).sel(pressure=plevels, lat=slice(ds[ids1].lat[0], ds[ids1].lat[-1]))) / ds[ids2].sel(time=pd.Timestamp(year, month, day, hour)).sel(pressure=plevels, lat=slice(ds[ids1].lat[0], ds[ids1].lat[-1])) * 100
            plt_mesh.append(axs[jcol].pcolormesh(
                plt_data.lat, plt_data.pressure, plt_data,
                norm=pltnorm2, cmap=pltcmp2, zorder=1))
        plt_text = fig.text(0.5, 0.02, f'{year}-{month:02d}-{day:02d} {hour:02d}:00 UTC', ha='center', va='bottom')
        cbar = fig.colorbar(
            plt_mesh[0], #cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), #
            format=remove_trailing_zero_pos,
            orientation="horizontal", ticks=pltticks, extend=extend,
            cax=fig.add_axes([0.05, fm_bottom-0.13, 0.4, 0.04]))
        cbar.ax.set_xlabel(f'{era5_varlabels[var1]}')
        cbar2 = fig.colorbar(
            plt_mesh[1], #cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), #
            format=remove_trailing_zero_pos,
            orientation="horizontal", ticks=pltticks2, extend=extend2,
            cax=fig.add_axes([0.55, fm_bottom-0.13, 0.4, 0.04]))
        cbar2.ax.set_xlabel(f'Difference in {era5_varlabels[var1].replace(r'[$g \; kg^{-1}$]', r'[$\%$]')}')
    
    plt_objs += plt_mesh + [plt_text]
    time2 = time.perf_counter()
    print(f'Execution time: {time2 - time1:.1f} s')
    print(f'Memory usage: {np.round(process.memory_info().rss/2**30, 3)} GB')
    return(plt_objs)


axs[0].set_ylabel(r'Pressure [$hPa$]')
fig.subplots_adjust(left=0.08, right=0.99, bottom=fm_bottom, top=fm_top)
ani = animation.FuncAnimation(
    fig, update_frames, frames=len(time_series), interval=500, blit=False)
if os.path.exists(omp4): os.remove(omp4)
ani.save(omp4,progress_callback=lambda iframe,n:print(f'Frame {iframe}/{n-1}'))






# endregion


# region plot am/sea/mon hourly cross section

# endregion




