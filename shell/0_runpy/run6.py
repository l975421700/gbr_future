

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


# region animate hourly cross section

imode = 'org' #'org' #
var2 = 'hus' #['hus', 'ta', 'ua', 'va', 'wap']
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
            barra_r2_mixr)
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


