

# qsub -I -q normal -P v46 -l walltime=6:00:00,ncpus=1,mem=96GB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22+gdata/qx55+gdata/gx60+gdata/py18+gdata/rv74+gdata/xp65


# region import packages

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
import glob
import rioxarray as rxr
import warnings
warnings.filterwarnings('ignore')

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=12)
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib import cm

# management
import os
import sys  # print(sys.path)
sys.path.append(os.getcwd() + '/code/gbr_future/module')
import psutil
process = psutil.Process()
# print(process.memory_info().rss / 2**30)
import warnings
warnings.filterwarnings('ignore')

# self defined
from namelist import (
    month_jan,
    monthini,
    seasons,
    seconds_per_d,
    zerok,
    era5_varlabels,
    cmip6_era5_var,
    )

from mapplot import (
    regional_plot,
    remove_trailing_zero_pos,
    )


from component_plot import (
    plt_mesh_pars,
)


# endregion


# region plot Himawari clp

# option
year, month, day, hour, minute = 2020, 6, 2, 3, 0
products = ['cloud']
categories = ['ctth']
# categories = ['ctth']
# ['ctth_alti', 'ctth_pres', 'ctth_tempe', 'ctth_effectiv']
# categories = ['cmic']
# ['cmic_cot', 'cmic_phase', 'cmic_reff', 'cmic_iwp', 'cmic_lwp', 'cwp']
vars = ['ctth_alti', 'ctth_pres', 'ctth_tempe', 'ctth_effectiv']
plt_regions = ['himawari']

# settings
himawari_bom = '/g/data/rv74/satellite-products/arc/der/himawari-ahi'
ancillary = xr.open_dataset('/g/data/ra22/satellite-products/arc/obs/himawari-ahi/fldk/latest/ancillary/00000000000000-P1S-ABOM_GEOM_SENSOR-PRJ_GEOS141_2000-HIMAWARI8-AHI.nc')
ancillary['lon'] = ancillary['lon'] % 360
himawari_rename = {
    'cmic_cot': 'COT',
    'cmic_iwp': 'clivi',
    'cmic_lwp': 'clwvi',
    'cmic_phase': 'clphase',
    'cmic_reff': 'Reff',
    'ctth_alti': 'CTH',
    'ctth_effectiv': 'clt',
    'ctth_pres': 'CTP',
    'ctth_tempe': 'CTT'}
extent = [-5499500., 5499500., -5499500., 5499500.]
transform = ccrs.Geostationary(central_longitude=140.7, satellite_height=35785863.0)

for iproduct in products: #os.listdir(himawari_bom): #
    print(f'#-------------------------------- {iproduct}')
    for icategory in categories: #os.listdir(f'{himawari_bom}/{iproduct}'): #
        print(f'#---------------- {icategory}')
        
        file = glob.glob(f'{himawari_bom}/{iproduct}/{icategory}/latest/{year}/{month:02d}/{day:02d}/*{hour:02d}{minute:02d}00Z.nc')[0]
        ds = rxr.open_rasterio(file, masked=True)[0]
        ds = ds.rio.write_crs(transform)
        ds = ds.rio.reproject('epsg:4326')
        ds = ds.squeeze().rename({'x': 'lon', 'y': 'lat'})
        ds['lon'] = ds['lon'] % 360
        ds = ds.sortby(['lon', 'lat'])
        ds = ds.sel(lon=slice(np.nanmin(ancillary['lon']), np.nanmax(ancillary['lon'])))
        
        for ivar in vars:
            # ivar = 'ctth_alti'
            # ['cmic_cot', 'cmic_phase', 'cmic_reff', 'cmic_iwp', 'cmic_lwp', 'cwp']
            # ['ctth_alti', 'ctth_pres', 'ctth_tempe', 'ctth_effectiv']
            print(f'#-------- {ivar}')
            
            if ivar == 'cmic_phase':
                plt_data = ds[ivar].rename(himawari_rename[ivar])
            elif ivar in himawari_rename.keys():
                plt_data = ds[ivar].attrs['scale_factor'] * ds[ivar].rename(himawari_rename[ivar]) + ds[ivar].attrs['add_offset']
            elif ivar=='cwp':
                plt_data = ((ds['cmic_iwp'].attrs['scale_factor'] * ds['cmic_iwp'] + ds['cmic_iwp'].attrs['add_offset']).fillna(0) + (ds['cmic_lwp'].attrs['scale_factor'] * ds['cmic_lwp'] + ds['cmic_lwp'].attrs['add_offset']).fillna(0)).rename(ivar)
            
            if ivar in ['cmic_iwp', 'cmic_lwp', 'cwp']:
                plt_data *= 1000
            elif ivar=='cmic_reff':
                plt_data *= 10**6
            elif ivar=='ctth_alti':
                plt_data /= 1000
            elif ivar=='ctth_pres':
                plt_data /= 100
            elif ivar=='ctth_effectiv':
                plt_data *= 100
            
            print(f'Min: {np.nanmin(plt_data)}')
            print(f'Mean: {np.nanmean(plt_data)}')
            print(f'Max: {np.nanmax(plt_data)}')
            
            if ivar in ['cmic_iwp', 'cmic_lwp', 'cwp']:
                pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                    cm_min=0, cm_max=800, cm_interval1=50, cm_interval2=100,
                    cmap='Purples_r',)
                extend = 'max'
            elif ivar=='cmic_cot':
                pltlevel = [0, 1.3, 3.6, 9.4, 23, 60, 379]
                pltticks = pltlevel
                pltnorm = BoundaryNorm(boundaries=pltlevel, ncolors=len(pltlevel)-1, clip=True)
                pltcmp = cm.get_cmap('Purples', len(pltlevel)-1)
                extend = 'neither'
            elif ivar=='cmic_phase':
                pltticks = [1, 2, 3, 4, 5]
                pltlabels = ['liquid', 'ice', 'mixed', 'cloud-free', 'undefined']
                colors = ['#1f77b4', '#00FFFF', 'tab:orange', '#ffffff', '#000000']
                pltcmp = ListedColormap(colors)
                pltnorm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], pltcmp.N)
                extend = 'neither'
            elif ivar=='cmic_reff':
                pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                    cm_min=0, cm_max=40, cm_interval1=2, cm_interval2=4,
                    cmap='Greens_r',)
                extend = 'max'
            elif ivar=='ctth_alti':
                pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                    cm_min=0, cm_max=18, cm_interval1=1, cm_interval2=2,
                    cmap='Blues_r',)
                extend = 'max'
            elif ivar=='ctth_pres':
                pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                    cm_min=100, cm_max=1000, cm_interval1=50, cm_interval2=100,
                    cmap='Blues',)
                extend = 'both'
            elif ivar=='ctth_tempe':
                pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                    cm_min=180, cm_max=310, cm_interval1=10, cm_interval2=20,
                    cmap='Blues',)
                extend = 'both'
            elif ivar=='ctth_effectiv':
                pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                    cm_min=0, cm_max=100, cm_interval1=5, cm_interval2=10,
                    cmap='Blues_r',)
                extend = 'neither'
            
            for iplt_region in plt_regions:
                print(f'#---- {iplt_region}')
                
                opng = f'figures/3_satellites/3.0_hamawari/3.0.2_clp/3.0.2.0 {iproduct} {icategory} {plt_data.name} {iplt_region} {year}{month:02d}{day:02d}{hour:02d}{minute:02d}.png'
                cbar_label = f'{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d} UTC Himawari {era5_varlabels[cmip6_era5_var[plt_data.name]]}'
                
                if iplt_region == 'himawari':
                    fig, ax = plt.subplots(figsize=np.array([7, 7+1])/2.54, subplot_kw={'projection': transform})
                    ax.imshow(np.ones((100,100,3)),extent=extent,transform=transform,zorder=0)
                    fig.subplots_adjust(left=0.01, right=0.99, bottom=1/(7+1), top=0.99)
                    
                    coastline = cfeature.NaturalEarthFeature(
                        'physical', 'coastline', '10m', edgecolor='k',
                        facecolor='none', lw=0.1)
                    ax.add_feature(coastline, zorder=2, alpha=0.75)
                    borders = cfeature.NaturalEarthFeature(
                        'cultural', 'admin_0_boundary_lines_land', '10m',
                        edgecolor='k', facecolor='none', lw=0.1)
                    ax.add_feature(borders, zorder=2, alpha=0.75)
                    gl = ax.gridlines(
                        crs=ccrs.PlateCarree(), lw=0.1, zorder=2, alpha=0.35,
                        color='k', linestyle='--',)
                    gl.xlocator = mticker.FixedLocator(np.arange(0, 360 + 1e-4, 10))
                    gl.ylocator = mticker.FixedLocator(np.arange(-90, 90 + 1e-4, 10))
                
                plt_mesh = ax.pcolormesh(
                    plt_data.lon, plt_data.lat, plt_data,
                    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree())
                
                cbar = fig.colorbar(
                    plt_mesh,
                    format=remove_trailing_zero_pos,
                    orientation="horizontal", ticks=pltticks, extend=extend,
                    cax=fig.add_axes([0.04, 0.8/(7+1), 0.92, 0.02]))
                cbar.ax.tick_params(labelsize=8, length=2, width=0.5, pad=1)
                cbar.ax.tick_params(which='minor', length=1, width=0.5)
                cbar.ax.set_xlabel(cbar_label, fontsize=8, labelpad=2)
                
                if ivar=='cmic_phase':
                    cbar.ax.set_xticklabels(pltlabels)
                
                fig.savefig(opng)




'''
#-------------------------------- alternative method: no reproject
year, month, day, hour, minute = 2020, 6, 2, 3, 0
iproduct = 'cloud'
icategory = 'cmic'
ivar = 'cwp'
iplt_region = 'himawari'

himawari_bom = '/g/data/rv74/satellite-products/arc/der/himawari-ahi'
ancillary = xr.open_dataset('/g/data/ra22/satellite-products/arc/obs/himawari-ahi/fldk/latest/ancillary/00000000000000-P1S-ABOM_GEOM_SENSOR-PRJ_GEOS141_2000-HIMAWARI8-AHI.nc')
ancillary['lon'] = ancillary['lon'] % 360
himawari_rename = {
    'cmic_cot': 'COT',
    'cmic_iwp': 'clivi',
    'cmic_lwp': 'clwvi',
    'cmic_phase': 'clphase',
    'cmic_reff': 'Reff',
    'ctth_alti': 'CTH',
    'ctth_effectiv': 'clt',
    'ctth_pres': 'CTP',
    'ctth_tempe': 'CTT'}
extent = [-5499500., 5499500., -5499500., 5499500.]
transform = ccrs.Geostationary(central_longitude=140.7, satellite_height=35785863.0)

file = glob.glob(f'{himawari_bom}/{iproduct}/{icategory}/latest/{year}/{month:02d}/{day:02d}/*{hour:02d}{minute:02d}00Z.nc')[0]
ds = xr.open_dataset(file)

# plt_data = ((ds['cmic_iwp'].attrs['scale_factor'] * ds['cmic_iwp'] + ds['cmic_iwp'].attrs['add_offset']).fillna(0) + (ds['cmic_lwp'].attrs['scale_factor'] * ds['cmic_lwp'] + ds['cmic_lwp'].attrs['add_offset']).fillna(0)).rename(ivar)
plt_data = (ds['cmic_iwp'].fillna(0) + ds['cmic_lwp'].fillna(0)).rename(ivar)
plt_data *= 1000
pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=800, cm_interval1=50, cm_interval2=100,
    cmap='Purples_r',)
extend = 'max'

opng = 'figures/0_gbr/test.png'
cbar_label = f'{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d} UTC Himawari {era5_varlabels[cmip6_era5_var[plt_data.name]]}'

fig, ax = plt.subplots(figsize=np.array([7, 7+1])/2.54, subplot_kw={'projection': transform})
ax.imshow(np.ones((100,100,3)),extent=extent,transform=transform,zorder=0)
fig.subplots_adjust(left=0.01, right=0.99, bottom=1/(7+1), top=0.99)

coastline = cfeature.NaturalEarthFeature(
    'physical', 'coastline', '10m', edgecolor='k',
    facecolor='none', lw=0.1)
ax.add_feature(coastline, zorder=2, alpha=0.75)
borders = cfeature.NaturalEarthFeature(
    'cultural', 'admin_0_boundary_lines_land', '10m',
    edgecolor='k', facecolor='none', lw=0.1)
ax.add_feature(borders, zorder=2, alpha=0.75)
gl = ax.gridlines(
    crs=ccrs.PlateCarree(), lw=0.1, zorder=2, alpha=0.35,
    color='k', linestyle='--',)
gl.xlocator = mticker.FixedLocator(np.arange(0, 360 + 1e-4, 10))
gl.ylocator = mticker.FixedLocator(np.arange(-90, 90 + 1e-4, 10))

plt_mesh = ax.pcolormesh(
    plt_data.nx, plt_data.ny, plt_data,
    norm=pltnorm, cmap=pltcmp, transform=transform)

cbar = fig.colorbar(
    plt_mesh,
    format=remove_trailing_zero_pos,
    orientation="horizontal", ticks=pltticks, extend=extend,
    cax=fig.add_axes([0.04, 0.8/(7+1), 0.92, 0.02]))
cbar.ax.tick_params(labelsize=8, length=2, width=0.5, pad=1)
cbar.ax.tick_params(which='minor', length=1, width=0.5)
cbar.ax.set_xlabel(cbar_label, fontsize=8, labelpad=2)

fig.savefig(opng)


#-------------------------------- original method

cwp = xr.open_dataset('/g/data/rv74/satellite-products/arc/der/himawari-ahi/cloud/cmic/latest/2020/06/02/S_NWC_CMIC_HIMA08_HIMA-N-NR_20200602T033000Z.nc')
ancil = xr.open_dataset('/g/data/ra22/satellite-products/arc/obs/himawari-ahi/fldk/latest/ancillary/00000000000000-P1S-ABOM_GEOM_SENSOR-PRJ_GEOS141_2000-HIMAWARI8-AHI.nc')

trim = 1000
np.sum(np.isnan(ancil['lon'].squeeze().values[trim:(-1 * trim), trim:(-1 * trim)]))

# 110.58, 157.34, -43.69, -7.01
fig, ax = regional_plot(extent=[137.12, 157.3, -28.76, -7.05], figsize = np.array([4.4, 6.6])/2.54)
pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=800, cm_interval1=50, cm_interval2=100,
    cmap='Purples_r') # 'Purples_r'

ax.pcolormesh(
    ancil['lon'].squeeze().values[trim:(-1 * trim), trim:(-1 * trim)],
    ancil['lat'].squeeze().values[trim:(-1 * trim), trim:(-1 * trim)],
    cwp['cmic_iwp'][trim:(-1 * trim), trim:(-1 * trim)].fillna(0) * 1000 + cwp['cmic_lwp'][trim:(-1 * trim), trim:(-1 * trim)].fillna(0) * 1000,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(), zorder=1)

fig.savefig('figures/0_gbr/test.png')



'''
# endregion

