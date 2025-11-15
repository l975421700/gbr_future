

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
    )


from component_plot import (
    plt_mesh_pars,
)


# endregion


# region plot Himawari clp

# option
year, month, day, hour, minute = 2020, 6, 2, 3, 0
products = ['cloud']
categories = ['cmic']
vars = ['cmic_iwp', 'cmic_lwp', 'cwp']
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
            print(f'#-------- {ivar}')
            
            if ivar in ['cmic_iwp', 'cmic_lwp']:
                # ivar = 'cmic_lwp'
                plt_data = ds[ivar].attrs['scale_factor'] * (ds[ivar].rename(himawari_rename[ivar]) - ds[ivar].attrs['add_offset'])
            elif ivar=='cwp':
                # ivar = 'cwp'
                plt_data = ((ds['cmic_iwp'].attrs['scale_factor'] * (ds['cmic_iwp'] - ds['cmic_iwp'].attrs['add_offset'])).fillna(0) + (ds['cmic_lwp'].attrs['scale_factor'] * (ds['cmic_lwp'] - ds['cmic_lwp'].attrs['add_offset'])).fillna(0)).rename(ivar)
            
            if ivar in ['cmic_iwp', 'cmic_lwp', 'cwp']:
                plt_data *= 1000
                pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                    cm_min=0, cm_max=1600, cm_interval1=100, cm_interval2=200,
                    cmap='Purples_r',)
                extend = 'max'
            
            for iplt_region in plt_regions:
                print(f'#---- {iplt_region}')
                
                opng = f'figures/3_satellites/3.0_hamawari/3.0.2_clp/3.0.2.0 himawari {iproduct} {icategory} {plt_data.name} {iplt_region} {year}{month:02d}{day:02d}{hour:02d}{minute:02d}.png'
                cbar_label = f'{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d} UTC Himawari {era5_varlabels[cmip6_era5_var[plt_data.name]]}'
                
                if iplt_region == 'himawari':
                    fig, ax = plt.subplots(figsize=np.array([7, 7+1])/2.54, subplot_kw={'projection': transform})
                    fig.subplots_adjust(left=0.01, right=0.99, bottom=1/(7+1), top=0.99)
                    
                ax.pcolormesh(
                    plt_data.lon, plt_data.lat, plt_data,
                    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree())
                
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
                fig.savefig(opng)






'''
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

