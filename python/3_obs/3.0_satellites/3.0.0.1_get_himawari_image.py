

# qsub -I -q normal -l walltime=1:00:00,ncpus=1,mem=40GB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+gdata/zv2+gdata/ra22+gdata/gx60


# region import packages

# data analysis
import numpy as np
import xarray as xr
from netCDF4 import Dataset
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
from skimage.measure import block_reduce
from scipy import stats
import geopandas as gpd

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rcParams['figure.dpi'] = 600
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})
mpl.rc('font', family='Times New Roman', size=10)
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import matplotlib.animation as animation
from matplotlib.patches import Rectangle

# management
import os
import sys  # print(sys.path)
sys.path.append(os.getcwd() + '/code/gbr_future/module')
import glob
import time
from pathlib import Path
import argparse
import calendar
import pandas as pd

# self defined
from mapplot import (
    regional_plot,
    )

# endregion


# region plot himawari true color or night microphysics

start_time = time.perf_counter()

year, month, day, hour, minute = 2020, 4, 1, 14, 0
ioption='true_color'
# ioption='night_microphysics'

dfolder = Path('/g/data/ra22/satellite-products/arc/obs/himawari-ahi/fldk/latest')
# dfolder = Path('/g/data/ra22/satellite-products/nrt/obs/himawari-ahi/fldk/latest')
opng = Path(f'figures/3_satellites/3.0_hamawari/3.0.0_image/3.0.0.0 himawari {ioption} {year}{month:02d}{day:02d}{hour:02d}{minute:02d}.png')
extent = [-5499500., 5499500., -5499500., 5499500.]
transform = ccrs.Geostationary(central_longitude=140.7, satellite_height=35785863.0)

band_map = {
    'true_color': ['B03', 'B02', 'B01'],
    'night_microphysics': ['B07', 'B13', 'B15']
}
bands = band_map[ioption]
plt_text = f'{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d} UTC\n{ioption.replace('_', ' ').title()} RGB Himawari 8/9'

channels = {}
for iband in bands:
    # iband='B07'
    print(f'#-------------------------------- {iband}')
    
    ifile = sorted(Path(dfolder/f'{year}/{month:02d}/{day:02d}/{hour:02d}{minute:02d}').glob(f'*OBS_{iband}*'))
    if not ifile:
        print('Warning: No file found')
        continue
    
    channels[iband] = Dataset(ifile[-1], 'r')
    var_name = next(var for var in channels[iband].variables if var.startswith("channel_00"))
    # print(f'#---------------- {var_name}')
    channels[iband] = np.squeeze(channels[iband].variables[var_name][:])
    
    if iband in ['B03', 'B02', 'B01']:
        channels[iband] = channels[iband].filled(0)
    
    if iband == 'B03':
        channels[iband] = block_reduce(channels[iband], block_size=(2, 2), func=np.mean)

if ioption=='night_microphysics':
    # stats.describe(channels['B13'], axis=None)
    channels['B15-B13'] = channels['B15']-channels['B13']
    channels['B13-B07'] = channels['B13']-channels['B07']
    
    channels['B15-B13'] = (channels['B15-B13'] + 4) / (2 + 4)
    channels['B13-B07'] = (channels['B13-B07'] + 0) / (10 + 0)
    channels['B13'] = (channels['B13'] - 243) / (293 - 243)
    
    bands=['B15-B13', 'B13-B07', 'B13']
    for iband in bands: channels[iband] = channels[iband].filled(0)
    gamma=1
else:
    gamma=2.2


rgb = np.zeros(channels[bands[0]].shape + (3,), dtype=np.float32)
for idx, iband in enumerate(bands):
    print(f'{idx} {iband}')
    rgb[:, :, idx] = channels[iband]
rgb = np.clip(rgb, 0, 1, out=rgb)
rgb **= (1 / gamma)

end_time = time.perf_counter()
print(f"Execution time: {end_time - start_time:.6f} seconds")


start_time = time.perf_counter()
fig, ax = plt.subplots(figsize=np.array([7, 7+1])/2.54, subplot_kw={'projection': transform})

ax.imshow(rgb, extent=extent, transform=transform,
          interpolation='none', origin='upper', resample=False)

coastline = cfeature.NaturalEarthFeature(
    'physical', 'coastline', '10m', edgecolor='yellow',
    facecolor='none', lw=0.1)
ax.add_feature(coastline, zorder=2, alpha=0.75)
borders = cfeature.NaturalEarthFeature(
    'cultural', 'admin_0_boundary_lines_land', '10m',
    edgecolor='yellow', facecolor='none', lw=0.1)
ax.add_feature(borders, zorder=2, alpha=0.75)
gl = ax.gridlines(
    crs=ccrs.PlateCarree(), lw=0.1, zorder=2, alpha=0.35,
    color='yellow', linestyle='--',)
gl.xlocator = mticker.FixedLocator(np.arange(0, 360 + 1e-4, 10))
gl.ylocator = mticker.FixedLocator(np.arange(-90, 90 + 1e-4, 10))
plt.text(0.5, -0.03, plt_text, transform=ax.transAxes, fontsize=8,
         ha='center', va='top', rotation='horizontal', linespacing=1.5)

fig.subplots_adjust(left=0.01, right=0.99, bottom=1/(7+1), top=0.99)
fig.savefig(opng, transparent=True)
plt.close()

end_time = time.perf_counter()
print(f"Execution time: {end_time - start_time:.6f} seconds")


# del channels, rgb


'''
#-------------------------------- projection using pyproj
import pyproj
ds = xr.open_dataset(ifile)

iproj = pyproj.CRS.from_proj4(ds.geostationary.proj4)
oproj = pyproj.CRS.from_epsg(4326)
transformer = pyproj.Transformer.from_crs(iproj, oproj, always_xy=True)

xx, yy = np.meshgrid(ds.x.values, ds.y.values)
lon, lat = transformer.transform(xx, yy)

np.nanmax(lon[np.isfinite(lon)])
np.nanmax(lat[np.isfinite(lat)])
dis = 3840000
print(transformer.transform(3840000, 3840000))


# fig, ax = regional_plot(extent=[40, 240, -85, 85], central_longitude=180, figsize = np.array([6.6, 6.6+0.9])/2.54, border_color='yellow', lw=0.1)

ifile = sorted(glob.glob(f'{dfolder}/{year}/{month:02d}/{day:02d}/{hour:02d}{minute:02d}/*OBS_{iband}*'))[-1]
channels[iband] = channels[iband].coarsen(y=2, x=2, boundary='trim').mean()

    # use xarray
    %%timeit
    channels[iband] = xr.open_dataset(ifile[-1], chunks={})
    var_name = next(var for var in channels[iband].data_vars if var.startswith('channel_00'))
    channels[iband] = channels[iband][var_name].squeeze().values
    if iband == 'B03':
        channels[iband] = block_reduce(channels[iband], block_size=(2, 2), func=np.mean)



# Himawari User's Guide: https://www.data.jma.go.jp/mscweb/en/support/support.html
# Real time image: https://www.data.jma.go.jp/mscweb/data/himawari/sat_img.php?area=fd_
Thresholds: https://user.eumetsat.int/resources/case-studies/night-time-fog-over-india
# RGB: https://www.jma.go.jp/jma/jma-eng/satellite/VLab/RGB_QG.html / rubbish
'''
# endregion


# region plot himawari true color and night microphysics

start_time = time.perf_counter()

year, month, day, hour, minute = 2020, 6, 2, 3, 0
dfolder = Path('/g/data/ra22/satellite-products/arc/obs/himawari-ahi/fldk/latest')
# dfolder = Path('/g/data/ra22/satellite-products/nrt/obs/himawari-ahi/fldk/latest')

opng = Path(f'figures/3_satellites/3.0_hamawari/3.0.0_image/3.0.0.0 himawari true_color and night_microphysics {year}{month:02d}{day:02d}{hour:02d}{minute:02d}.png')
extent = [-5499500., 5499500., -5499500., 5499500.]
transform = ccrs.Geostationary(central_longitude=140.7, satellite_height=35785863.0)
bands=['B03', 'B02', 'B01', 'B07', 'B13', 'B15']

plt_text = f'{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d} UTC\nHimawari True Color and Night Microphysics RGB'

channels = {}
for iband in bands:
    # iband='B07'
    print(f'#-------------------------------- {iband}')
    
    ifile = sorted(Path(dfolder/f'{year}/{month:02d}/{day:02d}/{hour:02d}{minute:02d}').glob(f'*OBS_{iband}*'))
    if not ifile:
        print('Warning: No file found')
        continue
    
    channels[iband] = Dataset(ifile[-1], 'r')
    var_name = next(var for var in channels[iband].variables if var.startswith("channel_00"))
    # print(f'#---------------- {var_name}')
    channels[iband] = np.squeeze(channels[iband].variables[var_name][:])
    
    if iband in ['B03', 'B02', 'B01']:
        channels[iband] = channels[iband].filled(0)
    
    if iband == 'B03':
        channels[iband] = block_reduce(channels[iband], block_size=(2, 2), func=np.mean)

channels['B15-B13'] = channels['B15']-channels['B13']
channels['B13-B07'] = channels['B13']-channels['B07']

channels['B15-B13'] = (channels['B15-B13'] + 4) / (2 + 4)
channels['B13-B07'] = (channels['B13-B07'] + 0) / (10 + 0)
channels['B13'] = (channels['B13'] - 243) / (293 - 243)

for iband in ['B15-B13', 'B13-B07', 'B13']:
    channels[iband] = channels[iband].filled(0)

rgb1 = np.zeros(channels['B03'].shape + (3,), dtype=np.float32)
for idx, iband in enumerate(['B03', 'B02', 'B01']):
    rgb1[:, :, idx] = channels[iband]
rgb1 = np.clip(rgb1, 0, 1, out=rgb1)
rgb1 **= (1 / 2.2)

rgb2 = np.zeros(channels['B15-B13'].shape + (3,), dtype=np.float32)
for idx, iband in enumerate(['B15-B13', 'B13-B07', 'B13']):
    rgb2[:, :, idx] = channels[iband]
rgb2 = np.clip(rgb2, 0, 1, out=rgb2)
rgb2 = np.kron(rgb2, np.ones((2, 2, 1)))

luminance = 0.2126 * rgb1[..., 0] + 0.7152 * rgb1[..., 1] + 0.0722 * rgb1[..., 2]
rgb = np.where((luminance > 0.15)[..., None], rgb1, rgb2)


# end_time = time.perf_counter()
# print(f"Execution time: {end_time - start_time:.1f} seconds")
# start_time = time.perf_counter()


fig, ax = plt.subplots(figsize=np.array([7, 7+1])/2.54, subplot_kw={'projection': transform})

ax.imshow(rgb, extent=extent, transform=transform,
          interpolation='none', origin='upper', resample=False)

coastline = cfeature.NaturalEarthFeature(
    'physical', 'coastline', '10m', edgecolor='yellow',
    facecolor='none', lw=0.1)
ax.add_feature(coastline, zorder=2, alpha=0.75)
borders = cfeature.NaturalEarthFeature(
    'cultural', 'admin_0_boundary_lines_land', '10m',
    edgecolor='yellow', facecolor='none', lw=0.1)
ax.add_feature(borders, zorder=2, alpha=0.75)
gl = ax.gridlines(
    crs=ccrs.PlateCarree(), lw=0.1, zorder=2, alpha=0.35,
    color='yellow', linestyle='--',)
gl.xlocator = mticker.FixedLocator(np.arange(0, 360 + 1e-4, 10))
gl.ylocator = mticker.FixedLocator(np.arange(-90, 90 + 1e-4, 10))
plt.text(0.5, -0.03, plt_text, transform=ax.transAxes, fontsize=8,
         ha='center', va='top', rotation='horizontal', linespacing=1.5)

fig.subplots_adjust(left=0.01, right=0.99, bottom=1/(7+1), top=0.99)
fig.savefig(opng, transparent=True)
plt.close()

end_time = time.perf_counter()
print(f"Execution time: {end_time - start_time:.1f} seconds")


# del channels, rgb1, rgb2, rgb


'''
# option 0: rgb sum
# rgb = np.where((rgb1.sum(axis=2) > 0.15)[..., None], rgb1, rgb2)

# # option 1: blending
# weight = np.clip((rgb1.sum(axis=2) - 0.15) / 0.05, 0, 1)[..., None]
# rgb = weight * rgb1 + (1 - weight) * rgb2

# # option 3: luminance and morphological operations
# from scipy.ndimage import binary_opening, binary_closing
# luminance = 0.2126 * rgb1[..., 0] + 0.7152 * rgb1[..., 1] + 0.0722 * rgb1[..., 2]
# daytime_mask = luminance > 0.05
# daytime_mask = binary_opening(daytime_mask, structure=np.ones((10,10)))  # Remove small noise
# daytime_mask = binary_closing(daytime_mask, structure=np.ones((10,10)))  # Fill small gaps
# rgb = np.where(daytime_mask[..., None], rgb1, rgb2)

'''
# endregion


# region plot himawari true color and night microphysics over c2_domain

# options
year, month, day, hour, minute = 2020, 6, 2, 3, 0
dfolder = Path('/g/data/ra22/satellite-products/arc/obs/himawari-ahi/fldk/latest')
# dfolder = Path('/g/data/ra22/satellite-products/nrt/obs/himawari-ahi/fldk/latest')
plt_region = 'c2_domain'

# settings
extent = [110.58, 157.34, -43.69, -7.01]
min_lon, max_lon, min_lat, max_lat = extent

# get data: to make it a function
bands=['B03', 'B02', 'B01'] #, 'B07', 'B13', 'B15'
channels = {}
for iband in bands:
    # iband='B07'
    print(f'#-------------------------------- {iband}')
    
    ifile = sorted(Path(dfolder/f'{year}/{month:02d}/{day:02d}/{hour:02d}{minute:02d}').glob(f'*OBS_{iband}*'))
    if not ifile:
        print('Warning: No file found')
        continue
    
    channels[iband] = Dataset(ifile[-1], 'r')
    var_name = next(var for var in channels[iband].variables if var.startswith("channel_00"))
    # print(f'#---------------- {var_name}')
    channels[iband] = np.squeeze(channels[iband].variables[var_name][:])
    
    if iband in ['B03', 'B02', 'B01']:
        channels[iband] = channels[iband].filled(0)
    
    if iband == 'B03':
        channels[iband] = block_reduce(channels[iband], block_size=(2, 2), func=np.mean)

# channels['B15-B13'] = channels['B15']-channels['B13']
# channels['B13-B07'] = channels['B13']-channels['B07']

# channels['B15-B13'] = (channels['B15-B13'] + 4) / (2 + 4)
# channels['B13-B07'] = (channels['B13-B07'] + 0) / (10 + 0)
# channels['B13'] = (channels['B13'] - 243) / (293 - 243)

# for iband in ['B15-B13', 'B13-B07', 'B13']:
#     channels[iband] = channels[iband].filled(0)

rgb1 = np.zeros(channels['B03'].shape + (3,), dtype=np.float32)
for idx, iband in enumerate(['B03', 'B02', 'B01']):
    rgb1[:, :, idx] = channels[iband]
rgb1 = np.clip(rgb1, 0, 1, out=rgb1)
gamma = 3
rgb1 **= (1 / gamma)

# rgb2 = np.zeros(channels['B15-B13'].shape + (3,), dtype=np.float32)
# for idx, iband in enumerate(['B15-B13', 'B13-B07', 'B13']):
#     rgb2[:, :, idx] = channels[iband]
# rgb2 = np.clip(rgb2, 0, 1, out=rgb2)
# rgb2 = np.kron(rgb2, np.ones((2, 2, 1)))

# luminance = 0.2126 * rgb1[..., 0] + 0.7152 * rgb1[..., 1] + 0.0722 * rgb1[..., 2]
# rgb = np.where((luminance > 0.15)[..., None], rgb1, rgb2).astype('float32')
rgb = rgb1.astype('float32')

fig, ax = regional_plot(
    figsize = np.array([6.6, 6]) / 2.54,
    extent=[110.58, 157.34, -43.69, -7.01], central_longitude=180,
    lw=0.1, border_color='yellow')
ax.imshow(
    rgb, extent=[-5499500., 5499500., -5499500., 5499500.],
    transform=ccrs.Geostationary(central_longitude=140.7, satellite_height=35785863.0),
    interpolation='none', origin='upper', resample=False)

# plot cloudsat tracks
geolocation_all = pd.read_pickle(f'data/obs/CloudSat_CALIPSO/2B-CLDCLASS-LIDAR.P1_R05/2020/geolocation_df.pkl')
geolocation_subset = geolocation_all.loc[(
    (geolocation_all.date_time >= pd.Timestamp(2020, 6, 2, 4, 0)) & \
        (geolocation_all.date_time <= pd.Timestamp(2020, 6, 2, 5, 0)) & \
            (geolocation_all.lat >= -12) & \
                # -16.2876
                (geolocation_all.lat <= -7))]
ax.scatter(geolocation_subset.lon, geolocation_subset.lat, s=1, c='tab:orange',
           lw=0, transform=ccrs.PlateCarree())

fig.text(0.5, 0.01, f'{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d} UTC\nHimawari True Color RGB',
         ha='center', va='bottom', linespacing=1.3)
fig.subplots_adjust(left=0.005, right=0.995, bottom=0.15, top=0.95)
opng = f'figures/3_satellites/3.0_hamawari/3.0.0_image/3.0.0.1 Himawari image {plt_region} {year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d} {gamma}.png'
fig.savefig(opng)



# endregion


# region animate himawari true color and/or night microphysics

parser=argparse.ArgumentParser()
parser.add_argument('-y', '--year', type=int, required=True,)
parser.add_argument('-m', '--month', type=int, required=True,)
args = parser.parse_args()

year=args.year
month=args.month
# year=2020; month=7

last_day = calendar.monthrange(year, month)[1]
time_series = pd.date_range(start=f'{year}-{month:02d}-01 00:00',
                            end=f'{year}-{month:02d}-{last_day} 23:50',
                            freq='10min')
time_step = 'hourly' #minutely
if time_step == 'hourly': time_series = time_series[::6]
# time_series=time_series[:24]

# ioption='true_color'
# ioption='night_microphysics'
ioption='true_color_and_night_microphysics'
band_map = {
    'true_color': ['B03', 'B02', 'B01'],
    'night_microphysics': ['B07', 'B13', 'B15'],
    'true_color_and_night_microphysics': ['B03', 'B02', 'B01', 'B07', 'B13', 'B15']}

dfolder = Path('/g/data/ra22/satellite-products/arc/obs/himawari-ahi/fldk/latest')
# dfolder = Path('/g/data/ra22/satellite-products/nrt/obs/himawari-ahi/fldk/latest')
omp4 = Path(f'figures/3_satellites/3.0_hamawari/3.0.0_image/3.0.0.0 himawari {ioption} {year}{month:02d} {time_step}.mp4')
extent = [-5499500., 5499500., -5499500., 5499500.]
transform = ccrs.Geostationary(central_longitude=140.7, satellite_height=35785863.0)

fig, ax = plt.subplots(figsize=np.array([7, 7+1])/2.54, subplot_kw={'projection': transform})
coastline = cfeature.NaturalEarthFeature(
    'physical', 'coastline', '10m', edgecolor='yellow',
    facecolor='none', lw=0.1)
ax.add_feature(coastline, zorder=2, alpha=0.75)
borders = cfeature.NaturalEarthFeature(
    'cultural', 'admin_0_boundary_lines_land', '10m',
    edgecolor='yellow', facecolor='none', lw=0.1)
ax.add_feature(borders, zorder=2, alpha=0.75)
gl = ax.gridlines(
    crs=ccrs.PlateCarree(), lw=0.1, zorder=2, alpha=0.35,
    color='yellow', linestyle='--',)
gl.xlocator = mticker.FixedLocator(np.arange(0, 360 + 1e-4, 10))
gl.ylocator = mticker.FixedLocator(np.arange(-90, 90 + 1e-4, 10))

plt_objs = []
def update_frames(itime):
    # itime = -1
    global plt_objs
    for plt_obj in plt_objs:
        plt_obj.remove()
    plt_objs = []
    
    start_time = time.perf_counter()
    day = time_series[itime].day
    hour = time_series[itime].hour
    minute = time_series[itime].minute
    title = f'{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d} UTC\n{ioption.replace('_', ' ').title().replace('And', 'and')} RGB Himawari 8/9'
    print(f'#-------------------------------- {year}{month:02d}{day:02d}{hour:02d}{minute:02d}')
    
    bands = band_map[ioption]
    channels = {}
    for iband in bands:
        # iband='B07'
        print(f'#---------------- {iband}')
        ifile = sorted(Path(dfolder/f'{year}/{month:02d}/{day:02d}/{hour:02d}{minute:02d}').glob(f'*OBS_{iband}*'))
        if not ifile:
            print('Warning: No file found')
            return
        
        channels[iband] = Dataset(ifile[-1], 'r')
        var_name = next(var for var in channels[iband].variables if var.startswith("channel_00"))
        channels[iband] = np.squeeze(channels[iband].variables[var_name][:])
        
        if iband in ['B03', 'B02', 'B01']:
            channels[iband] = channels[iband].filled(0)
        
        if iband == 'B03':
            channels[iband] = block_reduce(channels[iband], block_size=(2, 2), func=np.mean)
    
    if ioption in ['night_microphysics', 'true_color_and_night_microphysics']:
        channels['B15-B13'] = channels['B15']-channels['B13']
        channels['B13-B07'] = channels['B13']-channels['B07']
        
        channels['B15-B13'] = (channels['B15-B13'] + 4) / (2 + 4)
        channels['B13-B07'] = (channels['B13-B07'] + 0) / (10 + 0)
        channels['B13'] = (channels['B13'] - 243) / (293 - 243)
        
        if ioption=='night_microphysics': bands=['B15-B13', 'B13-B07', 'B13']
        for iband in ['B15-B13', 'B13-B07', 'B13']:
            channels[iband] = channels[iband].filled(0)
    
    if ioption in ['true_color', 'night_microphysics']:
        rgb = np.zeros(channels[bands[0]].shape + (3,), dtype=np.float32)
        for idx, iband in enumerate(bands):
            rgb[:, :, idx] = channels[iband]
        rgb = np.clip(rgb, 0, 1, out=rgb)
        if ioption=='true_color': rgb **= (1 / 2.2)
    elif ioption=='true_color_and_night_microphysics':
        rgb1 = np.zeros(channels['B03'].shape + (3,), dtype=np.float32)
        for idx, iband in enumerate(['B03', 'B02', 'B01']):
            rgb1[:, :, idx] = channels[iband]
        rgb1 = np.clip(rgb1, 0, 1, out=rgb1)
        rgb1 **= (1 / 2.2)
        
        rgb2 = np.zeros(channels['B15-B13'].shape + (3,), dtype=np.float32)
        for idx, iband in enumerate(['B15-B13', 'B13-B07', 'B13']):
            rgb2[:, :, idx] = channels[iband]
        rgb2 = np.clip(rgb2, 0, 1, out=rgb2)
        rgb2 = np.kron(rgb2, np.ones((2, 2, 1)))
        
        luminance = 0.2126 * rgb1[..., 0] + 0.7152 * rgb1[..., 1] + 0.0722 * rgb1[..., 2]
        rgb = np.where((luminance > 0.15)[..., None], rgb1, rgb2)
    
    plt_im = ax.imshow(
        rgb, extent=extent, transform=transform,
        interpolation='none', origin='upper', resample=False)
    plt_text = plt.text(
        0.5, -0.03, title, transform=ax.transAxes, fontsize=8,
        ha='center', va='top', rotation='horizontal', linespacing=1.5)
    
    del channels
    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.6f} seconds")
    plt_objs = [plt_im, plt_text]
    return(plt_objs)

fig.subplots_adjust(left=0.01, right=0.99, bottom=1/(7+1), top=0.99)

ani = animation.FuncAnimation(
    fig, update_frames, frames=len(time_series), interval=500, blit=False)
if os.path.exists(omp4): os.remove(omp4)
ani.save(omp4, progress_callback=lambda iframe, n: print(f'Frame {iframe}/{n}'))



'''
print(time_series[-1].day)
print(time_series[-1].hour)
'''
# endregion


# region plot himawari true color and night microphysics + CloudSat-Calipso 2B-CLDCLASS-LIDAR

geolocation_all = pd.read_pickle(f'data/obs/CloudSat_CALIPSO/2B-CLDCLASS-LIDAR.P1_R05/geolocation_all.pkl')

year, month, day, hour, minute = 2020, 6, 1, 0, 0
dfolder = Path('/g/data/ra22/satellite-products/arc/obs/himawari-ahi/fldk/latest')
# dfolder = Path('/g/data/ra22/satellite-products/nrt/obs/himawari-ahi/fldk/latest')

start_time = pd.Timestamp(year, month, day, hour, minute)
end_time = start_time + pd.Timedelta(minutes=10)
geolocation_subset = geolocation_all.loc[((geolocation_all.date_time > start_time) & (geolocation_all.date_time < end_time) & ((geolocation_all.lon > 50) | (geolocation_all.lon < -130)))]
print(geolocation_subset.shape[0])

opng = Path(f'figures/3_satellites/3.0_hamawari/3.0.0_image/3.0.0.0 himawari true_color and night_microphysics and CC_2B_CLDCLASS_LIDAR {year}{month:02d}{day:02d}{hour:02d}{minute:02d}.png')
extent = [-5499500., 5499500., -5499500., 5499500.]
transform = ccrs.Geostationary(central_longitude=140.7, satellite_height=35785863.0)
bands=['B03', 'B02', 'B01', 'B07', 'B13', 'B15']

plt_text = f'{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d} UTC CloudSat-CALIPSO tracks and\nTrue Color and Night Microphysics RGB Himawari 8/9'

channels = {}
for iband in bands:
    # iband='B07'
    print(f'#-------------------------------- {iband}')
    
    ifile = sorted(Path(dfolder/f'{year}/{month:02d}/{day:02d}/{hour:02d}{minute:02d}').glob(f'*OBS_{iband}*'))
    if not ifile:
        print('Warning: No file found')
        continue
    
    channels[iband] = Dataset(ifile[-1], 'r')
    var_name = next(var for var in channels[iband].variables if var.startswith("channel_00"))
    # print(f'#---------------- {var_name}')
    channels[iband] = np.squeeze(channels[iband].variables[var_name][:])
    
    if iband in ['B03', 'B02', 'B01']:
        channels[iband] = channels[iband].filled(0)
    
    if iband == 'B03':
        channels[iband] = block_reduce(channels[iband], block_size=(2, 2), func=np.mean)

channels['B15-B13'] = channels['B15']-channels['B13']
channels['B13-B07'] = channels['B13']-channels['B07']

channels['B15-B13'] = (channels['B15-B13'] + 4) / (2 + 4)
channels['B13-B07'] = (channels['B13-B07'] + 0) / (10 + 0)
channels['B13'] = (channels['B13'] - 243) / (293 - 243)

for iband in ['B15-B13', 'B13-B07', 'B13']:
    channels[iband] = channels[iband].filled(0)

rgb1 = np.zeros(channels['B03'].shape + (3,), dtype=np.float32)
for idx, iband in enumerate(['B03', 'B02', 'B01']):
    rgb1[:, :, idx] = channels[iband]
rgb1 = np.clip(rgb1, 0, 1, out=rgb1)
rgb1 **= (1 / 2.2)

rgb2 = np.zeros(channels['B15-B13'].shape + (3,), dtype=np.float32)
for idx, iband in enumerate(['B15-B13', 'B13-B07', 'B13']):
    rgb2[:, :, idx] = channels[iband]
rgb2 = np.clip(rgb2, 0, 1, out=rgb2)
rgb2 = np.kron(rgb2, np.ones((2, 2, 1)))

luminance = 0.2126 * rgb1[..., 0] + 0.7152 * rgb1[..., 1] + 0.0722 * rgb1[..., 2]
rgb = np.where((luminance > 0.15)[..., None], rgb1, rgb2)


fig, ax = plt.subplots(figsize=np.array([7, 7+1])/2.54, subplot_kw={'projection': transform})

ax.imshow(rgb, extent=extent, transform=transform,
          interpolation='none', origin='upper', resample=False)
ax.scatter(geolocation_subset.lon, geolocation_subset.lat, s=1, c='tab:orange',
           lw=0, transform=ccrs.PlateCarree())

min_lon, max_lon, min_lat, max_lat = [110.58, 157.34, -43.69, -7.01]
top_edge = [(lon, max_lat) for lon in np.linspace(min_lon, max_lon, 100)]
right_edge = [(max_lon, lat) for lat in np.linspace(max_lat, min_lat, 100)]
bottom_edge = [(lon, min_lat) for lon in np.linspace(max_lon, min_lon, 100)]
left_edge = [(min_lon, lat) for lat in np.linspace(min_lat, max_lat, 100)]
rectangle_coords = top_edge + right_edge + bottom_edge + left_edge + [top_edge[0]]
lons, lats = zip(*rectangle_coords)
ax.plot(lons, lats, color='red', lw=0.5, transform=ccrs.PlateCarree())

gbr_shp = gpd.read_file('data/others/Great_Barrier_Reef_Marine_Park_Boundary/Great_Barrier_Reef_Marine_Park_Boundary.shp')
gbr_shp.plot(ax=ax, edgecolor='tab:blue', facecolor='none', lw=0.8, zorder=2,
             transform=ccrs.PlateCarree())

coastline = cfeature.NaturalEarthFeature(
    'physical', 'coastline', '10m', edgecolor='yellow',
    facecolor='none', lw=0.1)
ax.add_feature(coastline, zorder=2, alpha=0.75)
borders = cfeature.NaturalEarthFeature(
    'cultural', 'admin_0_boundary_lines_land', '10m',
    edgecolor='yellow', facecolor='none', lw=0.1)
ax.add_feature(borders, zorder=2, alpha=0.75)
gl = ax.gridlines(
    crs=ccrs.PlateCarree(), lw=0.1, zorder=2, alpha=0.35,
    color='yellow', linestyle='--',)
gl.xlocator = mticker.FixedLocator(np.arange(-180, 180 + 1e-4, 10))
gl.ylocator = mticker.FixedLocator(np.arange(-90, 90 + 1e-4, 10))
plt.text(0.5, -0.03, plt_text, transform=ax.transAxes, fontsize=8,
         ha='center', va='top', rotation='horizontal', linespacing=1.5)

fig.subplots_adjust(left=0.01, right=0.99, bottom=1/(7+1), top=0.99)
fig.savefig(opng, transparent=True)
plt.close()




'''
'''
# endregion


# region animate himawari true color and night microphysics + CloudSat-Calipso 2B-CLDCLASS-LIDAR

parser=argparse.ArgumentParser()
parser.add_argument('-y', '--year', type=int, required=True,)
parser.add_argument('-m', '--month', type=int, required=True,)
args = parser.parse_args()

year=args.year
month=args.month
# year=2022; month=1

last_day = calendar.monthrange(year, month)[1]
time_series = pd.date_range(start=f'{year}-{month:02d}-01 00:00',
                            end=f'{year}-{month:02d}-{last_day} 23:50',
                            freq='10min')
time_step = 'hourly' # 'minutely' #
if time_step == 'hourly': time_series = time_series[::6]
# time_series=time_series[:24]

geolocation_all = pd.read_pickle(f'scratch/data/obs/CloudSat_CALIPSO/2B-CLDCLASS-LIDAR.P1_R05/geolocation_all.pkl')
geolocation_all = geolocation_all.loc[((geolocation_all.date_time >= time_series[0]) & (geolocation_all.date_time <= (time_series[-1] + pd.Timedelta('10min'))))]

# ioption='true_color'
# ioption='night_microphysics'
ioption='true_color_and_night_microphysics'
band_map = {
    'true_color': ['B03', 'B02', 'B01'],
    'night_microphysics': ['B07', 'B13', 'B15'],
    'true_color_and_night_microphysics': ['B03', 'B02', 'B01', 'B07', 'B13', 'B15']}

dfolder = Path('/g/data/ra22/satellite-products/arc/obs/himawari-ahi/fldk/latest')
# dfolder = Path('/g/data/ra22/satellite-products/nrt/obs/himawari-ahi/fldk/latest')
omp4 = Path(f'figures/3_satellites/3.0_hamawari/3.0.0_image/3.0.0.0 himawari {ioption} and cc_2B_CLDCLASS_LIDAR {year}{month:02d} {time_step}.mp4')
extent = [-5499500., 5499500., -5499500., 5499500.]
transform = ccrs.Geostationary(central_longitude=140.7, satellite_height=35785863.0)

fig, ax = plt.subplots(figsize=np.array([7, 7+1])/2.54, subplot_kw={'projection': transform})
coastline = cfeature.NaturalEarthFeature(
    'physical', 'coastline', '10m', edgecolor='yellow',
    facecolor='none', lw=0.1)
ax.add_feature(coastline, zorder=2, alpha=0.75)
borders = cfeature.NaturalEarthFeature(
    'cultural', 'admin_0_boundary_lines_land', '10m',
    edgecolor='yellow', facecolor='none', lw=0.1)
ax.add_feature(borders, zorder=2, alpha=0.75)
gl = ax.gridlines(
    crs=ccrs.PlateCarree(), lw=0.1, zorder=2, alpha=0.35,
    color='yellow', linestyle='--',)
gl.xlocator = mticker.FixedLocator(np.arange(-180, 180 + 1e-4, 10))
gl.ylocator = mticker.FixedLocator(np.arange(-90, 90 + 1e-4, 10))
min_lon, max_lon, min_lat, max_lat = [110.58, 157.34, -43.69, -7.01]
top_edge = [(lon, max_lat) for lon in np.linspace(min_lon, max_lon, 100)]
right_edge = [(max_lon, lat) for lat in np.linspace(max_lat, min_lat, 100)]
bottom_edge = [(lon, min_lat) for lon in np.linspace(max_lon, min_lon, 100)]
left_edge = [(min_lon, lat) for lat in np.linspace(min_lat, max_lat, 100)]
rectangle_coords = top_edge + right_edge + bottom_edge + left_edge + [top_edge[0]]
lons, lats = zip(*rectangle_coords)
ax.plot(lons, lats, color='red', lw=0.5, transform=ccrs.PlateCarree())

gbr_shp = gpd.read_file('data/others/Great_Barrier_Reef_Marine_Park_Boundary/Great_Barrier_Reef_Marine_Park_Boundary.shp')
gbr_shp.plot(ax=ax, edgecolor='tab:blue', facecolor='none', lw=0.8, zorder=2,
             transform=ccrs.PlateCarree())

plt_objs = []
def update_frames(itime):
    time1 = time.perf_counter()
    # itime = 0
    global plt_objs
    for plt_obj in plt_objs:
        plt_obj.remove()
    plt_objs = []
    
    day = time_series[itime].day
    hour = time_series[itime].hour
    minute = time_series[itime].minute
    title = f'{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d} UTC CloudSat-CALIPSO tracks and\n{ioption.replace('_', ' ').title().replace('And', 'and')} RGB Himawari 8/9'
    print(f'#-------------------------------- {year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}')
    
    bands = band_map[ioption]
    channels = {}
    for iband in bands:
        # iband='B07'
        print(f'#---------------- {iband}')
        ifile = sorted(Path(dfolder/f'{year}/{month:02d}/{day:02d}/{hour:02d}{minute:02d}').glob(f'*OBS_{iband}*'))
        if not ifile:
            print('Warning: No file found')
            return
        
        channels[iband] = Dataset(ifile[-1], 'r')
        var_name = next(var for var in channels[iband].variables if var.startswith("channel_00"))
        channels[iband] = np.squeeze(channels[iband].variables[var_name][:])
        
        if iband in ['B03', 'B02', 'B01']:
            channels[iband] = channels[iband].filled(0)
        
        if iband == 'B03':
            channels[iband] = block_reduce(channels[iband], block_size=(2, 2), func=np.mean)
    
    if ioption in ['night_microphysics', 'true_color_and_night_microphysics']:
        channels['B15-B13'] = channels['B15']-channels['B13']
        channels['B13-B07'] = channels['B13']-channels['B07']
        
        channels['B15-B13'] = (channels['B15-B13'] + 4) / (2 + 4)
        channels['B13-B07'] = (channels['B13-B07'] + 0) / (10 + 0)
        channels['B13'] = (channels['B13'] - 243) / (293 - 243)
        
        if ioption=='night_microphysics': bands=['B15-B13', 'B13-B07', 'B13']
        for iband in ['B15-B13', 'B13-B07', 'B13']:
            channels[iband] = channels[iband].filled(0)
    
    if ioption in ['true_color', 'night_microphysics']:
        rgb = np.zeros(channels[bands[0]].shape + (3,), dtype=np.float32)
        for idx, iband in enumerate(bands):
            rgb[:, :, idx] = channels[iband]
        rgb = np.clip(rgb, 0, 1, out=rgb)
        if ioption=='true_color': rgb **= (1 / 2.2)
    elif ioption=='true_color_and_night_microphysics':
        rgb1 = np.zeros(channels['B03'].shape + (3,), dtype=np.float32)
        for idx, iband in enumerate(['B03', 'B02', 'B01']):
            rgb1[:, :, idx] = channels[iband]
        rgb1 = np.clip(rgb1, 0, 1, out=rgb1)
        rgb1 **= (1 / 2.2)
        
        rgb2 = np.zeros(channels['B15-B13'].shape + (3,), dtype=np.float32)
        for idx, iband in enumerate(['B15-B13', 'B13-B07', 'B13']):
            rgb2[:, :, idx] = channels[iband]
        rgb2 = np.clip(rgb2, 0, 1, out=rgb2)
        rgb2 = np.kron(rgb2, np.ones((2, 2, 1)))
        
        luminance = 0.2126 * rgb1[..., 0] + 0.7152 * rgb1[..., 1] + 0.0722 * rgb1[..., 2]
        rgb = np.where((luminance > 0.15)[..., None], rgb1, rgb2)
    
    plt_im = ax.imshow(
        rgb, extent=extent, transform=transform,
        interpolation='none', origin='upper', resample=False)
    plt_text = plt.text(
        0.5, -0.03, title, transform=ax.transAxes, fontsize=8,
        ha='center', va='top', rotation='horizontal', linespacing=1.5)
    
    start_time = pd.Timestamp(year, month, day, hour, minute)
    end_time = start_time + pd.Timedelta(minutes=60)
    geolocation_subset = geolocation_all.loc[((geolocation_all.date_time > start_time) & (geolocation_all.date_time < end_time) & ((geolocation_all.lon > 50) | (geolocation_all.lon < -130)))]
    print(geolocation_subset.shape[0])
    plt_scatter = ax.scatter(
        geolocation_subset.lon, geolocation_subset.lat, s=1, c='tab:orange',
        lw=0, transform=ccrs.PlateCarree())
    
    plt_objs = [plt_im, plt_text, plt_scatter]
    del channels
    time2 = time.perf_counter()
    print(f'Execution time: {time2 - time1:.1f} seconds')
    return(plt_objs)

fig.subplots_adjust(left=0.01, right=0.99, bottom=1/(7+1), top=0.99)

ani = animation.FuncAnimation(
    fig, update_frames, frames=len(time_series[:3]), interval=500, blit=False)
if os.path.exists(omp4): os.remove(omp4)
ani.save(omp4, progress_callback=lambda iframe, n: print(f'Frame {iframe}/{n}'))


# endregion


