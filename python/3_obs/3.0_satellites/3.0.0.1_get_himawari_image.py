

# qsub -I -q copyq -l walltime=10:00:00,ncpus=1,mem=192GB,storage=gdata/v46+gdata/rt52+gdata/ob53+gdata/zv2+gdata/ra22


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

# management
import os
import sys  # print(sys.path)
sys.path.append(os.getcwd() + '/code/gbr_future/module')
import glob
import time
from pathlib import Path

# self defined
from mapplot import (
    regional_plot,
    )

# endregion


# region plot himawari data

start_time = time.perf_counter()

year, month, day, hour, minute = 2020, 7, 1, 8, 0
# ioption='true_color'
ioption='night_microphysics'

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


del channels, rgb


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


# region plot himawari data true color+night microphysics

year, month, day, hour, minute = 2020, 7, 1, 8, 0
dfolder = Path('/g/data/ra22/satellite-products/arc/obs/himawari-ahi/fldk/latest')
# dfolder = Path('/g/data/ra22/satellite-products/nrt/obs/himawari-ahi/fldk/latest')

opng = Path(f'figures/3_satellites/3.0_hamawari/3.0.0_image/3.0.0.0 himawari true_color and night_microphysics {year}{month:02d}{day:02d}{hour:02d}{minute:02d}.png')
extent = [-5499500., 5499500., -5499500., 5499500.]
transform = ccrs.Geostationary(central_longitude=140.7, satellite_height=35785863.0)
bands=['B03', 'B02', 'B01', 'B07', 'B13', 'B15']

plt_text = f'{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d} UTC\nTrue Color and Night Microphysics RGB Himawari 8/9'

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

rgb = np.where((rgb1.sum(axis=2) > 0.15)[..., None], rgb1, rgb2)


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


# endregion

