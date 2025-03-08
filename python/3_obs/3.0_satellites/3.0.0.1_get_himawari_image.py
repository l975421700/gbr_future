

# qsub -I -q copyq -l walltime=2:00:00,ncpus=1,mem=20GB,storage=gdata/v46+gdata/rt52+gdata/ob53+gdata/zv2+gdata/ra22


# region import packages

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
from skimage.measure import block_reduce

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

year, month, day, hour, minute = 2025, 3, 7, 12, 0
ioption='true_color'
# ioption='night_microphysics'

# dfolder = Path('/g/data/ra22/satellite-products/arc/obs/himawari-ahi/fldk/latest')
dfolder = Path('/g/data/ra22/satellite-products/nrt/obs/himawari-ahi/fldk/latest')
opng = Path(f'figures/3_satellites/3.0_hamawari/3.0.0_image/3.0.0.0 himawari {ioption} {year}{month:02d}{day:02d}{hour:02d}{minute:02d}.png')

band_map = {
    'true_color': ['B03', 'B02', 'B01'],
    'night_microphysics': ['B07', 'B13', 'B15']
}
bands = band_map[ioption]
plt_text = f'{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d} UTC\n{ioption.replace('_', ' ').title()} RGB Himawari 8/9'

channels = {}
for iband in bands:
    # iband='B03'
    print(f'#-------------------------------- {iband}')
    
    ifile = sorted(Path(dfolder/f'{year}/{month:02d}/{day:02d}/{hour:02d}{minute:02d}').glob(f'*OBS_{iband}*'))
    if not ifile:
        print('Warning: No file found')
        continue
    
    channels[iband] = xr.open_dataset(ifile[-1], chunks={})
    var_name = next(var for var in channels[iband].data_vars if var.startswith('channel_00'))
    channels[iband] = channels[iband][var_name].squeeze().values
    
    if iband == 'B03':
        channels[iband] = block_reduce(channels[iband], block_size=(2, 2), func=np.mean)

extent = [-5499500., 5499500., -5499500., 5499500.]
transform = ccrs.Geostationary(central_longitude=140.7, satellite_height=35785863.0)

rgb = np.zeros(channels[bands[0]].shape + (3,), dtype=np.float32)
for idx, iband in enumerate(bands):
    print(f'{idx} {iband}')
    rgb[:, :, idx] = channels[iband]

rgb = rgb.clip(0, 1) ** (1 / 2.2)

end_time = time.perf_counter()
print(f"Execution time: {end_time - start_time:.6f} seconds")


start_time = time.perf_counter()

fig, ax = plt.subplots(figsize=np.array([6.6, 6.6+0.9])/2.54, subplot_kw={'projection': transform})

ax.imshow(rgb, extent=extent,
          transform=transform, rasterized=True,
          interpolation='none', origin='upper', resample=False)

coastline = cfeature.NaturalEarthFeature(
    'physical', 'coastline', '10m', edgecolor='yellow',
    facecolor='none', lw=0.1)
ax.add_feature(coastline, zorder=2, rasterized=True, alpha=0.75)
borders = cfeature.NaturalEarthFeature(
    'cultural', 'admin_0_boundary_lines_land', '10m',
    edgecolor='yellow', facecolor='none', lw=0.1)
ax.add_feature(borders, zorder=2, rasterized=True, alpha=0.75)
ax.gridlines(
    crs=ccrs.PlateCarree(), lw=0.1, zorder=2, alpha=0.35,
    color='yellow', linestyle='--',
    xlocs = np.arange(0, 360 + 1e-4, 10), ylocs=np.arange(-90, 90 + 1e-4, 10))

plt.text(
        0.5, -0.03, plt_text, transform=ax.transAxes, fontsize=8,
        ha='center', va='top', rotation='horizontal', linespacing=1.5)

fig.subplots_adjust(left=0.01, right=0.99, bottom=0.9/(6.6+0.9), top=0.99)
fig.savefig(opng)
plt.close()

end_time = time.perf_counter()
print(f"Execution time: {end_time - start_time:.6f} seconds")



'''
#-------------------------------- projection using pyproj
import pyproj
# import rioxarray as rxr
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

rgb = np.clip(rgb, 0, 1)
rgb **= (1 / 2.2)

rgb = np.dstack([channels['B03'], channels['B02'], channels['B01']])


# change projection
'''
# endregion
