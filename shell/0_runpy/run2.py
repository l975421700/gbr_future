

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
import matplotlib.animation as animation

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
time_series=time_series[:24]

ioption='true_color'
# ioption='night_microphysics'
# ioption='true_color_and_night_microphysics'
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
        rgb = np.where((luminance > 0.05)[..., None], rgb1, rgb2)
    
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
