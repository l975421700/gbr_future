


# region import packages

# data analysis
import numpy as np
from netCDF4 import Dataset
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
from pathlib import Path

# endregion


# region plot himawari true color and night microphysics


year, month, day, hour, minute = 2024, 6, 1, 0, 0
dfolder = Path('/g/data/ra22/satellite-products/arc/obs/himawari-ahi/fldk/latest')
# dfolder = Path('/g/data/ra22/satellite-products/nrt/obs/himawari-ahi/fldk/latest')

opng = Path(f'trial.png')
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

luminance = 0.2126 * rgb1[..., 0] + 0.7152 * rgb1[..., 1] + 0.0722 * rgb1[..., 2]
rgb = np.where((luminance > 0.15)[..., None], rgb1, rgb2)


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
gl.xlocator = mticker.FixedLocator(np.arange(-180, 180 + 1e-4, 10))
gl.ylocator = mticker.FixedLocator(np.arange(-90, 90 + 1e-4, 10))
plt.text(0.5, -0.03, plt_text, transform=ax.transAxes, fontsize=8,
         ha='center', va='top', rotation='horizontal', linespacing=1.5)

fig.subplots_adjust(left=0.01, right=0.99, bottom=1/(7+1), top=0.99)
fig.savefig(opng, transparent=True)
plt.close()




# endregion


