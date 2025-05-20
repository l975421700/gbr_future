

# region import packages

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
from datetime import datetime

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=10)
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})
from PIL import Image
from matplotlib.colors import ListedColormap

# management
import os
import sys  # print(sys.path)
sys.path.append(os.getcwd() + '/code/gbr_future/module')
import psutil
process = psutil.Process()
# print(process.memory_info().rss / 2**30)
import warnings
warnings.filterwarnings('ignore')
import glob


def ticks_labels(
    xmin, xmax, ymin, ymax, xspacing, yspacing
    ):
    '''
    # input ----
    xmin, xmax, ymin, ymax: range of labels
    xspacing: spacing of x ticks
    yspacing: spacing of y ticks
    
    # output ----
    xticks_pos, xticks_label, yticks_pos, yticks_label
    '''
    
    import numpy as np
    
    # get the x ticks
    xticks_pos = np.arange(xmin, xmax + 1e-4, xspacing)
    if not isinstance(xspacing, int):
        xticks_pos = np.around(xticks_pos, 1)
    else:
        xticks_pos = xticks_pos.astype('int')
    
    # Associate with '° W', '°', and '° E'
    xticks_label = [''] * len(xticks_pos)
    for i in np.arange(len(xticks_pos)):
        if (xticks_pos[i] > 180):
            xticks_pos[i] = xticks_pos[i] - 360
        
        if (abs(xticks_pos[i]) == 180) | (xticks_pos[i] == 0):
            xticks_label[i] = str(abs(xticks_pos[i])) + '°'
        elif xticks_pos[i] < 0:
            xticks_label[i] = str(abs(xticks_pos[i])) + '° W'
        elif xticks_pos[i] > 0:
            xticks_label[i] = str(xticks_pos[i]) + '° E'
    
    # get the y ticks
    yticks_pos = np.arange(ymin, ymax + 1e-4, yspacing)
    if not isinstance(yspacing, int):
        yticks_pos = np.around(yticks_pos, 1)
    else:
        yticks_pos = yticks_pos.astype('int')
    
    # Associate with '° N', '°', and '° S'
    yticks_label = [''] * len(yticks_pos)
    for i in np.arange(len(yticks_pos)):
        if yticks_pos[i] < 0:
            yticks_label[i] = str(abs(yticks_pos[i])) + '° S'
        if yticks_pos[i] == 0:
            yticks_label[i] = str(yticks_pos[i]) + '°'
        if yticks_pos[i] > 0:
            yticks_label[i] = str(yticks_pos[i]) + '° N'
    
    return xticks_pos, xticks_label, yticks_pos, yticks_label

def regional_plot(
    extent=None,
    figsize=None,
    central_longitude = 0,
    xmajortick_int = 10, ymajortick_int = 10,
    xminortick_int = 5, yminortick_int = 5,
    lw=0.25, country_boundaries=True, border_color = 'black',
    grid_color = 'gray', lgrid=True,
    set_figure_margin = False, figure_margin=None,
    ticks_and_labels = False,
    ax_org=None, fontsize=10,
    ):
    '''
    ----Input
    ----output
    '''
    
    import numpy as np
    import cartopy.feature as cfeature
    import cartopy as ctp
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rc('font', family='Times New Roman', size=fontsize)
    
    ticklabel=ticks_labels(extent[0], extent[1], extent[2], extent[3],
                           xmajortick_int, ymajortick_int)
    xminorticks = np.arange(extent[0], extent[1] + 1e-4, xminortick_int)
    yminorticks = np.arange(extent[2], extent[3] + 1e-4, yminortick_int)
    transform = ctp.crs.PlateCarree(central_longitude=central_longitude)
    
    if (figsize is None):
        figsize = np.array([8.8, 8.8]) / 2.54
    
    if (ax_org is None):
        fig, ax = plt.subplots(
            1, 1, figsize=figsize, subplot_kw={'projection': transform},)
    else:
        ax = ax_org
    
    ax.set_extent(extent, crs = ctp.crs.PlateCarree())
    
    if ticks_and_labels:
        ax.set_xticks(ticklabel[0], crs = ctp.crs.PlateCarree())
        ax.set_xticklabels(ticklabel[1])
        ax.set_yticks(ticklabel[2])
        ax.set_yticklabels(ticklabel[3])
        ax.tick_params(length=2)
    
    if country_boundaries:
        coastline = cfeature.NaturalEarthFeature(
            'physical', 'coastline', '10m', edgecolor=border_color,
            facecolor='none', lw=lw)
        ax.add_feature(coastline, zorder=2)
        borders = cfeature.NaturalEarthFeature(
            'cultural', 'admin_0_boundary_lines_land', '10m',
            edgecolor=border_color,
            facecolor='none', lw=lw)
        ax.add_feature(borders, zorder=2)
    
    if (central_longitude == 0) & lgrid:
        ax.gridlines(
            crs=ctp.crs.PlateCarree(central_longitude=central_longitude),
            linewidth=lw, zorder=2,
            color=grid_color, alpha=0.5, linestyle='--',
            xlocs = xminorticks, ylocs=yminorticks,
            )
    elif lgrid:
        ax.gridlines(
            crs=ctp.crs.PlateCarree(central_longitude=central_longitude),
            linewidth=lw, zorder=2,
            color=grid_color, alpha=0.5, linestyle='--',
            )
    
    if set_figure_margin & (not(figure_margin is None)) & (ax_org is None):
        fig.subplots_adjust(
            left=figure_margin['left'], right=figure_margin['right'],
            bottom=figure_margin['bottom'], top=figure_margin['top'])
    elif (ax_org is None):
        fig.tight_layout()
    
    if (ax_org is None):
        return fig, ax
    else:
        return ax

# endregion


# region plot cloud liquid/ice water

extent = [110.58, 157.34, -43.69, -7.01]
min_lon, max_lon, min_lat, max_lat = extent
year, month, day, hour, minute = 2020, 6, 1, 0, 0
ids = 'ERA5' # 'BARRA-R2' # 'BARRA-C2' #

if ids == 'BARRA-C2':
    clwvi = xr.open_dataset(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/clwvi/latest/clwvi_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc').clwvi.sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))
    clivi = xr.open_dataset(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/clivi/latest/clivi_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc').clivi.sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))
    # prw = xr.open_dataset(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/prw/latest/prw_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc').prw.sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))
elif ids == 'BARRA-R2':
    clwvi = xr.open_dataset(f'/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/clwvi/latest/clwvi_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc').clwvi.sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))
    clivi = xr.open_dataset(f'/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/clivi/latest/clivi_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc').clivi.sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))
    # prw = xr.open_dataset(f'/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/prw/latest/prw_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc').prw.sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))
elif ids == 'ERA5':
    clwvi = xr.open_dataset(glob.glob(f'/g/data/rt52/era5/single-levels/reanalysis/tclw/{year}/tclw_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}??.nc')[0]).rename({'latitude': 'lat', 'longitude': 'lon'}).tclw.sel(lon=slice(min_lon, max_lon), lat=slice(max_lat, min_lat))
    clivi = xr.open_dataset(glob.glob(f'/g/data/rt52/era5/single-levels/reanalysis/tciw/{year}/tciw_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}??.nc')[0]).rename({'latitude': 'lat', 'longitude': 'lon'}).tciw.sel(lon=slice(min_lon, max_lon), lat=slice(max_lat, min_lat))
    # prw = xr.open_dataset(glob.glob(f'/g/data/rt52/era5/single-levels/reanalysis/tcwv/{year}/tcwv_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}??.nc')[0]).rename({'latitude': 'lat', 'longitude': 'lon'}).tcwv.sel(lon=slice(min_lon, max_lon), lat=slice(max_lat, min_lat))


# colorbar for clouds
max_value = 0.5
pltlevel = np.arange(0, max_value + 1e-4, 0.005)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
colors = np.ones((len(pltlevel)-1, 4))
colors[:, 3] = np.linspace(0, 1, len(pltlevel)-1)
pltcmp = ListedColormap(colors)


opng = f'figures/test.png'
fig, ax = regional_plot(extent=extent, central_longitude=180,
                        figsize = np.array([8.8, 7.8]) / 2.54)

ax.pcolormesh(
    clwvi.lon,
    clwvi.lat,
    clwvi.sel(time=datetime(year, month, day, hour, minute)) + clivi.sel(time=datetime(year, month, day, hour, minute)),
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree())
ax.text(0.5, -0.02, f'{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d} UTC\nCloud liquid&ice water in {ids} (white: >{max_value} ' + r'[$kg \; m^{-2}$])',
        ha='center', va='top', transform=ax.transAxes, linespacing=1.3)

img = Image.open(f'data/others/Blue Marble Next Generation w: Topography and Bathymetry/world.topo.bathy.2004{month:02d}.3x5400x2700.jpg')
ax.imshow(img, extent=[-180, 180, -90, 90], transform=ccrs.PlateCarree(), zorder=0)

fig.subplots_adjust(left=0.01, right=0.99, bottom=0.12, top=0.99)
fig.savefig(opng)


# endregion


