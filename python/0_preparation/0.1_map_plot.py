
# qsub -I -q normal -l walltime=02:00:00,ncpus=1,mem=192GB,storage=gdata/v46+gdata/rr1

# region import packages

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
from scipy import stats
import pandas as pd
from metpy.interpolate import cross_section
from statsmodels.stats import multitest
from metpy.calc import pressure_to_height_std, geopotential_to_height
from metpy.units import units
import metpy.calc as mpcalc
import requests
import rioxarray

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
import cartopy.crs as ccrs
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rcParams['figure.dpi'] = 300
mpl.rc('font', family='Times New Roman', size=10)
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})
import matplotlib.animation as animation
import seaborn as sns
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.ticker import AutoMinorLocator
import geopandas as gpd
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# management
import os
import sys  # print(sys.path)
sys.path.append(os.getcwd() + '/code/gbr_future/module')

# self defined
from mapplot import (
    globe_plot,
    regional_plot,
    ticks_labels,
    scale_bar,
    plot_maxmin_points,
    remove_trailing_zero,
    remove_trailing_zero_pos,
    )

from namelist import (
    month,
    monthini,
    seasons,
    seconds_per_d,
    zerok,
    panel_labels,
    )

from component_plot import (
    rainbow_text,
    change_snsbar_width,
    cplot_wind_vectors,
    cplot_lon180,
    cplot_lon180_ctr,
    plt_mesh_pars,
    plot_loc,
    draw_polygon,
)


# endregion


# region plot the globe

# igra2_station = pd.read_fwf(
#     'https://www1.ncdc.noaa.gov/pub/data/igra/igra2-station-list.txt',
#     names=['id', 'lat', 'lon', 'altitude', 'name', 'starty', 'endy', 'count'])
# fig, ax = globe_plot(figsize=np.array([17.6, 8.8]) / 2.54)
# plot_loc(igra2_station['lon'], igra2_station['lat'], ax, s=6,lw=0.6)
# fig.savefig('figures/0_gbr/0.1_study region/0.0.0_IGRA2 global distribution.png')


'''
# subset = igra2_station.loc[[sid.startswith('AS') for sid in igra2_station['id']]]


ax.add_feature(
    cfeature.LAND, color='green', zorder=2, edgecolor=None,lw=0)
ax.add_feature(
    cfeature.OCEAN, color='blue', zorder=2, edgecolor=None,lw=0)
'''
# endregion


# region plot Australia

# dem_data = rioxarray.open_rasterio('/g/data/rr1/Elevation/1secSRTM_DEMs_v1.0/DEM/Mosaic/dem1sv1_0')
dem_data = rioxarray.open_rasterio('/g/data/rr1/Elevation/3sec_SRTM_DEMsv1.0/DEMS_ESRI_GRID_32bit_Float/dems3sv1_0')
dem_data = dem_data.sel(x=slice(140, 155), y=slice(-10, -25)).squeeze()
dem_data = dem_data.where((dem_data!=-3.4028235e+38)&(dem_data!=0), np.nan)
dem_data = dem_data.astype("float16")
# dem_data = dem_data.coarsen(x=8, y=8, boundary="trim").mean()

bathy_data = rioxarray.open_rasterio('data/others/AusBathyTopo (Australia) 2024 250m/AusBathyTopo__Australia__2024_250m_MSL_cog.tif')
bathy_data = bathy_data.sel(x=slice(140, 155), y=slice(-10, -25)).squeeze()
bathy_data = bathy_data.where(bathy_data<0, np.nan)

gbr_shp = gpd.read_file('data/others/Great_Barrier_Reef_Marine_Park_Boundary/Great_Barrier_Reef_Marine_Park_Boundary.shp')

pltlevel = np.concatenate((np.arange(-6000, 0, 250), np.arange(0, 1201, 50)))
pltticks = np.concatenate((np.arange(-6000, 0, 1000), np.arange(0, 1201, 200)))
colors = np.vstack((
    plt.cm.Blues_r(np.linspace(0, 1, 256)),
    plt.cm.Greens(np.linspace(0, 1, 256))))
pltcmp = LinearSegmentedColormap.from_list("Combined",colors,N=len(pltlevel)-1)
pltnorm = TwoSlopeNorm(vmin=-6000, vcenter=0, vmax=1200)

fig, ax = regional_plot(
    extent=[140, 155, -25, -10], figsize = np.array([13.2, 14.8]) / 2.54,
    ticks_and_labels=True, fontsize=10, xmajortick_int=5, ymajortick_int=5,)

plt_mesh1 = ax.pcolormesh(
    dem_data.x,
    dem_data.y,
    dem_data.values,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(), rasterized=True)
plt_mesh2 = ax.pcolormesh(
    bathy_data.x,
    bathy_data.y,
    bathy_data.values,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(), rasterized=True)

gbr_shp.plot(ax=ax, edgecolor='tab:blue', facecolor='none', lw=0.8, zorder=2)
for iregion in range(4):
    # iregion = 0
    geo_em_ds = xr.open_dataset(f'data/sim/wrf/20160427_4nests/geo_em.d0{iregion+1}.nc')
    draw_polygon(ax, np.min(geo_em_ds.CLAT), np.max(geo_em_ds.CLAT),
                 np.min(geo_em_ds.CLONG), np.max(geo_em_ds.CLONG),)

cbar = fig.colorbar(
    plt_mesh1, # cm.ScalarMappable(norm=pltnorm, cmap=pltcmp),
    ax=ax, aspect=30,
    orientation="horizontal", shrink=1.05, ticks=pltticks, extend='both',
    pad=0.06, fraction=0.06,)
cbar.ax.set_xlabel('Topography and bathymetry [$m$]')
fig.subplots_adjust(left=0.07, right=0.97, bottom=0.06, top=0.98)
fig.savefig('figures/0_gbr/0.1_study region/0.0_gbr.png')




'''
# ax.add_feature(cfeature.OCEAN, color='white', edgecolor=None,lw=0, zorder=2)

# dem_data.to_netcdf('data/others/test/test.nc')
# bathy_data.to_netcdf('data/others/test/test1.nc')

era5_z = xr.open_dataset('/g/data/rt52/era5/single-levels/reanalysis/z/2016/z_era5_oper_sfc_20160101-20160131.nc')
plt_mesh = ax.pcolormesh(
    era5_z.longitude,
    era5_z.latitude,
    era5_z.z[0].values / 9.80665,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

# pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
#     cm_min=-5000, cm_max=1200, cm_interval1=100, cm_interval2=200, cmap='BrBG',
#     reversed=True, asymmetric=True,)

WillisIsland_loc={'lat':-16.2876,'lon':149.962}
plot_loc(WillisIsland_loc['lon'], WillisIsland_loc['lat'], ax)


WillisIsland_loc={'lat':-16.3,'lon':149.98}

lats = [-10.7, -24.5]
lons = [145, 154]
plot_loc(lons[0], lats[0], ax)
plot_loc(lons[1], lats[1], ax)

ax.scatter(
    x = WillisIsland_loc['lon'], y = WillisIsland_loc['lat'],
    s=10, c='none', lw=0.8, marker='o', edgecolors='tab:blue', zorder=2,
    transform=ccrs.PlateCarree(),)

'''
# endregion

