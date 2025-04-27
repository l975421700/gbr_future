

# qsub -I -q express -l walltime=2:00:00,ncpus=1,mem=192GB,storage=gdata/v46+gdata/rr1


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
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=10)
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})
import matplotlib.animation as animation
# import seaborn as sns
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.ticker import AutoMinorLocator
import geopandas as gpd
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

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
    month_jan,
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


# fig, ax = globe_plot(figsize=np.array([17.6, 8.8]) / 2.54)
# # plot igra2 locations
# igra2_station = pd.read_fwf(
#     'https://www1.ncdc.noaa.gov/pub/data/igra/igra2-station-list.txt',
#     names=['id', 'lat', 'lon', 'altitude', 'name', 'starty', 'endy', 'count'])
# plot_loc(igra2_station['lon'], igra2_station['lat'], ax, s=6,lw=0.6)
# opng='figures/0_gbr/0.1_study region/0.0.0_IGRA2 global distribution.png'

fig, ax = globe_plot(figsize=np.array([12, 8]) / 2.54, fm_bottom=0.13)

era5_z = xr.open_dataset('/g/data/rt52/era5/single-levels/reanalysis/z/2023/z_era5_oper_sfc_20230101-20230131.nc')['z'][0] / 9.80665
pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=5600, cm_interval1=200, cm_interval2=800, cmap='Greens_r',)
opng='figures/0_gbr/0.1_study region/0.0.0_global era5 z.png'
plt_mesh1 = ax.pcolormesh(
    era5_z.longitude,
    era5_z.latitude,
    era5_z.values,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree())
ax.add_feature(cfeature.OCEAN,color='white',zorder=1,edgecolor=None,lw=0)
cbar = fig.colorbar(
    plt_mesh1, ax=ax, aspect=40, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.8, ticks=pltticks, extend='max',
    pad=0.02, fraction=0.13,)
cbar.ax.set_xlabel('Topography in ERA5 [$m$]', ha='center', labelpad=4)


fig.savefig(opng)


'''
# subset = igra2_station.loc[[sid.startswith('AS') for sid in igra2_station['id']]]


ax.add_feature(
    cfeature.LAND, color='green', zorder=2, edgecolor=None,lw=0)
ax.add_feature(
    cfeature.OCEAN, color='blue', zorder=2, edgecolor=None,lw=0)
'''
# endregion


# region plot GBR [140, 155, -25, -10]

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
# for iregion in range(4):
#     # iregion = 0
#     geo_em_ds = xr.open_dataset(f'data/sim/wrf/20160427_4nests/geo_em.d0{iregion+1}.nc')
#     draw_polygon(ax, np.min(geo_em_ds.CLAT), np.max(geo_em_ds.CLAT),
#                  np.min(geo_em_ds.CLONG), np.max(geo_em_ds.CLONG),)

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


# region plot Australia [108, 160, -45.7, -5]

fig, ax = regional_plot(
    extent=[108, 160, -45.7, -5],
    figsize = np.array([8.8, 8.1])/2.54, lgrid=False, lw=0.15)

ax.set_xticks(np.arange(110, 160+1e-4, 10))
ax.set_yticks(np.arange(-40, -10+1e-4, 10))
ax.xaxis.set_major_formatter(LongitudeFormatter(degree_symbol='° '))
ax.yaxis.set_major_formatter(LatitudeFormatter(degree_symbol='° '))
ax.gridlines(
    crs=ccrs.PlateCarree(), linewidth=0.15, zorder=2,
    color='gray', alpha=0.5, linestyle='--',)

# # plot dem and bathy from Aus
# pltlevel = np.concatenate((np.arange(-6000, 0, 250), np.arange(0, 1201, 50)))
# pltticks = np.concatenate((np.arange(-6000, 0, 1000), np.arange(0, 1201, 200)))
# colors = np.vstack((
#     plt.cm.Blues_r(np.linspace(0, 1, 256)),
#     plt.cm.Greens(np.linspace(0, 1, 256))))
# pltcmp = LinearSegmentedColormap.from_list("Combined",colors,N=len(pltlevel)-1)
# pltnorm = TwoSlopeNorm(vmin=-6000, vcenter=0, vmax=1200)

# dem_data = rioxarray.open_rasterio('/g/data/rr1/Elevation/3sec_SRTM_DEMsv1.0/DEMS_ESRI_GRID_32bit_Float/dems3sv1_0').squeeze()
# dem_data = dem_data.where((dem_data!=-3.4028235e+38)&(dem_data!=0), np.nan)
# dem_data = dem_data.coarsen(x=10, y=10, boundary="trim").mean()
# plt_mesh1 = ax.pcolormesh(
#     dem_data.x,
#     dem_data.y,
#     dem_data.values,
#     norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree())
# bathy_data = rioxarray.open_rasterio('data/others/AusBathyTopo (Australia) 2024 250m/AusBathyTopo__Australia__2024_250m_MSL_cog.tif')
# bathy_data = bathy_data.sel(x=slice(108, 160), y=slice(-5, -45.7)).squeeze()
# bathy_data = bathy_data.where(bathy_data<0, np.nan)
# bathy_data = bathy_data.coarsen(x=4, y=4, boundary="trim").mean()
# plt_mesh2 = ax.pcolormesh(
#     bathy_data.x,
#     bathy_data.y,
#     bathy_data.values,
#     norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree())
# extend='both'
# cbar_label='Topography and bathymetry [$m$]'
# opng='figures/0_gbr/0.1_study region/0.0_Australia.png'

# plot orography from BARRA-C2
pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=2800, cm_interval1=100, cm_interval2=400, cmap='Greens_r',)
orog = xr.open_dataset('/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/fx/orog/latest/orog_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1.nc').orog
orog = orog.where(orog!=0, np.nan)
plt_mesh1 = ax.pcolormesh(
    orog.lon,
    orog.lat,
    orog.values,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree())
extend='max'
cbar_label='Topography in BARRA-C2 [$m$]'

min_lon, max_lon, min_lat, max_lat = [
    orog.lon.values[int(len(orog.lon)*0.05)],
    orog.lon.values[int(len(orog.lon)*0.95)],
    orog.lat.values[int(len(orog.lat)*0.05)],
    orog.lat.values[int(len(orog.lat)*0.95)]]
# min_lon, max_lon, min_lat, max_lat = [110.58, 157.34, -43.69, -7.01]
# orog.sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))
rec_m = ax.add_patch(Rectangle(
    (min_lon, min_lat), max_lon-min_lon, max_lat-min_lat,
    ec = 'red', color = 'None', lw = 1))


# Plot Cross section
CS_A1 = [min_lon, min_lat]
CS_A2 = [max_lon, max_lat]
CS_A3 = [max_lon, min_lat]
CS_A4 = [min_lon, max_lat]

ax.plot([CS_A1[0], CS_A2[0]], [CS_A1[1], CS_A2[1]], 'o-',
        color='tab:orange', lw=1, ms=4, transform=ccrs.PlateCarree())
ax.plot([CS_A3[0], CS_A4[0]], [CS_A3[1], CS_A4[1]], 'o-',
        color='tab:orange', lw=1, ms=4, transform=ccrs.PlateCarree())
ax.text(CS_A1[0]+1, CS_A1[1]+1, 'A1', ha='left', va='bottom')
ax.text(CS_A2[0]-1, CS_A2[1]-1, 'A2', ha='right', va='top')
ax.text(CS_A3[0]-1, CS_A3[1]+1, 'A3', ha='right', va='bottom')
ax.text(CS_A4[0]+1, CS_A4[1]-1, 'A4', ha='left', va='top')

opng='figures/0_gbr/0.1_study region/0.0_Australia1.png'

gbr_shp = gpd.read_file('data/others/Great_Barrier_Reef_Marine_Park_Boundary/Great_Barrier_Reef_Marine_Park_Boundary.shp')
gbr_shp.plot(ax=ax, edgecolor='tab:blue', facecolor='none', lw=1, zorder=2)

cbar = fig.colorbar(
    plt_mesh1,
    ax=ax, aspect=30,
    orientation="horizontal", shrink=1.05, ticks=pltticks, extend=extend,
    pad=0.11, fraction=0.07,)
cbar.ax.set_xlabel(cbar_label)
fig.subplots_adjust(left=0.12, right=0.94, bottom=0.12, top=0.995)
fig.savefig(opng)




'''
'''
# endregion


# region plot the Blue Marble

month=6
# ifile = f'data/others/Blue Marble Next Generation w: Topography and Bathymetry/world.topo.bathy.2004{month:02d}.3x21600x10800.jpg'
ifile = f'data/others/Blue Marble Next Generation w: Topography and Bathymetry/world.topo.bathy.2004{month:02d}.3x5400x2700.jpg'
img = Image.open(ifile)

opng = 'figures/test.png'
fig, ax = globe_plot(figsize=np.array([12, 8]) / 2.54, fm_bottom=0.13)
ax.imshow(img, extent=[-180, 180, -90, 90], transform=ccrs.PlateCarree())
fig.savefig(opng)


# endregion

