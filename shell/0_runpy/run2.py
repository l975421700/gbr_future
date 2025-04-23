

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
import pickle
from datetime import datetime
from skimage.measure import block_reduce
from netCDF4 import Dataset

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
from matplotlib import cm
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=10)
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})
import matplotlib.animation as animation
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.ticker import AutoMinorLocator
import geopandas as gpd
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import cartopy.feature as cfeature
from PIL import Image
from matplotlib.colors import ListedColormap

# management
import os
import sys  # print(sys.path)
sys.path.append(os.getcwd() + '/code/gbr_future/module')
import psutil
process = psutil.Process()
# print(process.memory_info().rss / 2**30)
import string
import warnings
warnings.filterwarnings('ignore')
import glob
import argparse
import calendar
from pathlib import Path
import time

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
    month_num,
    monthini,
    seasons,
    seconds_per_d,
    zerok,
    panel_labels,
    era5_varlabels,
    cmip6_era5_var,
    ds_color,
    )

from component_plot import (
    rainbow_text,
    change_snsbar_width,
    cplot_wind_vectors,
    cplot_lon180,
    cplot_lon180_ctr,
    plt_mesh_pars,
)

from calculations import (
    time_weighted_mean,
    mon_sea_ann,
    regrid,
    cdo_regrid,)

from statistics0 import (
    ttest_fdr_control,)

# endregion


# region animate cloud liquid/ice water and himawari images


dss = ['Himawari', 'ERA5', 'BARRA-R2', 'BARRA-C2']
extent = [110.58, 157.34, -43.69, -7.01]
min_lon, max_lon, min_lat, max_lat = extent
year, month = 2020, 7
start_day = pd.Timestamp(year, month, 1, 0, 0)
end_day = start_day + pd.Timedelta(days=calendar.monthrange(year, month)[1])
max_value = 0.5
pltlevel = np.arange(0, max_value + 1e-4, 0.005)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
colors = np.ones((len(pltlevel)-1, 4))
colors[:, 3] = np.linspace(0, 1, len(pltlevel)-1)
pltcmp = ListedColormap(colors)
omp4 = f'figures/4_um/4.0_barra/4.0.2_cloud image/4.0.2.1 Himawari, ERA5, BARRA-R2, and BARRA-C2 total column cloud liquid and ice water {year}-{month:02d}.mp4'
img = Image.open(f'data/others/Blue_Marble/world.2004{month:02d}.3x5400x2700.jpg')
dfolder = Path('/g/data/ra22/satellite-products/arc/obs/himawari-ahi/fldk/latest')
# dfolder = Path('/g/data/ra22/satellite-products/nrt/obs/himawari-ahi/fldk/latest')
nrow=1
ncol=len(dss)
bands=['B03', 'B02', 'B01', 'B07', 'B13', 'B15']
time_series = pd.date_range(start=start_day, end=end_day, freq='1h')[:-1]


geolocation_all = pd.read_pickle(f'scratch/data/obs/CloudSat_CALIPSO/2B-CLDCLASS-LIDAR.P1_R05/geolocation_all.pkl')
geolocation_all = geolocation_all.loc[((geolocation_all.date_time >= start_day) & (geolocation_all.date_time <= end_day) & (geolocation_all.lon >= min_lon) & (geolocation_all.lon <= max_lon) & (geolocation_all.lat >= min_lat) & (geolocation_all.lat <= max_lat))]

clwvi = {}
clivi = {}
# prw   = {}

clwvi['BARRA-C2'] = xr.open_dataset(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/clwvi/latest/clwvi_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc').clwvi.sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))
clivi['BARRA-C2'] = xr.open_dataset(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/clivi/latest/clivi_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc').clivi.sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))
# prw['BARRA-C2'] = xr.open_dataset(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/prw/latest/prw_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc').prw.sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))

clwvi['BARRA-R2'] = xr.open_dataset(f'/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/clwvi/latest/clwvi_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc').clwvi.sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))
clivi['BARRA-R2'] = xr.open_dataset(f'/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/clivi/latest/clivi_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc').clivi.sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))
# prw['BARRA-R2'] = xr.open_dataset(f'/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/prw/latest/prw_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1_1hr_{year}{month:02d}-{year}{month:02d}.nc').prw.sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))

clwvi['ERA5'] = xr.open_dataset(glob.glob(f'/g/data/rt52/era5/single-levels/reanalysis/tclw/{year}/tclw_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}??.nc')[0]).rename({'latitude': 'lat', 'longitude': 'lon'}).tclw.sel(lon=slice(min_lon, max_lon), lat=slice(max_lat, min_lat))
clivi['ERA5'] = xr.open_dataset(glob.glob(f'/g/data/rt52/era5/single-levels/reanalysis/tciw/{year}/tciw_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}??.nc')[0]).rename({'latitude': 'lat', 'longitude': 'lon'}).tciw.sel(lon=slice(min_lon, max_lon), lat=slice(max_lat, min_lat))
# prw['ERA5'] = xr.open_dataset(glob.glob(f'/g/data/rt52/era5/single-levels/reanalysis/tcwv/{year}/tcwv_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}??.nc')[0]).rename({'latitude': 'lat', 'longitude': 'lon'}).tcwv.sel(lon=slice(min_lon, max_lon), lat=slice(max_lat, min_lat))


fm_bottom=1.2/(4*nrow+1.2)
fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([4.4*ncol, 4*nrow + 1.2]) / 2.54,
    subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
    gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)

for jcol in range(ncol):
    axs[jcol] = regional_plot(extent=extent, central_longitude=180, ax_org=axs[jcol], lw=0.1, border_color='yellow')
    if jcol!=0:
        axs[jcol].imshow(img, extent=[-180, 180, -90, 90], transform=ccrs.PlateCarree(), zorder=0)
    axs[jcol].text(0, 1.02, f'({string.ascii_lowercase[jcol]}) {dss[jcol]}', ha='left', va='bottom', transform=axs[jcol].transAxes)

fig.text(0.5, 0.01, f'(a) True Color and Night Microphysics RGB and CloudSat-CALIPSO tracks in orange\n(b-d) Cloud liquid&ice water (opaque white: >={max_value} ' + r'[$kg \; m^{-2}$]) above NASA Blue Marble Earth image', ha='center', va='bottom', linespacing=1.3)


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
    print(f'#-------------------------------- {day:02d} {hour:02d}:{minute:02d}')
    
    
    channels = {}
    for iband in bands:
        # iband='B07'
        print(f'#---------------- {iband}')
        ifile = sorted(Path(dfolder/f'{year}/{month:02d}/{day:02d}/{hour:02d}{minute:02d}').glob(f'*OBS_{iband}*'))
        if not ifile:
            print('Warning: No file found')
            continue
        
        channels[iband] = Dataset(ifile[-1], 'r')
        var_name = next(var for var in channels[iband].variables if var.startswith("channel_00"))
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
    
    
    plt_text = fig.text(0.5, fm_bottom+0.02, f'{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d} UTC', ha='center', va='top', linespacing=1.3)
    
    start_time = pd.Timestamp(year, month, day, hour, minute)
    end_time = start_time + pd.Timedelta(minutes=60)
    geolocation_subset = geolocation_all.loc[((geolocation_all.date_time >= start_time) & (geolocation_all.date_time <= end_time))]
    for jcol in range(ncol):
        plt_scatter = axs[jcol].scatter(geolocation_subset.lon, geolocation_subset.lat, s=1, c='tab:orange', lw=0, transform=ccrs.PlateCarree(), zorder=2)
    
    plt_imshow = axs[0].imshow(rgb, extent=[-5499500., 5499500., -5499500., 5499500.], transform=ccrs.Geostationary(central_longitude=140.7, satellite_height=35785863.0), interpolation='none', origin='upper', resample=False)
    
    plt_meshs = []
    for jcol in range(ncol-1):
        # jcol=0
        plt_meshs.append(axs[jcol+1].pcolormesh(
            clwvi[dss[jcol+1]].lon,
            clwvi[dss[jcol+1]].lat,
            clwvi[dss[jcol+1]].sel(time=start_time) + clivi[dss[jcol+1]].sel(time=start_time),
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree()))
    
    plt_objs = [plt_scatter, plt_text, plt_imshow, ] + [plt_meshs[jcol] for jcol in range(ncol-1)]
    del channels
    time2 = time.perf_counter()
    print(f'Execution time: {time2 - time1:.1f} seconds')
    return(plt_objs)

fig.subplots_adjust(left=0.005, right=0.995, bottom=fm_bottom, top=0.95)
ani = animation.FuncAnimation(
    fig, update_frames, frames=len(time_series), interval=500, blit=False)
if os.path.exists(omp4): os.remove(omp4)
ani.save(omp4, progress_callback=lambda iframe, n: print(f'Frame {iframe}/{n}'))



# endregion

