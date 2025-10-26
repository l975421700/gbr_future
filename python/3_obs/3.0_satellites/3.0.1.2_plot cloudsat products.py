

# qsub -I -q normal -l walltime=3:00:00,ncpus=1,mem=96GB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+gdata/zv2+gdata/ra22+gdata/gx60


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
from datetime import datetime, timedelta
import glob
from pyhdf.SD import SD, SDC
from pyhdf.HDF import HDF
from pyhdf.VS import VS
import joblib
import argparse

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
from matplotlib.colors import BoundaryNorm, ListedColormap
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

from calculations import (
    find_ilat_ilon,
    )

# endregion


# region check data

year, month, day, hour = 2020, 6, 2, 4
# min_lon, max_lon, min_lat, max_lat = 110.58, 157.34, -43.69, -7.01
# up to Willis Island:
# min_lon, max_lon, min_lat, max_lat = 110.58, 157.34, -16.2876, -7.01
min_lon, max_lon, min_lat, max_lat = 110.58, 157.34, -12, -7
max_altitude = 6000

cs_info = {
    '2B-CLDCLASS-LIDAR.P1_R05': {},
    '2B-CLDCLASS.P1_R05': {},
    '2B-CWC-RO.P1_R05': {
        'RO_liq_effective_radius': {
            'factor': 10,   'offset': 0,    'valid_range': [0, 10000],
            'label': r'liquid effective radius [$um$]'},
        'RO_ice_effective_radius': {
            'factor': 10,   'offset': 0,    'valid_range': [0, 30000],
            'label': r'ice effective radius [$um$]'},
        'RO_liq_number_conc': {
            'factor': 10,   'offset': 0,    'valid_range': [0, 30000],
            'label': r'CDNC [$cm^{-3}$]'},
        'RO_ice_number_conc': {
            'factor': 10,   'offset': 0,    'valid_range': [0, 30000],
            'label': r'ice number concentration [$cm^{-3}$]'},
        'RO_liq_distrib_width_param': {
            'factor': 1000,   'offset': 0,    'valid_range': [0, 5000],
            'label': r'liquid distribution width parameter [$-$]'},
        'RO_ice_distrib_width_param': {
            'factor': 1000,   'offset': 0,    'valid_range': [0, 5000],
            'label': r'ice distribution width parameter [$-$]'},
        'RO_liq_water_content': {
            'factor': 1,    'offset': 0,    'valid_range': [0.0, 15000.0],
            'label': r'liquid water content [$mg \; m^{-3}$]'},
        'RO_ice_water_content': {
            'factor': 1,    'offset': 0,    'valid_range': [0.0, 10000.0],
            'label': r'ice water content [$mg \; m^{-3}$]'},
        'RO_ice_phase_fraction': {
            'factor': 1000, 'offset': 0,    'valid_range': [0, 1000],
            'label': r'ice phase fraction [$\%$]'},
        'LO_RO_effective_radius': {
            'factor': 10,   'offset': 0,    'valid_range': [0, 10000],
            'label': r'liquid-only effective radius [$um$]'},
        'LO_RO_number_conc': {
            'factor': 10,   'offset': 0,    'valid_range': [0, 30000],
            'label': r'liquid-only number concentration [$cm^{-3}$]'},
        'LO_RO_distrib_width_param': {
            'factor': 1000,   'offset': 0,    'valid_range': [0, 5000],
            'label': r'liquid-only distribution width parameter [$-$]'},
        'LO_RO_liquid_water_content': {
            'factor': 1,    'offset': 0,    'valid_range': [0.0, 15000.0],
            'label': r'liquid-only liquid water content [$mg \; m^{-3}$]'},
        'IO_RO_effective_radius': {
            'factor': 10,   'offset': 0,    'valid_range': [0, 30000],
            'label': r'ice-only effective radius [$um$]'},
        'IO_RO_log_number_conc': {
            'factor': 1000,    'offset': 0, 'valid_range': [-3000, 5000],
            'label': r'ice-only log(number concentration) [$log(L^{-1})$]'},
        'IO_RO_distrib_width_param': {
            'factor': 1000,   'offset': 0,    'valid_range': [0, 5000],
            'label': r'ice-only distribution width parameter [$-$]'},
        'IO_RO_ice_water_content': {
            'factor': 1,    'offset': 0,    'valid_range': [0.0, 10000.0],
            'label': r'ice-only ice water content [$mg \; m^{-3}$]'},
        },
    '2B-GEOPROF-LIDAR.P2_R05': {
        'CloudFraction': {
            'factor': 1, 'offset': 0, 'valid_range': [0, 100],
            'label': r'cloud fraction [$-$]'},
        },
    '2B-GEOPROF.P1_R05': {
        'CPR_Cloud_mask': {
            'factor': 1, 'offset': 0, 'valid_range': [0, 40],
            'label': r'CPR cloud mask [$-$]'},
        'Gaseous_Attenuation':  {
            'factor': 100, 'offset': 0, 'valid_range': [0, 1000],
            'label': r'gaseous attenuation [$dBZe$]'},
        'Radar_Reflectivity':   {
            'factor': 100, 'offset': 0, 'valid_range': [-4000, 5000],
            'label': r'radar reflectivity [$dBZe$]'},
        },
    }


for iproduct in [
    # '2B-CLDCLASS-LIDAR.P1_R05',
    # '2B-CLDCLASS.P1_R05',
    '2B-CWC-RO.P1_R05',
    # '2B-GEOPROF-LIDAR.P2_R05',
    # '2B-GEOPROF.P1_R05',
    ]:
    # iproduct = '2B-CLDCLASS.P1_R05'
    # iproduct = '2B-CWC-RO.P1_R05'
    # iproduct = '2B-GEOPROF.P1_R05'
    print(f'#-------------------------------- {iproduct}')
    
    doy = datetime(year, month, day).timetuple().tm_yday
    ptime = pd.Timestamp(year,month,day,hour) - pd.Timedelta('1h')
    year0, month0, day0, hour0 = ptime.year, ptime.month, ptime.day, ptime.hour
    doy0 = datetime(year0, month0, day0).timetuple().tm_yday
    
    try:
        ifile = glob.glob(f'data/obs/CloudSat_CALIPSO/{iproduct}/{year}/{doy:03d}/{year}{doy}{hour:02d}*.hdf')[0]
        date = datetime(year, month, day)
    except IndexError:
        ifile = glob.glob(f'data/obs/CloudSat_CALIPSO/{iproduct}/{year0}/{doy0:03d}/{year0}{doy0}{hour0:02d}*.hdf')[0]
        date = datetime(year0, month0, day0)
    print(ifile)
    
    hdf_vs = HDF(ifile).vstart()
    hdf_sd = SD(ifile, SDC.READ)
    # print(hdf_vs.vdatainfo())
    # print(hdf_sd.datasets().keys())
    
    height = hdf_sd.select('Height').get().astype(float)
    height[height==-9999] = np.nan
    lat = np.array(hdf_vs.attach('Latitude')[:]).squeeze()
    lon = np.array(hdf_vs.attach('Longitude')[:]).squeeze()
    lat_2d = np.tile(lat[:, np.newaxis], (1, height.shape[1]))
    lon_2d = np.tile(lon[:, np.newaxis], (1, height.shape[1]))
    mask = (lon >= min_lon) & (lon <= max_lon) & \
        (lat >= min_lat) & (lat <= max_lat)
    seconds = hdf_vs.attach('UTC_start')[:][0][0] + \
        np.array(hdf_vs.attach('Profile_time')[:]).squeeze()
    date_time = np.array([date + timedelta(seconds=s) for s in seconds])
    
    for ivar in ['RO_liq_number_conc']:
        # hdf_sd.datasets().keys()
        # ivar = 'CPR_Cloud_mask'
        # ivar = 'Radar_Reflectivity'
        # ivar = 'RO_liq_number_conc'
        # ivar = 'LO_RO_number_conc'
        # print(f'#---------------- {ivar}')
        if ivar in cs_info[iproduct].keys():
            print(f'#---------------- {ivar}')
            # print(cs_info[iproduct][ivar]['valid_range'])
        else:
            continue
        
        if ivar in ['RO_liq_number_conc', 'LO_RO_number_conc']:
            # pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
            #     cm_min=50, cm_max=100, cm_interval1=10, cm_interval2=10, cmap='pink',)
            pltlevel = np.arange(0, 100+1e-4, 10, dtype=np.float64)
            pltticks = np.arange(0, 100+1e-4, 10, dtype=np.float64)
            base_cmap = cm.get_cmap('pink_r', len(pltlevel))
            pltcmp = ListedColormap([base_cmap(i) for i in range(base_cmap.N)][1:])
            pltnorm = BoundaryNorm(pltlevel, ncolors=pltcmp.N, clip=True)
            extend='neither'
        elif ivar == 'RO_ice_number_conc':
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=240, cm_interval1=15, cm_interval2=30,
                cmap='viridis_r',)
            extend = 'max'
        else:
            continue
        
        if str(date_time[mask][0])[:10] == str(date_time[mask][-1])[:10]:
            time_se = f'{str(date_time[mask][0])[:10]} {str(date_time[mask][0])[11:16]} to {str(date_time[mask][-1])[11:16]} UTC'
        else:
            time_se = f'{str(date_time[mask][0])[:16]} to {str(date_time[mask][-1])[:16]} UTC'
        
        opng = f'figures/3_satellites/3.1_CloudSat_CALIPSO/3.1.0_vertical profiles/3.1.0.0 {iproduct} {ivar} {time_se} {min_lon}_{max_lon}_{min_lat}_{max_lat} below {max_altitude} m.png'
        
        cs_data = hdf_sd.select(ivar).get().astype(float)
        cs_data[(cs_data < cs_info[iproduct][ivar]['valid_range'][0]) | \
            (cs_data > cs_info[iproduct][ivar]['valid_range'][1])] = np.nan
        cs_data = (cs_data - cs_info[iproduct][ivar]['offset']) / \
            cs_info[iproduct][ivar]['factor']
        # print(stats.describe(cs_data, axis=None, nan_policy='omit'))
        
        fm_bottom = 0.35
        fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54)
        plt_mesh = ax.pcolormesh(
            lat_2d[mask, :], height[mask, :]/1000, cs_data[mask, :],
            norm=pltnorm, cmap=pltcmp,)
        
        ax.xaxis.set_major_formatter(LatitudeFormatter(degree_symbol='° '))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.set_xlim(-12, -7)
        
        ax.set_ylabel(f'Altitude [$km$]')
        # ax.set_yticks(np.arange(0, max_altitude+1e-4, 2000))
        # ax.set_yticklabels(np.arange(0,max_altitude/1000+1e-4,2).astype(int))
        ax.set_ylim(0, max_altitude/1000)
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        
        ax.grid(True, which='both', lw=0.5, c='gray', alpha=0.5, linestyle='--',)
        
        cbar = fig.colorbar(
            plt_mesh, #cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), #
            format=remove_trailing_zero_pos,
            orientation="horizontal", ticks=pltticks, extend=extend,
            cax=fig.add_axes([0.05, fm_bottom-0.12, 0.9, 0.03]))
        # cbar_label = f'{time_se}\n{cs_info[iproduct][ivar]['label']}\n{iproduct.replace('.P1_R05', '').replace('.P2_R05', '')}'
        cbar_label = f'CloudSat {time_se} {cs_info[iproduct][ivar]['label']}'
        cbar.ax.set_xlabel(cbar_label, linespacing=1.5)
        
        fig.subplots_adjust(left=0.14,right=0.96,bottom=fm_bottom,top=0.97)
        fig.savefig(opng)


aaa = cs_data[((lon_2d >= min_lon) & (lon_2d <= max_lon) & \
        (lat_2d >= min_lat) & (lat_2d <= max_lat) & (height <= max_altitude))].copy()
# aaa = cs_data[mask, :]
aaa[aaa==0] = np.nan
np.nanmin(aaa)
np.nanmax(aaa)
np.nanmean(aaa)


'''
# radar_reflectivity[(cpr_cloud_mask != 30) & (cpr_cloud_mask != 40)] = np.nan


RO_liq_number_conc = hdf_sd.select('RO_liq_number_conc').get().astype(float)
LO_RO_number_conc = hdf_sd.select('LO_RO_number_conc').get().astype(float)

RO_liq_number_conc[RO_liq_number_conc==-8888] = np.nan
RO_liq_number_conc[RO_liq_number_conc==-7777] = np.nan
RO_liq_number_conc[RO_liq_number_conc==-4444] = np.nan
RO_liq_number_conc[RO_liq_number_conc==-3333] = np.nan
RO_liq_number_conc /= 10
LO_RO_number_conc[LO_RO_number_conc==-8888] = np.nan
LO_RO_number_conc[LO_RO_number_conc==-4444] = np.nan
LO_RO_number_conc[LO_RO_number_conc==-3333] = np.nan
LO_RO_number_conc /= 10

cpr_cloud_mask = hdf_sd.select('CPR_Cloud_mask').get().astype(float)
radar_reflectivity = hdf_sd.select('Radar_Reflectivity').get().astype(float)

cpr_cloud_mask[cpr_cloud_mask==-9] = np.nan
radar_reflectivity[radar_reflectivity==-8888] = np.nan
radar_reflectivity[(radar_reflectivity<-4000)|(radar_reflectivity>5000)]=np.nan
radar_reflectivity /= 100
radar_reflectivity[(cpr_cloud_mask != 30) & (cpr_cloud_mask != 40)] = np.nan

# time_2d = np.tile(date_time[:, np.newaxis], (1, height.shape[1]))
# ax.pcolormesh(time_2d[mask, :], height[mask, :], height[mask, :], cmap='viridis')
# ax.pcolormesh(time_2d[mask, :], height[mask, :], cpr_cloud_mask[mask, :], cmap='viridis', shading='nearest')
# ax.pcolormesh(time_2d[mask, :], height[mask, :], radar_reflectivity[mask, :], cmap='viridis', shading='nearest')

stats.describe(height[mask, :], axis=None, nan_policy='omit')
stats.describe(cpr_cloud_mask[mask, :], axis=None, nan_policy='omit')
stats.describe(radar_reflectivity[mask, :], axis=None, nan_policy='omit')

hdf_sd.select('Height').attributes()
hdf_sd.select('Height').dimensions()
hdf_sd.select('Height').getfillvalue()

(height == -9999).sum()
(cpr_cloud_mask == -9).sum()
(radar_reflectivity == -8888).sum()

# check lat/lon
fig, ax = globe_plot()
# ax.scatter(lon, lat, s=2, lw=0, transform=ccrs.PlateCarree())
mask = (lon >= min_lon) & (lon <= max_lon) & (lat >= min_lat) & (lat <= max_lat)
ax.scatter(lon[mask], lat[mask], s=2, lw=0, transform=ccrs.PlateCarree())
fig.savefig('figures/test.png')


'''
# endregion






# region plot tracks of CloudSat-CALIPSO for one day

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=86400-1, cm_interval1=3600, cm_interval2=3600, cmap='viridis',)

fm_bottom=1.5/(6+1.5)
fig, ax = globe_plot(figsize=np.array([12, 6+1.5]) / 2.54, fm_bottom=fm_bottom)

year=2020
doy = 154
date = datetime(year, 1, 1) + timedelta(days=doy - 1)
month = date.month
day = date.day


fl = sorted(glob.glob(f'scratch/data/obs/CloudSat_CALIPSO/2B-CLDCLASS-LIDAR.P1_R05/{year}/{doy}/*'))
for ifile in fl:
    # ifile=fl[0]
    hdf = HDF(ifile).vstart()
    lat = np.array(hdf.attach('Latitude')[:]).squeeze()
    lon = np.array(hdf.attach('Longitude')[:]).squeeze()
    seconds = hdf.attach('UTC_start')[:][0][0] + np.array(hdf.attach('Profile_time')[:]).squeeze()
    date_time = np.array([date + timedelta(seconds=s) for s in seconds])
    
    plt_scatter = ax.scatter(lon, lat, s=2, c=seconds, cmap=pltcmp, norm=pltnorm, lw=0, transform=ccrs.PlateCarree())

cbar = fig.colorbar(
    plt_scatter, format=remove_trailing_zero_pos,
    orientation="horizontal", ticks=pltticks, extend='max',
    cax=fig.add_axes([0.05, fm_bottom-0.05, 0.9, 0.04]))
cbar.ax.set_xlabel(f'UTC on {str(date)[:10]} along the orbit of CloudSat/CALIPSO', labelpad=4)
cbar.ax.set_xticklabels(np.arange(0, 24, 1))

# iimage = f'data/others/Blue Marble Next Generation w: Topography and Bathymetry/world.topo.bathy.2004{month:02d}.3x5400x2700.jpg'
# img = Image.open(iimage)
# ax.imshow(img, extent=[-180, 180, -90, 90], transform=ccrs.PlateCarree())

opng = f'figures/3_satellites/3.1_CloudSat_CALIPSO/3.1.0_Orbit of CloudSat and CALIPSO on {str(date)[:10]}.png'
fig.savefig(opng)



'''
hdf2 = SD(ifile, SDC.READ)
hdf2.datasets().keys()
CloudFraction = hdf2.select('CloudFraction').get()
CloudLayerType = hdf2.select('CloudLayerType').get()
CloudTypeQuality = hdf2.select('CloudTypeQuality').get()


stats.describe(seconds[1:] - seconds[:-1])
'''
# endregion


# region plot grid counts of 2B-CLDCLASS-LIDAR

cc_cldclass_count = xr.open_dataset(f'scratch/data/obs/CloudSat_CALIPSO/2B-CLDCLASS-LIDAR.P1_R05/cc_cldclass_count.nc').cc_cldclass_count
pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=60000+1e-4, cm_interval1=2000, cm_interval2=10000, cmap='viridis',)

fm_bottom=1.5/(6+1.5)
fig, ax = globe_plot(figsize=np.array([12, 6+1.5]) / 2.54, fm_bottom=fm_bottom)

plt_mesh = ax.pcolormesh(
    cc_cldclass_count.lon, cc_cldclass_count.lat, cc_cldclass_count,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_mesh, format=remove_trailing_zero_pos,
    orientation="horizontal", ticks=pltticks, extend='max',
    cax=fig.add_axes([0.05, fm_bottom-0.05, 0.9, 0.04]))
cbar.ax.set_xlabel(f'Count of CloudSat/CALIPSO 2B-CLDCLASS-LIDAR observations', labelpad=4)
# cbar.ax.set_xticklabels(np.arange(0, 24, 1))

opng = f'figures/3_satellites/3.1_CloudSat_CALIPSO/3.1.0_CloudSat and CALIPSO 2B-CLDCLASS-LIDAR counts.png'
fig.savefig(opng)


'''
stats.describe(cc_cldclass_count.values, axis=None)
np.max(cc_cldclass_count)
(cc_cldclass_count == 0).sum() / 180 / 360 = 0.08888889
(cc_cldclass_count.sel(lat=slice(-80, 80)) == 0).sum()
'''
# endregion


# region check annual cycle of counts of 2B-CLDCLASS-LIDAR

geolocation_all = pd.read_pickle(f'scratch/data/obs/CloudSat_CALIPSO/2B-CLDCLASS-LIDAR.P1_R05/geolocation_all.pkl')

monthly_counts = geolocation_all.date_time.dt.month.value_counts().sort_index()


'''
date_time
1     146096211
2     139433665
3     154127407
4     150753986
5     169138482
6     166030189
7     147512206
8     194844275
9     157537558
10    173294050
11    190798227
12    165380608
Name: count, dtype: int64
'''
# endregion

