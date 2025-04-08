

# qsub -I -q express -l walltime=10:00:00,ncpus=1,mem=192GB,storage=gdata/v46+gdata/rr1+scratch/v46


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
mpl.rcParams['figure.dpi'] = 300
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

from calculations import (
    find_ilat_ilon,
    )

# endregion


# region get lat/lon/datetime information
# Memory Used: 11.23GB, Walltime Used: 01:36:27

parser=argparse.ArgumentParser()
parser.add_argument('-y', '--year', type=int, required=True,)
args = parser.parse_args()

year=args.year


if True:
# def process_year(year):
# for year in range(2006, 2021):
    # year=2020
    print(f'#-------------------------------- {year}')
    
    geolocation_df = pd.DataFrame(columns=['lat', 'lon', 'date_time'])
    
    folders = sorted(glob.glob(f'scratch/data/obs/CloudSat_CALIPSO/2B-CLDCLASS-LIDAR.P1_R05/{year}/???'))
    for ifolder in folders:
        # ifolder = folders[0]
        doy = int(ifolder[-3:])
        print(f'#---------------- {doy}')
        
        date = datetime(year, 1, 1) + timedelta(days=doy - 1)
        month = date.month
        day = date.day
        
        fl = sorted(glob.glob(f'{ifolder}/*'))
        for ifile in fl:
            # ifile=fl[0]
            
            hdf = HDF(ifile).vstart()
            lat = np.array(hdf.attach('Latitude')[:]).squeeze()
            lon = np.array(hdf.attach('Longitude')[:]).squeeze()
            seconds = hdf.attach('UTC_start')[:][0][0] + np.array(hdf.attach('Profile_time')[:]).squeeze()
            date_time = np.array([date + timedelta(seconds=s) for s in seconds])
            
            if geolocation_df.empty:
                geolocation_df = pd.DataFrame({'lat': lat, 'lon': lon, 'date_time': date_time})
            else:
                geolocation_df = pd.concat(
                [geolocation_df,
                 pd.DataFrame({'lat': lat, 'lon': lon, 'date_time': date_time})],
                ignore_index=True)
    
    ofile = f'scratch/data/obs/CloudSat_CALIPSO/2B-CLDCLASS-LIDAR.P1_R05/{year}/geolocation_df.pkl'
    if os.path.exists(ofile): os.remove(ofile)
    geolocation_df.to_pickle(ofile)
    
    # return f'Finished processing {year}'

# joblib.Parallel(n_jobs=15)(joblib.delayed(process_year)(year) for year in range(2006, 2021))


'''
#-------------------------------- check

year = 2018
geolocation_df = pd.read_pickle(f'scratch/data/obs/CloudSat_CALIPSO/2B-CLDCLASS-LIDAR.P1_R05/{year}/geolocation_df.pkl')

folders = sorted(glob.glob(f'scratch/data/obs/CloudSat_CALIPSO/2B-CLDCLASS-LIDAR.P1_R05/{year}/???'))
ifolder = folders[-1]
doy = int(ifolder[-3:])
date = datetime(year, 1, 1) + timedelta(days=doy - 1)
month = date.month
day = date.day
fl = sorted(glob.glob(f'{ifolder}/*'))
ifile=fl[0]

hdf = HDF(ifile).vstart()
lat = np.array(hdf.attach('Latitude')[:]).squeeze()
lon = np.array(hdf.attach('Longitude')[:]).squeeze()
seconds = hdf.attach('UTC_start')[:][0][0] + np.array(hdf.attach('Profile_time')[:]).squeeze()
date_time = np.array([date + timedelta(seconds=s) for s in seconds])

idxs = np.where(geolocation_df.lat == lat[0])[0][0]
idxe = np.where(geolocation_df.lat == lat[-1])[0][0]

print((geolocation_df.lat[idxs:(idxe+1)] == lat).all())
print((geolocation_df.lon[idxs:(idxe+1)] == lon).all())
print((geolocation_df.date_time[idxs:(idxe+1)] == date_time).all())

'''
# endregion


# region get geolocation_all: concat all geolocation_df

for year in range(2006, 2021):
    # year = 2006
    print(f'#-------------------------------- {year}')
    
    geolocation_df = pd.read_pickle(f'scratch/data/obs/CloudSat_CALIPSO/2B-CLDCLASS-LIDAR.P1_R05/{year}/geolocation_df.pkl')
    
    if (year==2006):
        geolocation_all = geolocation_df.copy()
    else:
        geolocation_all = pd.concat([
            geolocation_all,
            geolocation_df], ignore_index=True)

ofile = f'scratch/data/obs/CloudSat_CALIPSO/2B-CLDCLASS-LIDAR.P1_R05/geolocation_all.pkl'
if os.path.exists(ofile): os.remove(ofile)
geolocation_all.to_pickle(ofile)



'''
#-------------------------------- check
geolocation_all = pd.read_pickle(f'scratch/data/obs/CloudSat_CALIPSO/2B-CLDCLASS-LIDAR.P1_R05/geolocation_all.pkl')

year = 2014
geolocation_df = pd.read_pickle(f'scratch/data/obs/CloudSat_CALIPSO/2B-CLDCLASS-LIDAR.P1_R05/{year}/geolocation_df.pkl')

idxs = np.where(geolocation_all.date_time == geolocation_df.date_time.iloc[0])[0][0]
idxe = np.where(geolocation_all.date_time == geolocation_df.date_time.iloc[-1])[0][0]
print((geolocation_all.lat[idxs:(idxe+1)].values == geolocation_df.lat.values).all())
print((geolocation_all.lon[idxs:(idxe+1)].values == geolocation_df.lon.values).all())
print((geolocation_all.date_time[idxs:(idxe+1)].values == geolocation_df.date_time.values).all())
'''
# endregion


# region get observations count in each 1*1 grid cell

geolocation_all = pd.read_pickle(f'scratch/data/obs/CloudSat_CALIPSO/2B-CLDCLASS-LIDAR.P1_R05/geolocation_all.pkl')

lat = np.arange(-89.5, 90, 1)
lon = np.arange(-179.5, 180, 1)
cc_cldclass_count = xr.Dataset(
    {'cc_cldclass_count': (['lat', 'lon'],
                           np.zeros((len(lat), len(lon))))},
    coords={'lat': lat, 'lon': lon})

geolocation_all['ilats'] = np.int64(np.ceil(geolocation_all.lat) - 0.5 + 89.5)
geolocation_all['ilons'] = np.int64(np.ceil(geolocation_all.lon) - 0.5 + 179.5)

counts = geolocation_all[['ilats', 'ilons', 'date_time']].groupby(['ilats', 'ilons'], as_index=False).count()

for ilat, ilon, icount in zip(counts.ilats, counts.ilons, counts.date_time):
    cc_cldclass_count.cc_cldclass_count[ilat, ilon] = icount

print(geolocation_all.shape[0] == cc_cldclass_count.cc_cldclass_count.sum())

ofile = f'scratch/data/obs/CloudSat_CALIPSO/2B-CLDCLASS-LIDAR.P1_R05/cc_cldclass_count.nc'
if os.path.exists(ofile): os.remove(ofile)
cc_cldclass_count.to_netcdf(ofile)




'''
#-------------------------------- check
geolocation_all = pd.read_pickle(f'scratch/data/obs/CloudSat_CALIPSO/2B-CLDCLASS-LIDAR.P1_R05/geolocation_all.pkl')
cc_cldclass_count = xr.open_dataset(f'scratch/data/obs/CloudSat_CALIPSO/2B-CLDCLASS-LIDAR.P1_R05/cc_cldclass_count.nc')

lat = np.arange(-89.5, 90, 1)
lon = np.arange(-179.5, 180, 1)

ilat = 50
ilon = 50
slat = lat[ilat]
slon = lon[ilon]

print(cc_cldclass_count.cc_cldclass_count[ilat, ilon].values == ((geolocation_all.lat > (slat - 0.5)) & (geolocation_all.lat <= (slat + 0.5)) & (geolocation_all.lon > (slon - 0.5)) & (geolocation_all.lon <= (slon + 0.5))).sum())




# year = 2007
# geolocation_df = pd.read_pickle(f'scratch/data/obs/CloudSat_CALIPSO/2B-CLDCLASS-LIDAR.P1_R05/{year}/geolocation_df.pkl')

# alternative method 1
for ilat, ilon in zip(ilats[:1000], ilons[:1000]):
    cc_cldclass_count.cc_cldclass_count[ilat, ilon] += 1

# alternative method 2
ilats = np.int64(np.ceil(geolocation_df.lat) - 0.5 + 89.5)
ilons = np.int64(np.ceil(geolocation_df.lon) - 0.5 + 179.5)
for idxlat in range(180):
    for idxlon in range(360):
        # idxlat=20; idxlon=20
        cc_cldclass_count.cc_cldclass_count[idxlat, idxlon] = ((ilats == idxlat) & (ilons == idxlon)).sum()

# alternative method 3
for slat, slon in zip(geolocation_df.lat[:10], geolocation_df.lon[:10]):
    # slat=geolocation_df.lat[0]; slon=geolocation_df.lon[0]
    # print(f'{slat} {slon}')
    
    ilat, ilon = find_ilat_ilon(slat, slon, lat, lon.copy())
    
    if not ((abs(lat[ilat] - slat) <= 0.5) & (abs(lon[ilon] - slon) <= 0.5)):
        print('Warning: inferred indices too far')
    
    cc_cldclass_count.cc_cldclass_count[ilat, ilon] += 1


#-------------------------------- check
for ilat, ilon, slat, slon in zip(ilats[0:100000:10000], ilons[0:100000:10000], geolocation_df.lat[0:100000:10000], geolocation_df.lon[0:100000:10000]):
    
    if ((abs(lat[ilat] - slat) <= 0.5) & (abs(lon[ilon] - slon) <= 0.5)):
        print('True')



for iitem in np.arange(0, 30000000, 2000000):
    # iitem = 20000
    ilat, ilon = find_ilat_ilon(geolocation_df.lat[iitem], geolocation_df.lon[iitem], lat.copy(), lon.copy())
    
    print(f'Latitude diff.: {np.round(lat[ilat] - geolocation_df.lat[iitem], 1)}')
    print(f'Longitude diff.: {np.round(lon[ilon] - geolocation_df.lon[iitem], 1)}')

'''
# endregion




# region trial

import xarray as xr
from pyhdf.SD import SD, SDC
from pyhdf.HDF import HDF
from pyhdf.VS import VS

ifile = 'scratch/data/obs/CloudSat_CALIPSO/2B-CLDCLASS-LIDAR.P1_R05/2017/119/2017119003528_58527_CS_2B-CLDCLASS-LIDAR_GRANULE_P1_R05_E06_F01.hdf'

hdf = HDF(ifile).vstart()
len(hdf.attach('Latitude')[:])
len(hdf.attach('Longitude')[:])


hdf2 = SD(ifile, SDC.READ)
hdf2.datasets().keys()
CloudFraction = hdf2.select('CloudFraction').get()
CloudLayerType = hdf2.select('CloudLayerType').get()
CloudTypeQuality = hdf2.select('CloudTypeQuality').get()


'''
https://en.moonbooks.org/Jupyter/Read_and_Plot_CloudSat_2B_CLDCLASS_LIDAR_Product/
'''
# endregion

