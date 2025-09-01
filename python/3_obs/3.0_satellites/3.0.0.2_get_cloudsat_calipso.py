

# qsub -I -q normal -P nf33 -l walltime=3:00:00,ncpus=1,mem=40GB,storage=gdata/v46+gdata/rr1+scratch/v46


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


# region get CloudSat-CALIPSO data

product_var = {
    '2B-CWC-RO.P1_R05': [
        'Height',
        'RO_liq_water_content', 'RO_ice_water_content',
        'LO_RO_liquid_water_content', 'IO_RO_ice_water_content',
        'RO_liq_number_conc', 'RO_ice_number_conc',
        'LO_RO_number_conc'],
    '2C-RAIN-PROFILE.P1_R05': [
        'precip_liquid_water', 'precip_ice_water',
        'cloud_liquid_water'],
    '2C-SNOW-PROFILE.P1_R05': [
        'snow_water_content', 'snowfall_rate'],
    '2C-ICE.P1_R05': [
        'Temperature', 'IWC']}

year, month, day, hour = 2020, 6, 2, 4
doy = datetime(year, month, day).timetuple().tm_yday

for iproduct in product_var.keys():
    # ['2B-CWC-RO.P1_R05', '2C-RAIN-PROFILE.P1_R05', '2C-SNOW-PROFILE.P1_R05', '2C-ICE.P1_R05']
    # iproduct = '2B-CWC-RO.P1_R05'
    print(f'#-------------------------------- {iproduct}')
    
    fl = sorted(glob.glob(f'scratch/data/obs/CloudSat_CALIPSO/{iproduct}/{year}/{doy:03d}/{year}{doy}{hour:02d}*.hdf') + glob.glob(f'scratch/data/obs/CloudSat_CALIPSO/{iproduct}/{year}/{doy:03d}/{year}{doy}{hour-1:02d}*.hdf'))
    
    for idx, ifile in enumerate(fl):
        # ifile=fl[0]; idx=0
        print(f'#---------------- {idx}')
        # print(ifile)
        
        hdf_vs = HDF(ifile).vstart()
        hdf_sd = SD(ifile, SDC.READ)
        # print(hdf_sd.datasets().keys())
        # print(f'#----------------')
        # print(hdf_vs.vdatainfo())
        print(np.array(hdf_vs.attach('Longitude')[:]).squeeze().shape)
        
        for var in product_var[iproduct]:
            print(f'#-------- {var}')
            print(hdf_sd.select(var).get())
        
        
        lat = np.array(hdf_vs.attach('Latitude')[:]).squeeze()
        lon = np.array(hdf_vs.attach('Longitude')[:]).squeeze()
        seconds = hdf_vs.attach('UTC_start')[:][0][0] + np.array(hdf_vs.attach('Profile_time')[:]).squeeze()
        time = np.array([datetime(year, month, day) + timedelta(seconds=s) for s in seconds])
        
        
        
        
        
        
        
        
        
        









'''


year, month, day, hour = 2020, 6, 2, 4
doy = datetime(year, month, day).timetuple().tm_yday
for iproduct in [
    '2B-CLDCLASS-LIDAR.P1_R05',
    '2B-CLDCLASS.P1_R05',
    '2B-CWC-RO.P1_R05',
    '2B-CWC-RVOD.P1_R05', # No file after 2017
    '2B-FLXHR-LIDAR.P2_R05', # No file after 2017
    '2B-GEOPROF-LIDAR.P2_R05',
    '2B-GEOPROF.P1_R05',
    '2C-ICE.P1_R05',
    '2C-PRECIP-COLUMN.P1_R05',
    '2C-RAIN-PROFILE.P1_R05',
    '2C-SNOW-PROFILE.P1_R05',
    '2B-TB94.P1_R05', # No file after 2011
    ]:
    print(f'#-------------------------------- {iproduct}')
    fl = sorted(glob.glob(f'scratch/data/obs/CloudSat_CALIPSO/{iproduct}/{year}/{doy:03d}/{year}{doy}{hour:02d}*.hdf') + glob.glob(f'scratch/data/obs/CloudSat_CALIPSO/{iproduct}/{year}/{doy:03d}/{year}{doy}{hour-1:02d}*.hdf'))
    if len(fl) > 0:
        ifile=fl[0]
        hdf1 = SD(ifile, SDC.READ)
        for ivar in hdf1.datasets().keys():
            print(ivar)
    else:
        print('No file found')

#-------------------------------- 2B-CLDCLASS-LIDAR.P1_R05
dict_keys(['Height', 'CloudLayerBase', 'LayerBaseFlag', 'CloudLayerTop', 'LayerTopFlag', 'HorizontalOrientedIce', 'CloudFraction', 'CloudPhase', 'CloudPhaseConfidenceLevel', 'CloudLayerType', 'CloudTypeQuality', 'Phase_log', 'Water_layer_top'])
#-------------------------------- 2B-CLDCLASS.P1_R05
dict_keys(['Height', 'cloud_scenario', 'CloudLayerBase', 'CloudLayerTop', 'CloudLayerType'])
#-------------------------------- 2B-CWC-RO.P1_R05
dict_keys(['Height', 'RO_liq_effective_radius', 'RO_liq_effective_radius_uncertainty', 'RO_ice_effective_radius', 'RO_ice_effective_radius_uncertainty', 'RO_liq_number_conc', 'RO_liq_num_conc_uncertainty', 'RO_ice_number_conc', 'RO_ice_num_conc_uncertainty', 'RO_liq_distrib_width_param', 'RO_liq_distrib_width_param_uncertainty', 'RO_ice_distrib_width_param', 'RO_ice_distrib_width_param_uncertainty', 'RO_liq_water_content', 'RO_liq_water_content_uncertainty', 'RO_ice_water_content', 'RO_ice_water_content_uncertainty', 'RO_ice_phase_fraction', 'RO_radar_uncertainty', 'LO_RO_AP_geo_mean_radius', 'LO_RO_AP_sdev_geo_mean_radius', 'LO_RO_AP_number_conc', 'LO_RO_AP_sdev_num_conc', 'LO_RO_AP_distrib_width_param', 'LO_RO_AP_sdev_distrib_width_param', 'LO_RO_effective_radius', 'LO_RO_effective_radius_uncertainty', 'LO_RO_number_conc', 'LO_RO_num_conc_uncertainty', 'LO_RO_distrib_width_param', 'LO_RO_distrib_width_param_uncertainty', 'LO_RO_liquid_water_content', 'LO_RO_liquid_water_content_uncertainty', 'IO_RO_AP_log_geo_mean_diameter', 'IO_RO_AP_sdev_log_geo_mean_diameter', 'IO_RO_AP_log_number_conc', 'IO_RO_AP_sdev_log_num_conc', 'IO_RO_AP_distrib_width_param', 'IO_RO_AP_sdev_distrib_width_param', 'IO_RO_effective_radius', 'IO_RO_effective_radius_uncertainty', 'IO_RO_log_number_conc', 'IO_RO_log_num_conc_uncertainty', 'IO_RO_distrib_width_param', 'IO_RO_distrib_width_param_uncertainty', 'IO_RO_ice_water_content', 'IO_RO_ice_water_content_uncertainty'])
#-------------------------------- 2B-CWC-RVOD.P1_R05
No file found
#-------------------------------- 2B-FLXHR-LIDAR.P2_R05
No file found
#-------------------------------- 2B-GEOPROF-LIDAR.P2_R05
dict_keys(['Height', 'CloudFraction', 'VFMHistogram', 'UncertaintyCF', 'LayerBase', 'LayerTop', 'FlagBase', 'FlagTop', 'DistanceAvg', 'NumLidar'])
#-------------------------------- 2B-GEOPROF.P1_R05
dict_keys(['Height', 'CPR_Cloud_mask', 'Gaseous_Attenuation', 'Radar_Reflectivity'])
#-------------------------------- 2C-PRECIP-COLUMN.P1_R05
dict_keys(['unused'])
#-------------------------------- 2C-RAIN-PROFILE.P1_R05
dict_keys(['Height', 'precip_liquid_water', 'precip_ice_water', 'cloud_liquid_water', 'PWC_uncertainty', 'modeled_reflectivity', 'attenuation_correction', 'MS_correction'])
#-------------------------------- 2C-SNOW-PROFILE.P1_R05
dict_keys(['Height', 'snowfall_rate', 'snowfall_rate_uncert', 'log_N0', 'log_N0_uncert', 'log_lambda', 'log_lambda_uncert', 'snow_water_content', 'snow_water_content_uncert'])
#-------------------------------- 2B-TB94.P1_R05
No file found
#-------------------------------- 2C-ICE.P1_R05
dict_keys(['Height', 'Temperature', 'AP_re', 'AP_IWC', 'dBZe_uncertainty', 'TAB_uncertainty', 're', 'IWC', 'EXT_coef', 're_uncertainty', 'IWC_uncertainty', 'EXT_coef_uncertainty', 'dBZe_simulation', 'TAB_simulation', 'zone', 'ze_makeup', 'tab_para'])
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

