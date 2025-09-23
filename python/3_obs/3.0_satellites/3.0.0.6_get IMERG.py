

# qsub -I -q normal -P v46 -l walltime=3:00:00,ncpus=1,mem=60GB,jobfs=100MB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22+gdata/gx60


# region import packages

# data analysis
import numpy as np
import pandas as pd
import numpy.ma as ma
import glob
from datetime import datetime, timedelta
from pyhdf.SD import SD, SDC
from pyhdf.HDF import HDF
from pyhdf.VS import VS
from pyhdf.error import HDF4Error
from satpy.scene import Scene
from skimage.measure import block_reduce
import xarray as xr
from netCDF4 import Dataset
import pickle

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=10)
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({'mathtext.fontset': 'stix'})

# management
import os
import sys  # print(sys.path)
sys.path.append(os.getcwd() + '/code/gbr_future/module')
import psutil
process = psutil.Process()
# print(process.memory_info().rss / 2**30)

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
    mon_sea_ann,
    )

from metplot import si2reflectance, si2radiance, get_modis_latlonrgbs


# endregion


# region get IMERG alltime pr

fl = sorted(glob.glob('data/obs/IMERG/3B-MO.MS.MRG.3IMERG/*.HDF5'))

data_list = []
time_list = []

for ifile in fl:
    # ifile = fl[0]
    print(ifile)
    ds = Dataset(ifile, 'r')
    if not data_list:
        lat = ds.groups['Grid'].variables['lat'][:]
        lon = ds.groups['Grid'].variables['lon'][:]
    
    data_list.append(ds.groups['Grid'].variables['precipitation'][0].filled(np.nan) * 24)
    time_list.append(datetime.strptime(ifile.split('.')[-4][:8], '%Y%m%d'))
    ds.close()

imerg_mon = xr.DataArray(
    data=np.stack(data_list, axis=0),
    coords={'time': pd.to_datetime(time_list), 'lon': lon, 'lat': lat},
    dims=['time', 'lon', 'lat'],
    name='pr').transpose('time', 'lat', 'lon')

imerg_mon_alltime = mon_sea_ann(var_monthly=imerg_mon, mm=True,sm=True,am=True)
ofile = f'data/obs/IMERG/imerg_mon_alltime_pr.pkl'
if os.path.exists(ofile): os.remove(ofile)
with open(ofile, 'wb') as f:
    pickle.dump(imerg_mon_alltime, f)




'''
#-------------------------------- check
with open(f'data/obs/IMERG/imerg_mon_alltime_pr.pkl', 'rb') as f:
    imerg_mon_alltime = pickle.load(f)

fl = sorted(glob.glob('data/obs/IMERG/3B-MO.MS.MRG.3IMERG/*.HDF5'))
ifile = -16
ds = Dataset(fl[ifile], 'r')

print((imerg_mon_alltime['mon']['lat'] == ds.groups['Grid'].variables['lat'][:]).all())
print((imerg_mon_alltime['mon']['lon'] == ds.groups['Grid'].variables['lon'][:]).all())
print((imerg_mon_alltime['mon'].transpose('time', 'lon', 'lat')[ifile].values[np.isfinite(imerg_mon_alltime['mon'].transpose('time', 'lon', 'lat')[ifile].values)] == ds.groups['Grid'].variables['precipitation'][0].filled(np.nan)[np.isfinite(ds.groups['Grid'].variables['precipitation'][0].filled(np.nan))] * 24).all())




ifile = '/home/563/qg8515/data/obs/IMERG/3B-MO.MS.MRG.3IMERG/3B-MO.MS.MRG.3IMERG.20160101-S000000-E235959.01.V07B.HDF5'

ds = Dataset(ifile, 'r')
ds.groups['Grid'].variables['precipitation'][:]
ds.groups['Grid'].variables['time'][:]


import h5py
ds = h5py.File(ifile, 'r')
ds['Grid']['precipitation'][:]

ds.groups.keys()
ds.groups['Grid'].variables.keys()

'''
# endregion

