

# qsub -I -q normal -P v46 -l walltime=1:00:00,ncpus=1,mem=10GB,jobfs=100MB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22+gdata/py18+gdata/gx60+gdata/xp65+gdata/qx55+gdata/rv74+gdata/al33+gdata/rr3+gdata/hr22+scratch/gx60+scratch/gb02+gdata/gb02


# region import packages

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
from metpy.calc import pressure_to_height_std
from metpy.units import units
import pickle
import glob
import h5py
from datetime import datetime

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=12)
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})
from cartopy.mpl.ticker import LatitudeFormatter
from matplotlib.ticker import AutoMinorLocator
import cartopy.feature as cfeature

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

# self defined
from mapplot import (
    globe_plot,
    remove_trailing_zero_pos,
    )

from namelist import (
    seconds_per_d,
    zerok,
    era5_varlabels,
    cmip6_era5_var,
    )

from component_plot import (
    plt_mesh_pars,
)

from calculations import (
    mon_sea_ann,
    time_weighted_mean,
    global_land_ocean_rmsd,
    global_land_ocean_mean,
    regrid,)

from statistics0 import (
    ttest_fdr_control,)

from um_postprocess import (
    interp_to_pressure_levels, preprocess_amoutput, amvargroups, am3_label)

# endregion


# region get AMSR data

fl = sorted(glob.glob(f'data/obs/AMSR/AMSR_U2_L3_MonthlyOcean_V01_??????.he5'))

with h5py.File(fl[0], 'r') as ds:
    lat = ds['HDFEOS']['GRIDS']['GRID']['Data Fields']['Latitude'][:, 0]
    lon = ds['HDFEOS']['GRIDS']['GRID']['Data Fields']['Longitude'][0, :]

all_lwp = []
all_times = []

for ifile in fl:
    all_times.append(datetime.strptime(ifile.split('.')[0][-6:], '%Y%m'))
    
    with h5py.File(ifile, 'r') as ds:
        data = ds['HDFEOS']['GRIDS']['GRID']['Data Fields']['LiquidWaterPath'][:]
        data[data < 0] = np.nan
        all_lwp.append(data)


amsr_lwp = xr.Dataset(
    data_vars = {'LWP': (('time', 'lat', 'lon'), np.array(all_lwp))},
    coords={'time': all_times, 'lat': lat, 'lon': lon})

amsr_lwp.to_netcdf('data/obs/AMSR/amsr_lwp.nc')




amsr_lwp = xr.open_dataset('data/obs/AMSR/amsr_lwp.nc')

amsr_lwp_alltime = mon_sea_ann(var_monthly=amsr_lwp['LWP'].sel(time=slice('2016', '2023')).sortby(['lat', 'lon']), lcopy=False, mm=True, sm=True, am=True)
ofile = f'data/obs/AMSR/amsr_lwp_alltime.pkl'
if os.path.exists(ofile): os.remove(ofile)
with open(ofile, 'wb') as f:
    pickle.dump(amsr_lwp_alltime, f)



'''
#---------------------------------------- check
ofile = f'data/obs/AMSR/amsr_lwp_alltime.pkl'
with open(ofile, 'rb') as f:
    amsr_lwp_alltime = pickle.load(f)
amsr_lwp_alltime['am'].squeeze().to_netcdf('scratch/run/test.nc')


#---------------------------------------- check
amsr_lwp = xr.open_dataset('data/obs/AMSR/amsr_lwp.nc')
ofile = f'data/obs/AMSR/amsr_lwp_alltime.pkl'
with open(ofile, 'rb') as f:
    amsr_lwp_alltime = pickle.load(f)

data1 = amsr_lwp_alltime['mon'].values
data2 = amsr_lwp['LWP'].sel(time=slice('2016', '2023')).sortby(['lat', 'lon']).values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())


#---------------------------------------- check
amsr_lwp = xr.open_dataset('data/obs/AMSR/amsr_lwp.nc')

fl = sorted(glob.glob(f'data/obs/AMSR/AMSR_U2_L3_MonthlyOcean_V01_??????.he5'))
itime = -20
ds = h5py.File(fl[itime], 'r')


print(amsr_lwp['LWP'][itime].time)
print(datetime.strptime(fl[itime].split('.')[0][-6:], '%Y%m'))

data1 = ds['HDFEOS']['GRIDS']['GRID']['Data Fields']['LiquidWaterPath'][:]
data1[data1 < 0] = np.nan
data2 = amsr_lwp['LWP'][itime].values
print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())

print((ds['HDFEOS']['GRIDS']['GRID']['Data Fields']['Latitude'][:, 0] == amsr_lwp.lat.values).all())

print((ds['HDFEOS']['GRIDS']['GRID']['Data Fields']['Longitude'][0, :] == amsr_lwp.lon.values).all())


print(ds['HDFEOS']['GRIDS']['GRID']['Data Fields'].keys())


'''
# endregion

