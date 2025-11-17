

# region import packages

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
import glob
import rioxarray as rxr
import calendar
import pickle

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=12)
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})
import matplotlib.ticker as mticker
import cartopy.feature as cfeature

# management
import os
import sys  # print(sys.path)
sys.path.append(os.getcwd() + '/code/gbr_future/module')
import psutil
process = psutil.Process()
# print(process.memory_info().rss / 2**30)
import warnings
warnings.filterwarnings('ignore')

from namelist import (
    month_jan,
    monthini,
    seasons,
    seconds_per_d,
    zerok,
    era5_varlabels,
    cmip6_era5_var,
    )

from mapplot import (
    regional_plot,
    remove_trailing_zero_pos,
    )

from component_plot import (
    plt_mesh_pars,)

from calculations import (
    mon_sea_ann,
    )

# endregion


# region get alltime monthly and hourly Himawari cmic

years = '2016'
yeare = '2024'
for ivar in ['clwvi']:
    # ivar = 'clivi'
    print(f'#-------------------------------- {ivar}')
    
    for iperiod in ['mhm']:
        # iperiod = 'mhm'
        print(f'#---------------- {iperiod}')
        
        if iperiod == 'mm':
            fl = sorted(glob.glob(f'data/obs/jaxa/{ivar}/{ivar}_??????.nc'))
            ofile = f'data/obs/jaxa/{ivar}/{ivar}_alltime.pkl'
        elif iperiod == 'mhm':
            fl = sorted(glob.glob(f'data/obs/jaxa/{ivar}/{ivar}_hourly_*.nc'))
            ofile = f'data/obs/jaxa/{ivar}/{ivar}_hourly_alltime.pkl'
        
        himawari_mon = xr.open_mfdataset(fl, combine='by_coords', parallel=True, data_vars='minimal',compat='override', coords='minimal')[ivar].sel(time=slice(years, yeare))
        
        if ivar in ['clwvi', 'clivi']:
            himawari_mon *= 1000
        
        himawari_mon_alltime = mon_sea_ann(
            var_monthly=himawari_mon, lcopy=False, mm=True, sm=True, am=True)
        
        if os.path.exists(ofile): os.remove(ofile)
        with open(ofile,'wb') as f:
            pickle.dump(himawari_mon_alltime, f)




# endregion

