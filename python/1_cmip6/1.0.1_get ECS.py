

# qsub -I -q normal -P v46 -l walltime=4:00:00,ncpus=1,mem=192GB,jobfs=10GB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22+gdata/py18+gdata/gx60+gdata/xp65+gdata/qx55+gdata/rv74+gdata/al33+gdata/rr3


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
import json

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
import seaborn as sns
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.ticker import AutoMinorLocator
import geopandas as gpd
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import cartopy.feature as cfeature

# management
import os
import sys  # print(sys.path)
sys.path.append(os.getcwd() + '/code/gbr_future/module')
import psutil
process = psutil.Process()
# print(process.memory_info().rss / 2**30)
import string

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
)

from calculations import (
    mon_sea_ann,
    cdo_regrid,)

# endregion


# region get ECS

cmips = ['cmip6']
experiment_ids  = ['piControl', 'abrupt-4xCO2']
table_ids       = ['Amon']
variable_ids    = ['tas', 'rsut', 'rsdt', 'rlut']

def calc_ecs(tas, imb):
    a, b = np.polyfit(tas, imb, 1)
    ecs = -0.5 * (b/a)
    return xr.DataArray(ecs)

ds = {}
for icmip in cmips:
    # icmip = 'cmip6'
    print(f'#-------------------------------- {icmip}')
    ds[icmip] = {}
    for experiment_id in experiment_ids:
        # experiment_id = 'piControl'
        print(f'#---------------- {experiment_id}')
        ds[icmip][experiment_id] = {}
        for table_id in table_ids:
            # table_id = 'Amon'
            print(f'#-------- {table_id}')
            ds[icmip][experiment_id][table_id] = {}
            for variable_id in variable_ids:
                # variable_id = 'tas'
                print(f'#---- {variable_id}')
                
                ofile4 = f'data/sim/cmip/{icmip}/{experiment_id}/{table_id}_{variable_id}_regridded_alltime_ens_gzm.pkl'
                with open(ofile4, 'rb') as f:
                    ds[icmip][experiment_id][table_id][variable_id] = pickle.load(f)
    
    source_ids = sorted(set.intersection(
        *(set(ds[icmip][e][t][v]['am']['gm'].source_id.values.astype('object'))
          for e in experiment_ids for t in table_ids for v in variable_ids)))
    experiment_da = xr.DataArray(experiment_ids, dims='experiment_id', coords={'experiment_id': experiment_ids})
    
    ds_combined = xr.Dataset(data_vars={
        variable_id: xr.concat(
            [ds[icmip][experiment_id]['Amon'][variable_id]['ann']['gm'].sel(source_id = source_ids)
             for experiment_id in experiment_ids],
            dim=experiment_da, coords='minimal', compat='override')
        for variable_id in variable_ids
        })
    ds_combined['imbalance'] = ds_combined[['rsdt','rsut','rlut']].to_array().sum('variable')
    
    ds_anom = ds_combined[['tas', 'imbalance']].sel(experiment_id='abrupt-4xCO2') - ds_combined[['tas', 'imbalance']].sel(experiment_id='piControl').mean(dim='time')
    
    ecs = xr.apply_ufunc(calc_ecs, ds_anom.tas, ds_anom.imbalance, vectorize=True, input_core_dims=[['time'], ['time']]).rename('ecs')
    ecs = ecs.sortby(ecs).reset_coords(drop=True).to_dataframe(name='ecs').reset_index()
    
    ofile = f'data/sim/cmip/{icmip}/ecs.pkl'
    if os.path.exists(ofile): os.remove(ofile)
    with open(ofile, 'wb') as f: pickle.dump(ecs, f)




'''
with open(f'data/sim/cmip/{icmip}/ecs.pkl', 'rb') as f: ecs = pickle.load(f)

https://projectpythia.org/cmip6-cookbook/notebooks/example-workflows/ecs-cmip6.html

    ds_mean = ds_combined[['tas', 'imbalance']].sel(experiment_id='piControl').mean(dim='time')
    ds_anom = ds_combined[['tas', 'imbalance']] - ds_mean
    ds_abrupt = ds_anom.sel(experiment_id='abrupt-4xCO2')

# stats.describe(ecs)
# DescribeResult(nobs=np.int64(46), minmax=(np.float64(1.8383265574525445), np.float64(5.650972913287361)), mean=np.float64(3.7781882806651326), variance=np.float64(1.252691070483524), skewness=np.float64(0.12394400829147324), kurtosis=np.float64(-1.2515608765999118))
'''
# endregion

