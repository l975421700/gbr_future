

# qsub -I -q normal -l walltime=4:00:00,ncpus=1,mem=192GB,jobfs=192GB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38


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
)

from calculations import (
    mon_sea_ann,
    cdo_regrid,)

# endregion


# region import data


cmip6_data_regridded_alltime_ens_gzm = {}
for experiment_id in ['piControl', 'abrupt-4xCO2']:
    print(f'#-------------------------------- {experiment_id}')
    cmip6_data_regridded_alltime_ens_gzm[experiment_id] = {}
    for table_id, variable_id in zip(['Amon', 'Amon', 'Amon', 'Amon'], ['tas', 'rsut', 'rsdt', 'rlut']):
        print(f'#---------------- {table_id} {variable_id}')
        if not table_id in cmip6_data_regridded_alltime_ens_gzm[experiment_id].keys():
            cmip6_data_regridded_alltime_ens_gzm[experiment_id][table_id] = {}
        
        with open(f'/home/563/qg8515/scratch/data/sim/cmip6/{experiment_id}_{table_id}_{variable_id}_regridded_alltime_ens_gzm.pkl', 'rb') as f:
            cmip6_data_regridded_alltime_ens_gzm[experiment_id][table_id][variable_id] = pickle.load(f)

with open('/home/563/qg8515/scratch/data/sim/cmip6/cmip6_ids.pkl', 'rb') as f:
    cmip6_ids = pickle.load(f)
source_ids = list(cmip6_ids.keys())
for experiment_id in ['piControl', 'abrupt-4xCO2']:
    for table_id, variable_id in zip(['Amon', 'Amon', 'Amon', 'Amon'], ['tas', 'rsut', 'rsdt', 'rlut']):
        source_ids = sorted(set(source_ids) & set(list(cmip6_data_regridded_alltime_ens_gzm[experiment_id][table_id][variable_id]['ann']['gm']['source_id'].values.astype('object'))))
        # print(len(source_ids))


'''

for experiment_id in ['piControl', 'abrupt-4xCO2']:
    print(f'#-------------------------------- {experiment_id}')
    for table_id, variable_id in zip(['Amon', 'Amon', 'Amon', 'Amon'], ['tas', 'rsut', 'rsdt', 'rlut']):
        print(f'#---------------- {table_id} {variable_id}')
        for ialltime in cmip6_data_regridded_alltime_ens_gzm[experiment_id][table_id][variable_id].keys():
            print(f'#-------- {ialltime}')
            print(cmip6_data_regridded_alltime_ens_gzm[experiment_id][table_id][variable_id][ialltime]['zm'].shape)
            print(cmip6_data_regridded_alltime_ens_gzm[experiment_id][table_id][variable_id][ialltime]['gm'].shape)


for experiment_id in ['piControl', 'abrupt-4xCO2']:
    print(f'#-------------------------------- {experiment_id}')
    for table_id, variable_id in zip(['Amon', 'Amon', 'Amon', 'Amon'], ['tas', 'rsut', 'rsdt', 'rlut']):
        print(f'#---------------- {table_id} {variable_id}')
        for source_id in cmip6_data_regridded_alltime[experiment_id][table_id][variable_id].keys():
            print(f'#-------- {source_id}')
            print(cmip6_data_regridded_alltime[experiment_id][table_id][variable_id][source_id]['am'].shape)

'''
# endregion


# region calculate ECS

expts = ['piControl', 'abrupt-4xCO2']
expt_da = xr.DataArray(expts, dims='experiment_id', coords={'experiment_id': expts})

dataset = xr.Dataset(data_vars={
    variable_id: xr.concat([
        cmip6_data_regridded_alltime_ens_gzm[experiment_id]['Amon'][variable_id]['ann']['gm'].sel(source_id = source_ids)
        for experiment_id in ['piControl', 'abrupt-4xCO2']], dim=expt_da, coords='minimal', compat='override')
    for variable_id in ['tas', 'rsut', 'rsdt', 'rlut']
})


dataset['imbalance'] = dataset['rsdt'] + dataset['rsut'] + dataset['rlut']
ds_mean = dataset[['tas', 'imbalance']].sel(experiment_id='piControl').mean(dim='time')
ds_anom = dataset[['tas', 'imbalance']] - ds_mean

ds_abrupt = ds_anom.sel(experiment_id='abrupt-4xCO2').reset_coords(drop=True)

def calc_ecs(tas, imb):
    a, b = np.polyfit(tas, imb, 1)
    ecs = -0.5 * (b/a)
    return xr.DataArray(ecs)

ds_abrupt['ecs'] = xr.apply_ufunc(calc_ecs, ds_abrupt.tas, ds_abrupt.imbalance, vectorize=True, input_core_dims=[['time'], ['time']])

stats.describe(ds_abrupt['ecs'])


'''
https://projectpythia.org/cmip6-cookbook/notebooks/example-workflows/ecs-cmip6.html
'''
# endregion
