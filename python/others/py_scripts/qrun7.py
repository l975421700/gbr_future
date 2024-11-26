

# region import packages

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
import pandas as pd
import intake
from cdo import Cdo
cdo=Cdo()
import fsspec

# management
import os
import sys  # print(sys.path)
sys.path.append('/home/563/qg8515/code/gbr_future/module')
import pickle

# self defined function
from calculations import (
    mon_sea_ann,
    cdo_regrid,
    time_weighted_mean,
    )
from cmip import (
    combined_preprocessing,
    drop_all_bounds,
    open_dsets,
    open_delayed,
    )

# get cmip table info
cmip_info = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
esm_datastore = intake.open_esm_datastore("https://storage.googleapis.com/cmip6/pangeo-cmip6.json")

'''
cmip_info['experiment_id'].unique()
cmip_info['institution_id'].unique()
'''
# endregion


# region get 'ssp126', 'Omon', 'tos'

# Service Units:      74.53
# Memory Used: 59.0GB
# Walltime Used: 00:46:35

#-------------------------------- configurations

experiment_id = 'ssp126'
table_id = 'Omon'
variable_id = 'tos'

member_id = ['r1i1p1f1', 'r1i1p1f2', 'r1i1p2f1', 'r2i1p1f1', 'r1i1p1f3', 'r4i1p1f1', 'r1i1p3f1']

esm_data = esm_datastore.search(**{
    'experiment_id': experiment_id,
    'table_id': table_id,
    'variable_id': variable_id,
    'member_id': member_id,
    })
esm_data_subset = esm_data.df.sort_values(
    ['source_id', 'member_id', 'grid_label', 'version'],
    ascending=[True, True, True, False]).groupby('source_id').first()

output_file = '/home/563/qg8515/data/sim/cmip6/' + experiment_id + '_' + table_id + '_' + variable_id + '.pkl'
output_file_regrid = '/home/563/qg8515/data/sim/cmip6/' + experiment_id + '_' + table_id + '_' + variable_id + '_regrid.pkl'
output_file_regrid_alltime = '/home/563/qg8515/data/sim/cmip6/' + experiment_id + '_' + table_id + '_' + variable_id + '_regrid_alltime.pkl'

intermediate_file = '/home/563/qg8515/data/sim/cmip6/' + experiment_id + '_' + table_id + '_' + variable_id + '_intf.pkl'
intermediate_file1 = '/home/563/qg8515/data/sim/cmip6/' + experiment_id + '_' + table_id + '_' + variable_id + '_intf1.pkl'


#-------------------------------- get data

dsets = {}
for group, df in esm_data_subset.groupby('source_id'):
    dsets[group] = open_delayed(df)

datasets = dask.compute(dsets)[0]

with open(output_file, 'wb') as f: pickle.dump(datasets, f)


#-------------------------------- get regridded data

# with open(output_file, 'rb') as f: datasets = pickle.load(f)

datasets_regrid = {}

for imodel in datasets.keys():
    # imodel = 'ACCESS-ESM1-5'
    print('#---------------- ' + imodel)
    
    datasets_regrid[imodel] = cdo_regrid(datasets[imodel], intermediate_file, intermediate_file1).pipe(combined_preprocessing).pipe(drop_all_bounds)

with open(output_file_regrid, 'wb') as f: pickle.dump(datasets_regrid, f)


#-------------------------------- get mon_sea_ann regridded data

# with open(output_file_regrid, 'rb') as f: datasets_regrid = pickle.load(f)

datasets_regrid_alltime = {}

for imodel in datasets_regrid.keys():
    # imodel = 'ACCESS-ESM1-5'
    print('#---------------- ' + imodel)
    
    datasets_regrid_alltime[imodel] = mon_sea_ann(
        var_monthly=datasets_regrid[imodel][variable_id], lcopy = False)

with open(output_file_regrid_alltime, 'wb') as f:
    pickle.dump(datasets_regrid_alltime, f)


'''
with open('data/sim/cmip6/ssp126_Omon_tos_regrid_alltime.pkl', 'rb') as f:
    ssp126_Omon_tos = pickle.load(f)

print(esm_data)
print(esm_data.df)
print(esm_data.df[['source_id', 'member_id']].to_string())
esm_data.df.groupby("source_id").first()
esm_data.df.groupby("source_id").nunique()
print(esm_data.df.sort_values(
    ['source_id', 'member_id', 'grid_label', 'version'],
    ascending=[True, True, True, False])[['source_id', 'member_id', 'grid_label', 'version']].to_string())

# 'ssp585', 'ssp126', 'amip'
# 'Amon', 'tas'


#-------------------------------- check
with open(output_file, 'rb') as f: datasets = pickle.load(f)
with open(output_file_regrid, 'rb') as f: datasets_regrid = pickle.load(f)
with open(output_file_regrid_alltime, 'rb') as f: datasets_regrid_alltime = pickle.load(f)

for imodel in datasets.keys():
    # imodel = 'ACCESS-ESM1-5'
    print('#---------------- ' + imodel)
    
    print(str(datasets[imodel].time[0].values)[:10] + ' to ' + str(datasets[imodel].time[-1].values)[:10] + ' ' + str(len(datasets[imodel].time)/12) + ' ' + str(datasets[imodel][variable_id].shape))
    print(str(datasets_regrid[imodel].time[0].values)[:10] + ' to ' + str(datasets_regrid[imodel].time[-1].values)[:10] + ' ' + str(len(datasets_regrid[imodel].time)/12) + ' ' + str(datasets_regrid[imodel][variable_id].shape))
    print(str(datasets_regrid_alltime[imodel]['mon'].time[0].values)[:10] + ' to ' + str(datasets_regrid_alltime[imodel]['mon'].time[-1].values)[:10] + ' ' + str(len(datasets_regrid_alltime[imodel]['mon'].time)/12) + ' ' + str(datasets_regrid_alltime[imodel]['mon'].shape))
    print(datasets_regrid_alltime[imodel].keys())
    
    # print(datasets[imodel])
    # print(datasets_regrid[imodel])
    # print(datasets_regrid_alltime[imodel])

import psutil
process = psutil.Process()
print(process.memory_info().rss / 2**30)

'''
# endregion
