

# qsub -I -q copyq -l walltime=02:00:00,ncpus=1,mem=100GB,jobfs=100GB,storage=gdata/v46


# region import packages

# data analysis
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
import pandas as pd
import intake
from cdo import Cdo
cdo=Cdo()

# management
import sys  # print(sys.path)
sys.path.append('/home/563/qg8515/code/gbr_future/module')
import pickle
import psutil
process = psutil.Process()
import gc

# self defined function
from calculations import (
    mon_sea_ann,
    cdo_regrid,
    )
from cmip import (
    combined_preprocessing,
    drop_all_bounds,
    open_delayed,
    )

cmip_info = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
esm_datastore = intake.open_esm_datastore("https://storage.googleapis.com/cmip6/pangeo-cmip6.json")

ffolder = '/home/563/qg8515/data/sim/cmip6/'

'''
cmip_info['experiment_id'].unique()
cmip_info['institution_id'].unique()
'''
# endregion


# historical
# region get 'historical', 'Omon', 'tos'

# configurations
exp_id = 'historical'
table_id = 'Omon'
var_id = 'tos'

esm_data = esm_datastore.search(**{
    'experiment_id': exp_id, 'table_id': table_id, 'variable_id': var_id,
    'member_id': ['r1i1p1f1', 'r1i1p1f2', 'r1i1p2f1', 'r2i1p1f1', 'r1i1p1f3',],})
esm_data_subset = esm_data.df.sort_values(
    ['source_id', 'member_id', 'grid_label', 'version'],
    ascending=[True, True, True, False]).groupby('source_id').first()

outf = f'{ffolder}{exp_id}_{table_id}_{var_id}.pkl'
outf_rgd = f'{ffolder}{exp_id}_{table_id}_{var_id}_rgd.pkl'
outf_rgd_alltime = f'{ffolder}{exp_id}_{table_id}_{var_id}_rgd_alltime.pkl'

dsets = {}
for group, df in esm_data_subset.groupby('source_id'):
    dsets[group] = open_delayed(df)

dsets = dask.compute(dsets)[0]
with open(outf, 'wb') as f: pickle.dump(dsets, f)

dsets_rgd = {}
for imodel in dsets.keys():
    print(imodel)
    if (len(dsets[imodel].sel(time=slice('1979', '2014')).time) > 0):
        dsets_rgd[imodel] = cdo_regrid(dsets[imodel].sel(time=slice('1979', '2014'))).pipe(combined_preprocessing).pipe(drop_all_bounds)

del dsets
with open(outf_rgd, 'wb') as f: pickle.dump(dsets_rgd, f)

print('get mon_sea_ann regridded data')
dsets_rgd_alltime = {}
for imodel in dsets_rgd.keys():
    print(imodel)
    dsets_rgd_alltime[imodel] = mon_sea_ann(
        var_monthly=dsets_rgd[imodel][var_id], lcopy=False)

del dsets_rgd
with open(outf_rgd_alltime, 'wb') as f: pickle.dump(dsets_rgd_alltime, f)


'''
with open('data/sim/cmip6/historical_Omon_tos_rgd_alltime.pkl', 'rb') as f:
    historical_Omon_tos = pickle.load(f)

print(esm_data)
print(esm_data.df)
esm_data.df.groupby("source_id").first()
esm_data.df.groupby("source_id").nunique()
print(esm_data.df.sort_values(
    ['source_id', 'member_id', 'grid_label', 'version'],
    ascending=[True, True, True, False])[['source_id', 'member_id', 'grid_label', 'version']].to_string())


#-------------------------------- check
with open(outf, 'rb') as f: dsets = pickle.load(f)
with open(outf_rgd, 'rb') as f: dsets_rgd = pickle.load(f)
with open(outf_rgd_alltime, 'rb') as f: dsets_rgd_alltime = pickle.load(f)

for imodel in dsets.keys():
    # imodel = 'ACCESS-ESM1-5'
    print('#---------------- ' + imodel)
    
    print(dsets[imodel][var_id].shape)
    
    print(str(dsets[imodel].time[0].values)[:10] + ' to ' + str(dsets[imodel].time[-1].values)[:10] + ' ' + str(len(dsets[imodel].time)/12) + ' ' + str(dsets[imodel][var_id].shape))
    print(str(dsets_rgd[imodel].time[0].values)[:10] + ' to ' + str(dsets_rgd[imodel].time[-1].values)[:10] + ' ' + str(len(dsets_rgd[imodel].time)/12) + ' ' + str(dsets_rgd[imodel][var_id].shape))
    print(str(dsets_rgd_alltime[imodel]['mon'].time[0].values)[:10] + ' to ' + str(dsets_rgd_alltime[imodel]['mon'].time[-1].values)[:10] + ' ' + str(len(dsets_rgd_alltime[imodel]['mon'].time)/12) + ' ' + str(dsets_rgd_alltime[imodel]['mon'].shape))
    print(dsets_rgd_alltime[imodel].keys())
    
    # print(dsets[imodel])
    # print(dsets_rgd[imodel])
    # print(dsets_rgd_alltime[imodel])

import psutil
process = psutil.Process()

'''
# endregion


# region get 'historical', 'Amon', 'tas'

# configurations
exp_id = 'historical'
table_id = 'Amon'
var_id = 'tas'

esm_data = esm_datastore.search(**{
    'experiment_id': exp_id, 'table_id': table_id, 'variable_id': var_id,
    'member_id': ['r1i1p1f1', 'r1i1p1f2', 'r1i1p2f1', 'r2i1p1f1', 'r1i1p1f3',],})
esm_data_subset = esm_data.df.sort_values(
    ['source_id', 'member_id', 'grid_label', 'version'],
    ascending=[True, True, True, False]).groupby('source_id').first()

outf = f'{ffolder}{exp_id}_{table_id}_{var_id}.pkl'
outf_rgd = f'{ffolder}{exp_id}_{table_id}_{var_id}_rgd.pkl'
outf_rgd_alltime = f'{ffolder}{exp_id}_{table_id}_{var_id}_rgd_alltime.pkl'

dsets = {}
for group, df in esm_data_subset.groupby('source_id'):
    dsets[group] = open_delayed(df)

dsets = dask.compute(dsets)[0]
with open(outf, 'wb') as f: pickle.dump(dsets, f)

dsets_rgd = {}
for imodel in dsets.keys():
    print(imodel)
    if (len(dsets[imodel].sel(time=slice('1979', '2014')).time) > 0):
        dsets_rgd[imodel] = cdo_regrid(dsets[imodel].sel(time=slice('1979', '2014'))).pipe(combined_preprocessing).pipe(drop_all_bounds)

del dsets
with open(outf_rgd, 'wb') as f: pickle.dump(dsets_rgd, f)

print('get mon_sea_ann regridded data')
dsets_rgd_alltime = {}
for imodel in dsets_rgd.keys():
    print(imodel)
    dsets_rgd_alltime[imodel] = mon_sea_ann(
        var_monthly=dsets_rgd[imodel][var_id], lcopy=False)

del dsets_rgd
with open(outf_rgd_alltime, 'wb') as f: pickle.dump(dsets_rgd_alltime, f)


'''
with open('data/sim/cmip6/test/historical_Amon_tas_rgd_alltime.pkl', 'rb') as f:
    historical_Amon_tas = pickle.load(f)

print(esm_data)
print(esm_data.df)
esm_data.df.groupby("source_id").first()
esm_data.df.groupby("source_id").nunique()
print(esm_data.df.sort_values(
    ['source_id', 'member_id', 'grid_label', 'version'],
    ascending=[True, True, True, False])[['source_id', 'member_id', 'grid_label', 'version']].to_string())

# 'ssp585', 'ssp126', 'amip'
# 'Amon', 'tas'


#-------------------------------- check
with open(outf, 'rb') as f: dsets = pickle.load(f)
with open(outf_rgd, 'rb') as f: dsets_rgd = pickle.load(f)
with open(outf_rgd_alltime, 'rb') as f: dsets_rgd_alltime = pickle.load(f)

for imodel in dsets.keys():
    # imodel = 'ACCESS-ESM1-5'
    print('#---------------- ' + imodel)
    
    print(str(dsets[imodel].time[0].values)[:10] + ' to ' + str(dsets[imodel].time[-1].values)[:10] + ' ' + str(len(dsets[imodel].time)/12) + ' ' + str(dsets[imodel][var_id].shape))
    print(str(dsets_rgd[imodel].time[0].values)[:10] + ' to ' + str(dsets_rgd[imodel].time[-1].values)[:10] + ' ' + str(len(dsets_rgd[imodel].time)/12) + ' ' + str(dsets_rgd[imodel][var_id].shape))
    print(str(dsets_rgd_alltime[imodel]['mon'].time[0].values)[:10] + ' to ' + str(dsets_rgd_alltime[imodel]['mon'].time[-1].values)[:10] + ' ' + str(len(dsets_rgd_alltime[imodel]['mon'].time)/12) + ' ' + str(dsets_rgd_alltime[imodel]['mon'].shape))
    print(dsets_rgd_alltime[imodel].keys())
    
    # print(dsets[imodel])
    # print(dsets_rgd[imodel])
    # print(dsets_rgd_alltime[imodel])

import psutil
process = psutil.Process()

'''
# endregion


# region get 'historical', 'Amon', 'pr'

# configurations
exp_id = 'historical'
table_id = 'Amon'
var_id = 'pr'

esm_data = esm_datastore.search(**{
    'experiment_id': exp_id, 'table_id': table_id, 'variable_id': var_id,
    'member_id': ['r1i1p1f1', 'r1i1p1f2', 'r1i1p2f1', 'r2i1p1f1', 'r1i1p1f3',],})
esm_data_subset = esm_data.df.sort_values(
    ['source_id', 'member_id', 'grid_label', 'version'],
    ascending=[True, True, True, False]).groupby('source_id').first()

outf = f'{ffolder}{exp_id}_{table_id}_{var_id}.pkl'
outf_rgd = f'{ffolder}{exp_id}_{table_id}_{var_id}_rgd.pkl'
outf_rgd_alltime = f'{ffolder}{exp_id}_{table_id}_{var_id}_rgd_alltime.pkl'

dsets = {}
for group, df in esm_data_subset.groupby('source_id'):
    dsets[group] = open_delayed(df)

dsets = dask.compute(dsets)[0]
with open(outf, 'wb') as f: pickle.dump(dsets, f)

dsets_rgd = {}
for imodel in dsets.keys():
    print(imodel)
    if (len(dsets[imodel].sel(time=slice('1979', '2014')).time) > 0):
        dsets_rgd[imodel] = cdo_regrid(dsets[imodel].sel(time=slice('1979', '2014'))).pipe(combined_preprocessing).pipe(drop_all_bounds)

del dsets
with open(outf_rgd, 'wb') as f: pickle.dump(dsets_rgd, f)

print('get mon_sea_ann regridded data')
dsets_rgd_alltime = {}
for imodel in dsets_rgd.keys():
    print(imodel)
    dsets_rgd_alltime[imodel] = mon_sea_ann(
        var_monthly=dsets_rgd[imodel][var_id], lcopy=False)

del dsets_rgd
with open(outf_rgd_alltime, 'wb') as f: pickle.dump(dsets_rgd_alltime, f)


'''
with open('data/sim/cmip6/test/historical_Amon_pr.pkl', 'rb') as f:
    historical_Amon_pr = pickle.load(f)

print(esm_data)
print(esm_data.df)
esm_data.df.groupby("source_id").first()
esm_data.df.groupby("source_id").nunique()
print(esm_data.df.sort_values(
    ['source_id', 'member_id', 'grid_label', 'version'],
    ascending=[True, True, True, False])[['source_id', 'member_id', 'grid_label', 'version']].to_string())

# 'ssp585', 'ssp126', 'amip'
# 'Amon', 'tas'


#-------------------------------- check
with open(outf, 'rb') as f: dsets = pickle.load(f)
with open(outf_rgd, 'rb') as f: dsets_rgd = pickle.load(f)
with open(outf_rgd_alltime, 'rb') as f: dsets_rgd_alltime = pickle.load(f)

for imodel in dsets.keys():
    # imodel = 'ACCESS-ESM1-5'
    print('#---------------- ' + imodel)
    
    print(str(dsets[imodel].time[0].values)[:10] + ' to ' + str(dsets[imodel].time[-1].values)[:10] + ' ' + str(len(dsets[imodel].time)/12) + ' ' + str(dsets[imodel][var_id].shape))
    print(str(dsets_rgd[imodel].time[0].values)[:10] + ' to ' + str(dsets_rgd[imodel].time[-1].values)[:10] + ' ' + str(len(dsets_rgd[imodel].time)/12) + ' ' + str(dsets_rgd[imodel][var_id].shape))
    print(str(dsets_rgd_alltime[imodel]['mon'].time[0].values)[:10] + ' to ' + str(dsets_rgd_alltime[imodel]['mon'].time[-1].values)[:10] + ' ' + str(len(dsets_rgd_alltime[imodel]['mon'].time)/12) + ' ' + str(dsets_rgd_alltime[imodel]['mon'].shape))
    print(dsets_rgd_alltime[imodel].keys())
    
    # print(dsets[imodel])
    # print(dsets_rgd[imodel])
    # print(dsets_rgd_alltime[imodel])

import psutil
process = psutil.Process()

'''
# endregion


# ssp585
# region get 'ssp585', 'Omon', 'tos'

# configurations
exp_id = 'ssp585'
table_id = 'Omon'
var_id = 'tos'

esm_data = esm_datastore.search(**{
    'experiment_id': exp_id, 'table_id': table_id, 'variable_id': var_id,
    'member_id': ['r1i1p1f1', 'r1i1p1f2', 'r1i1p2f1', 'r2i1p1f1', 'r1i1p1f3', 'r4i1p1f1', 'r1i1p3f1'],})
esm_data_subset = esm_data.df.sort_values(
    ['source_id', 'member_id', 'grid_label', 'version'],
    ascending=[True, True, True, False]).groupby('source_id').first()

outf = f'{ffolder}{exp_id}_{table_id}_{var_id}.pkl'
outf_rgd = f'{ffolder}{exp_id}_{table_id}_{var_id}_rgd.pkl'
outf_rgd_alltime = f'{ffolder}{exp_id}_{table_id}_{var_id}_rgd_alltime.pkl'

dsets = {}
for group, df in esm_data_subset.groupby('source_id'):
    dsets[group] = open_delayed(df)

dsets = dask.compute(dsets)[0]
with open(outf, 'wb') as f: pickle.dump(dsets, f)

dsets_rgd = {}
for imodel in dsets.keys():
    print(imodel)
    if (len(dsets[imodel].sel(time=slice('2015', '2100')).time) > 0):
        dsets_rgd[imodel] = cdo_regrid(dsets[imodel].sel(time=slice('2015', '2100'))).pipe(combined_preprocessing).pipe(drop_all_bounds)

del dsets
with open(outf_rgd, 'wb') as f: pickle.dump(dsets_rgd, f)

print('get mon_sea_ann regridded data')
dsets_rgd_alltime = {}
for imodel in dsets_rgd.keys():
    print(imodel)
    dsets_rgd_alltime[imodel] = mon_sea_ann(
        var_monthly=dsets_rgd[imodel][var_id], lcopy=False)

del dsets_rgd
with open(outf_rgd_alltime, 'wb') as f: pickle.dump(dsets_rgd_alltime, f)


'''
with open('data/sim/cmip6/ssp585_Omon_tos_rgd_alltime.pkl', 'rb') as f:
    ssp585_Omon_tos = pickle.load(f)

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
with open(outf, 'rb') as f: dsets = pickle.load(f)
with open(outf_rgd, 'rb') as f: dsets_rgd = pickle.load(f)
with open(outf_rgd_alltime, 'rb') as f: dsets_rgd_alltime = pickle.load(f)

for imodel in dsets.keys():
    # imodel = 'ACCESS-ESM1-5'
    print('#---------------- ' + imodel)
    
    print(str(dsets[imodel].time[0].values)[:10] + ' to ' + str(dsets[imodel].time[-1].values)[:10] + ' ' + str(len(dsets[imodel].time)/12) + ' ' + str(dsets[imodel][var_id].shape))
    print(str(dsets_rgd[imodel].time[0].values)[:10] + ' to ' + str(dsets_rgd[imodel].time[-1].values)[:10] + ' ' + str(len(dsets_rgd[imodel].time)/12) + ' ' + str(dsets_rgd[imodel][var_id].shape))
    print(str(dsets_rgd_alltime[imodel]['mon'].time[0].values)[:10] + ' to ' + str(dsets_rgd_alltime[imodel]['mon'].time[-1].values)[:10] + ' ' + str(len(dsets_rgd_alltime[imodel]['mon'].time)/12) + ' ' + str(dsets_rgd_alltime[imodel]['mon'].shape))
    print(dsets_rgd_alltime[imodel].keys())
    
    # print(dsets[imodel])
    # print(dsets_rgd[imodel])
    # print(dsets_rgd_alltime[imodel])

import psutil
process = psutil.Process()

'''
# endregion


# region get 'ssp585', 'Amon', 'tas'

# configurations
exp_id = 'ssp585'
table_id = 'Amon'
var_id = 'tas'

esm_data = esm_datastore.search(**{
    'experiment_id': exp_id, 'table_id': table_id, 'variable_id': var_id,
    'member_id': ['r1i1p1f1', 'r1i1p1f2', 'r1i1p2f1', 'r2i1p1f1', 'r1i1p1f3', 'r4i1p1f1', 'r1i1p3f1'],})
esm_data_subset = esm_data.df.sort_values(
    ['source_id', 'member_id', 'grid_label', 'version'],
    ascending=[True, True, True, False]).groupby('source_id').first()

outf = f'{ffolder}{exp_id}_{table_id}_{var_id}.pkl'
outf_rgd = f'{ffolder}{exp_id}_{table_id}_{var_id}_rgd.pkl'
outf_rgd_alltime = f'{ffolder}{exp_id}_{table_id}_{var_id}_rgd_alltime.pkl'

dsets = {}
for group, df in esm_data_subset.groupby('source_id'):
    dsets[group] = open_delayed(df)

dsets = dask.compute(dsets)[0]
with open(outf, 'wb') as f: pickle.dump(dsets, f)

dsets_rgd = {}
for imodel in dsets.keys():
    print(imodel)
    if (len(dsets[imodel].sel(time=slice('2015', '2100')).time) > 0):
        dsets_rgd[imodel] = cdo_regrid(dsets[imodel].sel(time=slice('2015', '2100'))).pipe(combined_preprocessing).pipe(drop_all_bounds)

del dsets
with open(outf_rgd, 'wb') as f: pickle.dump(dsets_rgd, f)

print('get mon_sea_ann regridded data')
dsets_rgd_alltime = {}
for imodel in dsets_rgd.keys():
    print(imodel)
    dsets_rgd_alltime[imodel] = mon_sea_ann(
        var_monthly=dsets_rgd[imodel][var_id], lcopy=False)

del dsets_rgd
with open(outf_rgd_alltime, 'wb') as f: pickle.dump(dsets_rgd_alltime, f)


'''
with open('data/sim/cmip6/ssp585_Amon_tas_rgd_alltime.pkl', 'rb') as f:
    ssp585_Amon_tas = pickle.load(f)

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
with open(outf, 'rb') as f: dsets = pickle.load(f)
with open(outf_rgd, 'rb') as f: dsets_rgd = pickle.load(f)
with open(outf_rgd_alltime, 'rb') as f: dsets_rgd_alltime = pickle.load(f)

for imodel in dsets.keys():
    # imodel = 'ACCESS-ESM1-5'
    print('#---------------- ' + imodel)
    
    print(str(dsets[imodel].time[0].values)[:10] + ' to ' + str(dsets[imodel].time[-1].values)[:10] + ' ' + str(len(dsets[imodel].time)/12) + ' ' + str(dsets[imodel][var_id].shape))
    print(str(dsets_rgd[imodel].time[0].values)[:10] + ' to ' + str(dsets_rgd[imodel].time[-1].values)[:10] + ' ' + str(len(dsets_rgd[imodel].time)/12) + ' ' + str(dsets_rgd[imodel][var_id].shape))
    print(str(dsets_rgd_alltime[imodel]['mon'].time[0].values)[:10] + ' to ' + str(dsets_rgd_alltime[imodel]['mon'].time[-1].values)[:10] + ' ' + str(len(dsets_rgd_alltime[imodel]['mon'].time)/12) + ' ' + str(dsets_rgd_alltime[imodel]['mon'].shape))
    print(dsets_rgd_alltime[imodel].keys())
    
    # print(dsets[imodel])
    # print(dsets_rgd[imodel])
    # print(dsets_rgd_alltime[imodel])

import psutil
process = psutil.Process()

'''
# endregion


# region get 'ssp585', 'Amon', 'pr'

# configurations
exp_id = 'ssp585'
table_id = 'Amon'
var_id = 'pr'

esm_data = esm_datastore.search(**{
    'experiment_id': exp_id, 'table_id': table_id, 'variable_id': var_id,
    'member_id': ['r1i1p1f1', 'r1i1p1f2', 'r1i1p2f1', 'r2i1p1f1', 'r1i1p1f3', 'r4i1p1f1', 'r1i1p3f1'],})
esm_data_subset = esm_data.df.sort_values(
    ['source_id', 'member_id', 'grid_label', 'version'],
    ascending=[True, True, True, False]).groupby('source_id').first()

outf = f'{ffolder}{exp_id}_{table_id}_{var_id}.pkl'
outf_rgd = f'{ffolder}{exp_id}_{table_id}_{var_id}_rgd.pkl'
outf_rgd_alltime = f'{ffolder}{exp_id}_{table_id}_{var_id}_rgd_alltime.pkl'

dsets = {}
for group, df in esm_data_subset.groupby('source_id'):
    dsets[group] = open_delayed(df)

dsets = dask.compute(dsets)[0]
with open(outf, 'wb') as f: pickle.dump(dsets, f)

dsets_rgd = {}
for imodel in dsets.keys():
    print(imodel)
    if (len(dsets[imodel].sel(time=slice('2015', '2100')).time) > 0):
        dsets_rgd[imodel] = cdo_regrid(dsets[imodel].sel(time=slice('2015', '2100'))).pipe(combined_preprocessing).pipe(drop_all_bounds)

del dsets
with open(outf_rgd, 'wb') as f: pickle.dump(dsets_rgd, f)

print('get mon_sea_ann regridded data')
dsets_rgd_alltime = {}
for imodel in dsets_rgd.keys():
    print(imodel)
    dsets_rgd_alltime[imodel] = mon_sea_ann(
        var_monthly=dsets_rgd[imodel][var_id], lcopy=False)

del dsets_rgd
with open(outf_rgd_alltime, 'wb') as f: pickle.dump(dsets_rgd_alltime, f)


'''
with open('data/sim/cmip6/ssp585_Amon_pr_rgd_alltime.pkl', 'rb') as f:
    ssp585_Amon_pr = pickle.load(f)

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
with open(outf, 'rb') as f: dsets = pickle.load(f)
with open(outf_rgd, 'rb') as f: dsets_rgd = pickle.load(f)
with open(outf_rgd_alltime, 'rb') as f: dsets_rgd_alltime = pickle.load(f)

for imodel in dsets.keys():
    # imodel = 'ACCESS-ESM1-5'
    print('#---------------- ' + imodel)
    
    print(str(dsets[imodel].time[0].values)[:10] + ' to ' + str(dsets[imodel].time[-1].values)[:10] + ' ' + str(len(dsets[imodel].time)/12) + ' ' + str(dsets[imodel][var_id].shape))
    print(str(dsets_rgd[imodel].time[0].values)[:10] + ' to ' + str(dsets_rgd[imodel].time[-1].values)[:10] + ' ' + str(len(dsets_rgd[imodel].time)/12) + ' ' + str(dsets_rgd[imodel][var_id].shape))
    print(str(dsets_rgd_alltime[imodel]['mon'].time[0].values)[:10] + ' to ' + str(dsets_rgd_alltime[imodel]['mon'].time[-1].values)[:10] + ' ' + str(len(dsets_rgd_alltime[imodel]['mon'].time)/12) + ' ' + str(dsets_rgd_alltime[imodel]['mon'].shape))
    print(dsets_rgd_alltime[imodel].keys())
    
    # print(dsets[imodel])
    # print(dsets_rgd[imodel])
    # print(dsets_rgd_alltime[imodel])

import psutil
process = psutil.Process()

'''
# endregion


# ssp126
# region get 'ssp126', 'Omon', 'tos'

# configurations
exp_id = 'ssp126'
table_id = 'Omon'
var_id = 'tos'

esm_data = esm_datastore.search(**{
    'experiment_id': exp_id, 'table_id': table_id, 'variable_id': var_id,
    'member_id': ['r1i1p1f1', 'r1i1p1f2', 'r1i1p2f1', 'r2i1p1f1', 'r1i1p1f3', 'r4i1p1f1', 'r1i1p3f1'],})
esm_data_subset = esm_data.df.sort_values(
    ['source_id', 'member_id', 'grid_label', 'version'],
    ascending=[True, True, True, False]).groupby('source_id').first()

outf = f'{ffolder}{exp_id}_{table_id}_{var_id}.pkl'
outf_rgd = f'{ffolder}{exp_id}_{table_id}_{var_id}_rgd.pkl'
outf_rgd_alltime = f'{ffolder}{exp_id}_{table_id}_{var_id}_rgd_alltime.pkl'

dsets = {}
for group, df in esm_data_subset.groupby('source_id'):
    dsets[group] = open_delayed(df)

dsets = dask.compute(dsets)[0]
with open(outf, 'wb') as f: pickle.dump(dsets, f)

dsets_rgd = {}
for imodel in dsets.keys():
    print(imodel)
    if (len(dsets[imodel].sel(time=slice('2015', '2100')).time) > 0):
        dsets_rgd[imodel] = cdo_regrid(dsets[imodel].sel(time=slice('2015', '2100'))).pipe(combined_preprocessing).pipe(drop_all_bounds)

del dsets
with open(outf_rgd, 'wb') as f: pickle.dump(dsets_rgd, f)

print('get mon_sea_ann regridded data')
dsets_rgd_alltime = {}
for imodel in dsets_rgd.keys():
    print(imodel)
    dsets_rgd_alltime[imodel] = mon_sea_ann(
        var_monthly=dsets_rgd[imodel][var_id], lcopy=False)

del dsets_rgd
with open(outf_rgd_alltime, 'wb') as f: pickle.dump(dsets_rgd_alltime, f)


'''
with open('data/sim/cmip6/ssp126_Omon_tos_rgd_alltime.pkl', 'rb') as f:
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
with open(outf, 'rb') as f: dsets = pickle.load(f)
with open(outf_rgd, 'rb') as f: dsets_rgd = pickle.load(f)
with open(outf_rgd_alltime, 'rb') as f: dsets_rgd_alltime = pickle.load(f)

for imodel in dsets.keys():
    # imodel = 'ACCESS-ESM1-5'
    print('#---------------- ' + imodel)
    
    print(str(dsets[imodel].time[0].values)[:10] + ' to ' + str(dsets[imodel].time[-1].values)[:10] + ' ' + str(len(dsets[imodel].time)/12) + ' ' + str(dsets[imodel][var_id].shape))
    print(str(dsets_rgd[imodel].time[0].values)[:10] + ' to ' + str(dsets_rgd[imodel].time[-1].values)[:10] + ' ' + str(len(dsets_rgd[imodel].time)/12) + ' ' + str(dsets_rgd[imodel][var_id].shape))
    print(str(dsets_rgd_alltime[imodel]['mon'].time[0].values)[:10] + ' to ' + str(dsets_rgd_alltime[imodel]['mon'].time[-1].values)[:10] + ' ' + str(len(dsets_rgd_alltime[imodel]['mon'].time)/12) + ' ' + str(dsets_rgd_alltime[imodel]['mon'].shape))
    print(dsets_rgd_alltime[imodel].keys())
    
    # print(dsets[imodel])
    # print(dsets_rgd[imodel])
    # print(dsets_rgd_alltime[imodel])

import psutil
process = psutil.Process()

'''
# endregion


# region get 'ssp126', 'Amon', 'tas'

# configurations
exp_id = 'ssp126'
table_id = 'Amon'
var_id = 'tas'

esm_data = esm_datastore.search(**{
    'experiment_id': exp_id, 'table_id': table_id, 'variable_id': var_id,
    'member_id': ['r1i1p1f1', 'r1i1p1f2', 'r1i1p2f1', 'r2i1p1f1', 'r1i1p1f3', 'r4i1p1f1', 'r1i1p3f1'],})
esm_data_subset = esm_data.df.sort_values(
    ['source_id', 'member_id', 'grid_label', 'version'],
    ascending=[True, True, True, False]).groupby('source_id').first()

outf = f'{ffolder}{exp_id}_{table_id}_{var_id}.pkl'
outf_rgd = f'{ffolder}{exp_id}_{table_id}_{var_id}_rgd.pkl'
outf_rgd_alltime = f'{ffolder}{exp_id}_{table_id}_{var_id}_rgd_alltime.pkl'

dsets = {}
for group, df in esm_data_subset.groupby('source_id'):
    dsets[group] = open_delayed(df)

dsets = dask.compute(dsets)[0]
with open(outf, 'wb') as f: pickle.dump(dsets, f)

dsets_rgd = {}
for imodel in dsets.keys():
    print(imodel)
    if (len(dsets[imodel].sel(time=slice('2015', '2100')).time) > 0):
        dsets_rgd[imodel] = cdo_regrid(dsets[imodel].sel(time=slice('2015', '2100'))).pipe(combined_preprocessing).pipe(drop_all_bounds)

del dsets
with open(outf_rgd, 'wb') as f: pickle.dump(dsets_rgd, f)

print('get mon_sea_ann regridded data')
dsets_rgd_alltime = {}
for imodel in dsets_rgd.keys():
    print(imodel)
    dsets_rgd_alltime[imodel] = mon_sea_ann(
        var_monthly=dsets_rgd[imodel][var_id], lcopy=False)

del dsets_rgd
with open(outf_rgd_alltime, 'wb') as f: pickle.dump(dsets_rgd_alltime, f)


'''
with open('data/sim/cmip6/ssp126_Amon_tas_rgd_alltime.pkl', 'rb') as f:
    ssp126_Amon_tas = pickle.load(f)


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
with open(outf, 'rb') as f: dsets = pickle.load(f)
with open(outf_rgd, 'rb') as f: dsets_rgd = pickle.load(f)
with open(outf_rgd_alltime, 'rb') as f: dsets_rgd_alltime = pickle.load(f)

for imodel in dsets.keys():
    # imodel = 'ACCESS-ESM1-5'
    print('#---------------- ' + imodel)
    
    print(str(dsets[imodel].time[0].values)[:10] + ' to ' + str(dsets[imodel].time[-1].values)[:10] + ' ' + str(len(dsets[imodel].time)/12) + ' ' + str(dsets[imodel][var_id].shape))
    print(str(dsets_rgd[imodel].time[0].values)[:10] + ' to ' + str(dsets_rgd[imodel].time[-1].values)[:10] + ' ' + str(len(dsets_rgd[imodel].time)/12) + ' ' + str(dsets_rgd[imodel][var_id].shape))
    print(str(dsets_rgd_alltime[imodel]['mon'].time[0].values)[:10] + ' to ' + str(dsets_rgd_alltime[imodel]['mon'].time[-1].values)[:10] + ' ' + str(len(dsets_rgd_alltime[imodel]['mon'].time)/12) + ' ' + str(dsets_rgd_alltime[imodel]['mon'].shape))
    print(dsets_rgd_alltime[imodel].keys())
    
    # print(dsets[imodel])
    # print(dsets_rgd[imodel])
    # print(dsets_rgd_alltime[imodel])

import psutil
process = psutil.Process()

'''
# endregion


# region get 'ssp126', 'Amon', 'pr'

# configurations
exp_id = 'ssp126'
table_id = 'Amon'
var_id = 'pr'

esm_data = esm_datastore.search(**{
    'experiment_id': exp_id, 'table_id': table_id, 'variable_id': var_id,
    'member_id': ['r1i1p1f1', 'r1i1p1f2', 'r1i1p2f1', 'r2i1p1f1', 'r1i1p1f3', 'r4i1p1f1', 'r1i1p3f1'],})
esm_data_subset = esm_data.df.sort_values(
    ['source_id', 'member_id', 'grid_label', 'version'],
    ascending=[True, True, True, False]).groupby('source_id').first()

outf = f'{ffolder}{exp_id}_{table_id}_{var_id}.pkl'
outf_rgd = f'{ffolder}{exp_id}_{table_id}_{var_id}_rgd.pkl'
outf_rgd_alltime = f'{ffolder}{exp_id}_{table_id}_{var_id}_rgd_alltime.pkl'

dsets = {}
for group, df in esm_data_subset.groupby('source_id'):
    dsets[group] = open_delayed(df)

dsets = dask.compute(dsets)[0]
with open(outf, 'wb') as f: pickle.dump(dsets, f)

dsets_rgd = {}
for imodel in dsets.keys():
    print(imodel)
    if (len(dsets[imodel].sel(time=slice('2015', '2100')).time) > 0):
        dsets_rgd[imodel] = cdo_regrid(dsets[imodel].sel(time=slice('2015', '2100'))).pipe(combined_preprocessing).pipe(drop_all_bounds)

del dsets
with open(outf_rgd, 'wb') as f: pickle.dump(dsets_rgd, f)

print('get mon_sea_ann regridded data')
dsets_rgd_alltime = {}
for imodel in dsets_rgd.keys():
    print(imodel)
    dsets_rgd_alltime[imodel] = mon_sea_ann(
        var_monthly=dsets_rgd[imodel][var_id], lcopy=False)

del dsets_rgd
with open(outf_rgd_alltime, 'wb') as f: pickle.dump(dsets_rgd_alltime, f)


'''
with open('data/sim/cmip6/ssp126_Amon_pr_rgd_alltime.pkl', 'rb') as f:
    ssp126_Amon_pr = pickle.load(f)


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
with open(outf, 'rb') as f: dsets = pickle.load(f)
with open(outf_rgd, 'rb') as f: dsets_rgd = pickle.load(f)
with open(outf_rgd_alltime, 'rb') as f: dsets_rgd_alltime = pickle.load(f)

for imodel in dsets.keys():
    # imodel = 'ACCESS-ESM1-5'
    print('#---------------- ' + imodel)
    
    print(str(dsets[imodel].time[0].values)[:10] + ' to ' + str(dsets[imodel].time[-1].values)[:10] + ' ' + str(len(dsets[imodel].time)/12) + ' ' + str(dsets[imodel][var_id].shape))
    print(str(dsets_rgd[imodel].time[0].values)[:10] + ' to ' + str(dsets_rgd[imodel].time[-1].values)[:10] + ' ' + str(len(dsets_rgd[imodel].time)/12) + ' ' + str(dsets_rgd[imodel][var_id].shape))
    print(str(dsets_rgd_alltime[imodel]['mon'].time[0].values)[:10] + ' to ' + str(dsets_rgd_alltime[imodel]['mon'].time[-1].values)[:10] + ' ' + str(len(dsets_rgd_alltime[imodel]['mon'].time)/12) + ' ' + str(dsets_rgd_alltime[imodel]['mon'].shape))
    print(dsets_rgd_alltime[imodel].keys())
    
    # print(dsets[imodel])
    # print(dsets_rgd[imodel])
    # print(dsets_rgd_alltime[imodel])

import psutil
process = psutil.Process()

'''
# endregion


# amip
# region get 'amip', 'Amon', 'tas'

# configurations
exp_id = 'amip'
table_id = 'Amon'
var_id = 'tas'

esm_data = esm_datastore.search(**{
    'experiment_id': exp_id, 'table_id': table_id, 'variable_id': var_id,
    'member_id': ['r1i1p1f1', 'r1i1p1f2', 'r1i1p2f1', 'r2i1p1f1', 'r1i1p1f3', 'r4i1p1f1', 'r1i1p3f1', 'r1i1p1f4'],})
esm_data_subset = esm_data.df.sort_values(
    ['source_id', 'member_id', 'grid_label', 'version'],
    ascending=[True, True, True, False]).groupby('source_id').first()

outf = f'{ffolder}{exp_id}_{table_id}_{var_id}.pkl'
outf_rgd = f'{ffolder}{exp_id}_{table_id}_{var_id}_rgd.pkl'
outf_rgd_alltime = f'{ffolder}{exp_id}_{table_id}_{var_id}_rgd_alltime.pkl'

dsets = {}
for group, df in esm_data_subset.groupby('source_id'):
    dsets[group] = open_delayed(df)

dsets = dask.compute(dsets)[0]
with open(outf, 'wb') as f: pickle.dump(dsets, f)

dsets_rgd = {}
for imodel in dsets.keys():
    print(imodel)
    if (len(dsets[imodel].sel(time=slice('1979', '2014')).time) > 0):
        dsets_rgd[imodel] = cdo_regrid(dsets[imodel].sel(time=slice('1979', '2014'))).pipe(combined_preprocessing).pipe(drop_all_bounds)

del dsets
with open(outf_rgd, 'wb') as f: pickle.dump(dsets_rgd, f)

print('get mon_sea_ann regridded data')
dsets_rgd_alltime = {}
for imodel in dsets_rgd.keys():
    print(imodel)
    dsets_rgd_alltime[imodel] = mon_sea_ann(
        var_monthly=dsets_rgd[imodel][var_id], lcopy=False)

del dsets_rgd
with open(outf_rgd_alltime, 'wb') as f: pickle.dump(dsets_rgd_alltime, f)


'''
with open('data/sim/cmip6/amip_Amon_tas_rgd_alltime.pkl', 'rb') as f:
    amip_Amon_tas = pickle.load(f)

print(esm_data)
print(esm_data.df)
esm_data.df.groupby("source_id").first()
esm_data.df.groupby("source_id").nunique()
print(esm_data.df.sort_values(
    ['source_id', 'member_id', 'grid_label', 'version'],
    ascending=[True, True, True, False])[['source_id', 'member_id', 'grid_label', 'version']].to_string())

# 'ssp585', 'ssp126', 'amip'
# 'Amon', 'tas'


#-------------------------------- check
with open(outf, 'rb') as f: dsets = pickle.load(f)
with open(outf_rgd, 'rb') as f: dsets_rgd = pickle.load(f)
with open(outf_rgd_alltime, 'rb') as f: dsets_rgd_alltime = pickle.load(f)

for imodel in dsets.keys():
    # imodel = 'ACCESS-ESM1-5'
    print('#---------------- ' + imodel)
    
    print(str(dsets[imodel].time[0].values)[:10] + ' to ' + str(dsets[imodel].time[-1].values)[:10] + ' ' + str(len(dsets[imodel].time)/12) + ' ' + str(dsets[imodel][var_id].shape))
    print(str(dsets_rgd[imodel].time[0].values)[:10] + ' to ' + str(dsets_rgd[imodel].time[-1].values)[:10] + ' ' + str(len(dsets_rgd[imodel].time)/12) + ' ' + str(dsets_rgd[imodel][var_id].shape))
    print(str(dsets_rgd_alltime[imodel]['mon'].time[0].values)[:10] + ' to ' + str(dsets_rgd_alltime[imodel]['mon'].time[-1].values)[:10] + ' ' + str(len(dsets_rgd_alltime[imodel]['mon'].time)/12) + ' ' + str(dsets_rgd_alltime[imodel]['mon'].shape))
    print(dsets_rgd_alltime[imodel].keys())
    
    # print(dsets[imodel])
    # print(dsets_rgd[imodel])
    # print(dsets_rgd_alltime[imodel])


'''
# endregion


# region get 'amip', 'Amon', 'pr'

# configurations
exp_id = 'amip'
table_id = 'Amon'
var_id = 'pr'

esm_data = esm_datastore.search(**{
    'experiment_id': exp_id, 'table_id': table_id, 'variable_id': var_id,
    'member_id': ['r1i1p1f1', 'r1i1p1f2', 'r1i1p2f1', 'r2i1p1f1', 'r1i1p1f3', 'r4i1p1f1', 'r1i1p3f1', 'r1i1p1f4', 'r2i1p1f2'],})
esm_data_subset = esm_data.df.sort_values(
    ['source_id', 'member_id', 'grid_label', 'version'],
    ascending=[True, True, True, False]).groupby('source_id').first()

outf = f'{ffolder}{exp_id}_{table_id}_{var_id}.pkl'
outf_rgd = f'{ffolder}{exp_id}_{table_id}_{var_id}_rgd.pkl'
outf_rgd_alltime = f'{ffolder}{exp_id}_{table_id}_{var_id}_rgd_alltime.pkl'

dsets = {}
for group, df in esm_data_subset.groupby('source_id'):
    dsets[group] = open_delayed(df)

dsets = dask.compute(dsets)[0]
with open(outf, 'wb') as f: pickle.dump(dsets, f)

dsets_rgd = {}
for imodel in dsets.keys():
    print(imodel)
    if (len(dsets[imodel].sel(time=slice('1979', '2014')).time) > 0):
        dsets_rgd[imodel] = cdo_regrid(dsets[imodel].sel(time=slice('1979', '2014'))).pipe(combined_preprocessing).pipe(drop_all_bounds)

del dsets
with open(outf_rgd, 'wb') as f: pickle.dump(dsets_rgd, f)

print('get mon_sea_ann regridded data')
dsets_rgd_alltime = {}
for imodel in dsets_rgd.keys():
    print(imodel)
    dsets_rgd_alltime[imodel] = mon_sea_ann(
        var_monthly=dsets_rgd[imodel][var_id], lcopy=False)

del dsets_rgd
with open(outf_rgd_alltime, 'wb') as f: pickle.dump(dsets_rgd_alltime, f)


'''
with open('data/sim/cmip6/amip_Amon_pr_rgd_alltime.pkl', 'rb') as f:
    amip_Amon_pr = pickle.load(f)

print(esm_data)
print(esm_data.df)
esm_data.df.groupby("source_id").first()
esm_data.df.groupby("source_id").nunique()
print(esm_data.df.sort_values(
    ['source_id', 'member_id', 'grid_label', 'version'],
    ascending=[True, True, True, False])[['source_id', 'member_id', 'grid_label', 'version']].to_string())

# 'ssp585', 'ssp126', 'amip'
# 'Amon', 'tas'


#-------------------------------- check
with open(outf, 'rb') as f: dsets = pickle.load(f)
with open(outf_rgd, 'rb') as f: dsets_rgd = pickle.load(f)
with open(outf_rgd_alltime, 'rb') as f: dsets_rgd_alltime = pickle.load(f)

for imodel in dsets.keys():
    # imodel = 'ACCESS-ESM1-5'
    print('#---------------- ' + imodel)
    
    print(str(dsets[imodel].time[0].values)[:10] + ' to ' + str(dsets[imodel].time[-1].values)[:10] + ' ' + str(len(dsets[imodel].time)/12) + ' ' + str(dsets[imodel][var_id].shape))
    print(str(dsets_rgd[imodel].time[0].values)[:10] + ' to ' + str(dsets_rgd[imodel].time[-1].values)[:10] + ' ' + str(len(dsets_rgd[imodel].time)/12) + ' ' + str(dsets_rgd[imodel][var_id].shape))
    print(str(dsets_rgd_alltime[imodel]['mon'].time[0].values)[:10] + ' to ' + str(dsets_rgd_alltime[imodel]['mon'].time[-1].values)[:10] + ' ' + str(len(dsets_rgd_alltime[imodel]['mon'].time)/12) + ' ' + str(dsets_rgd_alltime[imodel]['mon'].shape))
    print(dsets_rgd_alltime[imodel].keys())
    
    # print(dsets[imodel])
    # print(dsets_rgd[imodel])
    # print(dsets_rgd_alltime[imodel])


'''
# endregion


