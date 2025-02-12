

# when adding new variables:
# 1) update cmip6_units in namelist.py
# 2) check <change the units and sign convention>


# qsub -I -q normal -l walltime=4:00:00,ncpus=1,mem=192GB,jobfs=10GB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38


# region import packages

# data analysis
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
import pandas as pd
import intake
from cdo import Cdo
cdo=Cdo()
import xarray as xr
# from dask.diagnostics import ProgressBar
# pbar = ProgressBar()
# pbar.register()
import numpy as np
import os

# management
import sys  # print(sys.path)
sys.path.append('/home/563/qg8515/code/gbr_future/module')
import pickle
import psutil
process = psutil.Process()
# print(process.memory_info().rss / 2**30)
import gc
import warnings
warnings.filterwarnings('ignore')

# self defined function
from calculations import (
    mon_sea_ann,
    cdo_regrid,
    )
from xmip.preprocessing import rename_cmip6, broadcast_lonlat, correct_lon, promote_empty_dims, replace_x_y_nominal_lat_lon, correct_units, correct_coordinates, parse_lon_lat_bounds, maybe_convert_bounds_to_vertex, maybe_convert_vertex_to_bounds, combined_preprocessing

from namelist import cmip6_units, zerok, seconds_per_d


'''
cmip_info['experiment_id'].unique()
cmip_info['institution_id'].unique()
'''
# endregion


# region get source_ids and preferred member_ids

cmip6 = intake.open_catalog('/g/data/hh5/public/apps/nci-intake-catalogue/catalogue_new.yaml').esgf.cmip6

# get the exps
source_intersect = cmip6.df.source_id.unique()

for experiment_id in [['piControl', 'esm-piControl'], ['abrupt-4xCO2'], ['historical', 'esm-hist']]:
    print(experiment_id)
    for table_id, variable_id in zip(['Amon', 'Amon', 'Amon', 'Amon'], ['tas', 'rsut', 'rsdt', 'rlut']):
        print(f'{table_id} {variable_id}')
        
        source_intersect = sorted(set(source_intersect) & set(cmip6.search(experiment_id=experiment_id, table_id=table_id, variable_id=variable_id).df.source_id.unique()))

print(len(source_intersect))

# get the members
exp_counts = cmip6.search(source_id=source_intersect, experiment_id=['piControl', 'amip', 'abrupt-4xCO2', 'historical', 'ssp585', 'esm-piControl', 'esm-hist', 'esm-ssp585']).df.groupby(['source_id', 'member_id']).experiment_id.nunique()

cmip6_ids = {
    source_id: member_id
    for source_id, member_id in exp_counts.groupby(level=0).idxmax().values}

with open('scratch/data/sim/cmip6/cmip6_ids.pkl', 'wb') as f:
    pickle.dump(cmip6_ids, f)



'''
# check
with open('scratch/data/sim/cmip6/cmip6_ids.pkl', 'rb') as f:
    cmip6_ids = pickle.load(f)
for source_id, member_id in cmip6_ids.items():
    # source_id = source_ids[0]; member_id = member_ids[0]
    nexp = cmip6.search(source_id=source_id, member_id=member_id, experiment_id=['piControl', 'amip', 'abrupt-4xCO2', 'historical', 'ssp585', 'esm-piControl', 'esm-hist', 'esm-ssp585']).df.experiment_id.nunique()
    print(f'{source_id}  {member_id} {nexp} {exp_counts.groupby(level=0).max()[source_id]}')

# check what is missing in removed source
source_subset = sorted(cmip6.search(experiment_id=['piControl', 'amip', 'abrupt-4xCO2', 'historical', 'ssp585', 'esm-piControl', 'esm-hist', 'esm-ssp585']).df.source_id.unique()) #72
source_removed = sorted(set(source_subset) - set(source_intersect)) #22

for source_id in ['CAS-ESM2-0', 'ICON-ESM-LR', 'MCM-UA-1-0']:
    print(f'#---------------- {source_id}')
    for experiment_id in [['piControl', 'esm-piControl'], ['abrupt-4xCO2'], ['historical', 'esm-hist']]:
        print(experiment_id)
        for table_id, variable_id in zip(['Amon', 'Amon', 'Amon', 'Amon', 'Amon'], ['tas', 'rsut', 'rsdt', 'rlut', 'pr', ]):
            print(f'{table_id} {variable_id}')
            print(cmip6.search(source_id=source_id, experiment_id=experiment_id, table_id=table_id, variable_id=variable_id).df.shape)


source_piControl = cmip6.search(experiment_id=['piControl', 'esm-piControl'], table_id='Amon', variable_id='tas').df.source_id.unique()
source_abrupt4CO2 = cmip6.search(experiment_id=['abrupt-4xCO2'], table_id='Amon', variable_id='tas').df.source_id.unique()
source_historical = cmip6.search(experiment_id=['historical', 'esm-hist'], table_id='Amon', variable_id='tas').df.source_id.unique()
source_intersect = sorted(set(source_piControl) & set(source_abrupt4CO2) & set(source_historical))
print(len(source_intersect))
# cmip6.search(experiment_id=['piControl', 'esm-piControl'], table_id='Amon').df.variable_id.unique()

exp_amip = cmip6.search(experiment_id=['amip']).df['source_id'].unique()
exp_ssp585 = cmip6.search(experiment_id=['ssp585', 'esm-ssp585']).df['source_id'].unique()
# & set(exp_amip) & set(exp_ssp585)

set(source_intersect) == set(cmip6.search(experiment_id=['piControl', 'abrupt-4xCO2', 'historical'], table_id='Amon', variable_id=['tas', 'rsut', 'rsdt', 'rlut'], require_all_on=["source_id"]).df.source_id.unique())

'''
# endregion


# region get original data

cmip6 = intake.open_catalog('/g/data/hh5/public/apps/nci-intake-catalogue/catalogue_new.yaml').esgf.cmip6
with open('/home/563/qg8515/scratch/data/sim/cmip6/cmip6_ids.pkl', 'rb') as f:
    cmip6_ids = pickle.load(f)

cmip6_data = {}

for experiment_id in [['ssp585', 'esm-ssp585']]:
    # experiment_id = ['amip']
    # [['piControl', 'esm-piControl'], ['abrupt-4xCO2'], ['historical', 'esm-hist'], ['amip'], ['ssp585', 'esm-ssp585']]
    print(f'#-------------------------------- {experiment_id}')
    cmip6_data[experiment_id[0]] = {}
    
    for table_id, variable_id in zip(['Amon'], ['tas']):
        # table_id = 'Omon'; variable_id = 'tos'
        # ['Amon'], ['tas']
        # ['Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Omon'], ['tas', 'rsut', 'rsdt', 'rlut', 'pr', 'tos']
        # ['Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Amon'], ['clt', 'evspsbl', 'hfls', 'hfss', 'psl', 'rlds', 'rldscs', 'rlus', 'rlutcs', 'rsds', 'rsdscs', 'rsus', 'rsuscs', 'rsutcs']
        print(f'#---------------- {table_id} {variable_id}')
        cmip6_data[experiment_id[0]][table_id]={}
        cmip6_data[experiment_id[0]][table_id][variable_id]={}
        
        for source_id, member_id in cmip6_ids.items():
            # source_id = list(cmip6_ids.keys())[0]; member_id = cmip6_ids[source_id]
            # source_id = 'NorESM2-LM'; member_id = cmip6_ids[source_id]
            # source_id = 'AWI-CM-1-1-MR'; member_id = 'r1i1p1f1'
            # source_id = 'EC-Earth3-AerChem'; member_id = 'r1i1p1f1'
            print(f'#-------- {source_id}  {member_id}')
            
            data_catalogue = cmip6.search(experiment_id=experiment_id, table_id=table_id, variable_id=variable_id, source_id=source_id, member_id=member_id).df
            
            if len(data_catalogue) == 0:
                print('Change to other member_ids')
                data_catalogue = cmip6.search(experiment_id=experiment_id, table_id=table_id, variable_id=variable_id, source_id=source_id).df
                if len(data_catalogue) == 0:
                    print('Warning no data found')
                    continue
            
            # choose the latest version
            if len(data_catalogue.version.unique()) > 1:
                print(f'Choose version: {data_catalogue.version.unique()}')
                version = sorted(data_catalogue.version.unique(), reverse=True)[0]
                data_catalogue=data_catalogue[data_catalogue.version==version]
                print(f'{version} chosen')
            
            # choose grid_label
            if len(data_catalogue.grid_label.unique()) > 1:
                print(f'Choose grid_label: {data_catalogue.grid_label.unique()}')
                grid_label = sorted(data_catalogue.grid_label.unique())[0]
                data_catalogue=data_catalogue[data_catalogue.grid_label==grid_label]
                print(f'{grid_label} chosen')
            
            # choose member_id
            if len(data_catalogue.member_id.unique()) > 1:
                print(f'Choose member_id: {data_catalogue.member_id.unique()}')
                member_id = sorted(data_catalogue.member_id.unique())[0]
                data_catalogue=data_catalogue[data_catalogue.member_id==member_id]
                print(f'{member_id} chosen')
            
            # choose experiment_id
            if len(data_catalogue.experiment_id.unique()) > 1:
                print(f'Choose experiment_id: {data_catalogue.experiment_id.unique()}')
                exp_id = sorted(data_catalogue.experiment_id.unique(), reverse=True)[0]
                data_catalogue=data_catalogue[data_catalogue.experiment_id==exp_id]
                print(f'{exp_id} chosen')
            
            # get data
            try:
                dset = xr.open_mfdataset(sorted(data_catalogue.path.values), use_cftime=True, parallel=True, data_vars='minimal', compat='override', coords='minimal')
                
                if len(dset.time) < 120:
                    print('Warning simulation length less than 10 yrs')
                    continue
                
                cmip6_data[experiment_id[0]][table_id][variable_id][source_id]=dset
            except FileNotFoundError:
                print('Warning no file found')
            except ValueError:
                print('Warning file opening error')
        
        if len(cmip6_data[experiment_id[0]][table_id][variable_id]) > 0:
            ofile = f'/home/563/qg8515/scratch/data/sim/cmip6/{experiment_id[0]}_{table_id}_{variable_id}.pkl'
            if os.path.exists(ofile): os.remove(ofile)
            with open(ofile, 'wb') as f:
                pickle.dump(cmip6_data[experiment_id[0]][table_id][variable_id], f)
            del cmip6_data[experiment_id[0]][table_id][variable_id]




'''
# check availabel data
cmip6 = intake.open_catalog('/g/data/hh5/public/apps/nci-intake-catalogue/catalogue_new.yaml').esgf.cmip6
with open('/home/563/qg8515/scratch/data/sim/cmip6/cmip6_ids.pkl', 'rb') as f:
    cmip6_ids = pickle.load(f)

data_catalogue = cmip6.search(experiment_id=['piControl', 'esm-piControl', 'abrupt-4xCO2', 'historical', 'esm-hist', 'amip', 'ssp585', 'esm-ssp585'], source_id=list(cmip6_ids.keys()), variable_id='rluscs').df
cmip6.search(variable_id='rluscs').df


#-------------------------------- check
cmip6 = intake.open_catalog('/g/data/hh5/public/apps/nci-intake-catalogue/catalogue_new.yaml').esgf.cmip6
with open('/home/563/qg8515/scratch/data/sim/cmip6/cmip6_ids.pkl', 'rb') as f:
    cmip6_ids = pickle.load(f)
cmip6_data = {}

ith_source_id=-1

for experiment_id in [['piControl', 'esm-piControl'], ['abrupt-4xCO2'], ['historical', 'esm-hist'], ['amip'], ['ssp585', 'esm-ssp585']]:
    # experiment_id = ['piControl', 'esm-piControl']
    # [['piControl', 'esm-piControl'], ['abrupt-4xCO2'], ['historical', 'esm-hist'], ['amip'], ['ssp585', 'esm-ssp585']]
    print(f'#-------------------------------- {experiment_id}')
    cmip6_data[experiment_id[0]] = {}
    
    for table_id, variable_id in zip(['Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Omon'], ['tas', 'rsut', 'rsdt', 'rlut', 'pr', 'tos']):
        # table_id = 'Amon'; variable_id = 'tas'
        # ['Amon'], ['tas']
        # ['Omon'], ['tos']
        # ['Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Omon'], ['tas', 'rsut', 'rsdt', 'rlut', 'pr', 'tos']
        print(f'#---------------- {table_id} {variable_id}')
        cmip6_data[experiment_id[0]][table_id]={}
        
        with open(f'/home/563/qg8515/scratch/data/sim/cmip6/{experiment_id[0]}_{table_id}_{variable_id}.pkl', 'rb') as f:
            cmip6_data[experiment_id[0]][table_id][variable_id] = pickle.load(f)
        
        for source_id in cmip6_data[experiment_id[0]][table_id][variable_id].keys():
            print(f'#-------- {source_id}')
            print(cmip6_data[experiment_id[0]][table_id][variable_id][source_id][variable_id].shape)
        
        source_id = list(cmip6_data[experiment_id[0]][table_id][variable_id].keys())[ith_source_id]
        
        data_catalogue = cmip6.search(experiment_id=experiment_id, table_id=table_id, variable_id=variable_id, source_id=source_id, member_id=cmip6_ids[source_id]).df
        if len(data_catalogue) == 0:
            print('Change to other member_ids')
            data_catalogue = cmip6.search(experiment_id=experiment_id, table_id=table_id, variable_id=variable_id, source_id=source_id).df
        
        # choose the latest version
        if len(data_catalogue.version.unique()) > 1:
            print(f'Choose version: {data_catalogue.version.unique()}')
            version = sorted(data_catalogue.version.unique(), reverse=True)[0]
            data_catalogue=data_catalogue[data_catalogue.version==version]
            print(f'{version} chosen')
        
        # choose grid_label
        if len(data_catalogue.grid_label.unique()) > 1:
            print(f'Choose grid_label: {data_catalogue.grid_label.unique()}')
            grid_label = sorted(data_catalogue.grid_label.unique())[0]
            data_catalogue=data_catalogue[data_catalogue.grid_label==grid_label]
            print(f'{grid_label} chosen')
        
        # choose member_id
        if len(data_catalogue.member_id.unique()) > 1:
            print(f'Choose member_id: {data_catalogue.member_id.unique()}')
            member_id = sorted(data_catalogue.member_id.unique())[0]
            data_catalogue=data_catalogue[data_catalogue.member_id==member_id]
            print(f'{member_id} chosen')
        
        # choose experiment_id
        if len(data_catalogue.experiment_id.unique()) > 1:
            print(f'Choose experiment_id: {data_catalogue.experiment_id.unique()}')
            exp_id = sorted(data_catalogue.experiment_id.unique(), reverse=True)[0]
            data_catalogue=data_catalogue[data_catalogue.experiment_id==exp_id]
            print(f'{exp_id} chosen')
        
        dset = xr.open_mfdataset(sorted(data_catalogue.path.values), use_cftime=True, parallel=True)
        
        print(dset[variable_id].shape)
        print(cmip6_data[experiment_id[0]][table_id][variable_id][source_id][variable_id].shape)
        
        del dset, cmip6_data[experiment_id[0]][table_id][variable_id]



# check
experiment_id = ['ssp585', 'esm-ssp585']
table_id = 'Amon'; variable_id = 'rsdt'; source_id = 'CIESM'
# table_id = 'Omon'; variable_id = 'tos'; source_id = 'GISS-E2-1-H'

cmip6_data = {}
cmip6_data[experiment_id[0]] = {}
cmip6_data[experiment_id[0]][table_id]={}
with open(f'scratch/data/sim/cmip6/{experiment_id[0]}_{table_id}_{variable_id}.pkl', 'rb') as f:
    cmip6_data[experiment_id[0]][table_id][variable_id] = pickle.load(f)


cmip6_data[experiment_id[0]][table_id][variable_id][source_id]



'''
# endregion


# region get alltime, regridded_alltime, or regridded_alltime_ens data

cmip6_data = {}
cmip6_data_alltime = {}
cmip6_data_regridded_alltime = {}
cmip6_data_regridded_alltime_ens = {}

for experiment_id in ['piControl', 'historical', 'amip', 'ssp585']:
    # experiment_id = 'piControl'
    # ['piControl', 'abrupt-4xCO2', 'historical', 'amip', 'ssp585']
    print(f'#-------------------------------- {experiment_id}')
    cmip6_data[experiment_id] = {}
    cmip6_data_alltime[experiment_id] = {}
    cmip6_data_regridded_alltime[experiment_id] = {}
    cmip6_data_regridded_alltime_ens[experiment_id] = {}
    
    for table_id, variable_id in zip(['Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Amon'], ['clt', 'evspsbl', 'hfls', 'hfss', 'psl', 'rlds', 'rldscs', 'rlus', 'rlutcs', 'rsds', 'rsdscs', 'rsus', 'rsuscs', 'rsutcs']):
        # table_id = 'Amon'; variable_id = 'tas'
        # ['Amon'], ['tas']
        # ['Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Omon'], ['tas', 'rsut', 'rsdt', 'rlut', 'pr', 'tos']
        # ['Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Amon'], ['clt', 'evspsbl', 'hfls', 'hfss', 'psl', 'rlds', 'rldscs', 'rlus', 'rlutcs', 'rsds', 'rsdscs', 'rsus', 'rsuscs', 'rsutcs']
        print(f'#---------------- {table_id} {variable_id}')
        cmip6_data[experiment_id][table_id]={}
        cmip6_data_alltime[experiment_id][table_id] = {}
        cmip6_data_regridded_alltime[experiment_id][table_id] = {}
        cmip6_data_regridded_alltime_ens[experiment_id][table_id] = {}
        
        cmip6_data_alltime[experiment_id][table_id][variable_id] = {}
        cmip6_data_regridded_alltime[experiment_id][table_id][variable_id] = {}
        cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id] = {}
        
        with open(f'/home/563/qg8515/scratch/data/sim/cmip6/{experiment_id}_{table_id}_{variable_id}.pkl', 'rb') as f:
            cmip6_data[experiment_id][table_id][variable_id] = pickle.load(f)
        
        for source_id in cmip6_data[experiment_id][table_id][variable_id].keys():
            # source_id ='AWI-CM-1-1-MR'
            # source_id ='MPI-ESM-1-2-HAM'
            print(f'#-------- {source_id}')
            dset = cmip6_data[experiment_id][table_id][variable_id][source_id].copy()
            # print((dset[variable_id].time[0].dt.year.values))
            
            # ensure enough simulation length
            if (experiment_id in ['piControl', 'abrupt-4xCO2']) & (len(dset.time) < 150 * 12):
                print('Warning simulation length less than 150 yrs: ignored')
                continue
            elif (experiment_id in ['historical']) & (len(dset.time) < 165 * 12):
                print('Warning simulation length less than 165 yrs: ignored')
                continue
            elif (experiment_id in ['amip']) & (len(dset.time) < 36 * 12):
                print('Warning simulation length less than 36 yrs: ignored')
                continue
            elif (experiment_id in ['ssp585']) & (len(dset.time) < 85 * 12):
                print('Warning simulation length less than 85 yrs: ignored')
                continue
            
            # ensure the last month is Dec
            if dset[variable_id].time[-1].dt.month != 12:
                print('Warning last month is not December')
                continue
            
            # ensure correct periods are selected
            if experiment_id in ['piControl']:
                dset = dset.sel(time=slice(dset.time[-150 * 12], dset.time[-1]))
                if (len(dset.time)/12 != 150) | ((np.max(dset.time.dt.year) - np.min(dset.time.dt.year)) != 149):
                    print(f'Warning differred time length: {len(dset.time)/12} {(np.max(dset.time.dt.year) - np.min(dset.time.dt.year)).values}')
                    continue
                dset = dset.assign_coords(time=pd.date_range(start='1850-01-01', periods=150 * 12, freq='1ME'))
            elif experiment_id in ['abrupt-4xCO2']:
                dset = dset.sel(time=slice(dset.time[0], dset.time[150 * 12-1]))
                if (len(dset.time)/12 != 150) | ((np.max(dset.time.dt.year) - np.min(dset.time.dt.year)) != 149):
                    print(f'Warning differred time length: {len(dset.time)/12} {(np.max(dset.time.dt.year) - np.min(dset.time.dt.year)).values}')
                    continue
                dset = dset.assign_coords(time=pd.date_range(start='1850-01-01', periods=150 * 12, freq='1ME'))
            elif experiment_id in ['historical']:
                dset = dset.sel(time=slice('1850', '2014'))
                if len(dset.time)/12 != 165:
                    print(f'Warning differred time length: {len(dset.time)/12}')
                    continue
                dset = dset.assign_coords(time=pd.date_range(start='1850-01-01', periods=165 * 12, freq='1ME'))
            elif experiment_id in ['amip']:
                dset = dset.sel(time=slice('1979', '2014'))
                if len(dset.time)/12 != 36:
                    print(f'Warning differred time length: {len(dset.time)/12}')
                    continue
                dset = dset.assign_coords(time=pd.date_range(start='1979-01-01', periods=36 * 12, freq='1ME'))
            elif experiment_id in ['ssp585']:
                dset = dset.sel(time=slice('2015', '2099'))
                if len(dset.time)/12 != 85:
                    print(f'Warning differred time length: {len(dset.time)/12}')
                    continue
                dset = dset.assign_coords(time=pd.date_range(start='2015-01-01', periods=85 * 12, freq='1ME'))
            
            # ensure correct units
            if dset[variable_id].units != cmip6_units[variable_id]:
                print(f'Warning inconsistent units: {dset[variable_id].units} rather than {cmip6_units[variable_id]}')
                continue
            
            # change the units and sign convention
            if variable_id in ['tas']:
                # change from K to degC
                dset[variable_id] = dset[variable_id] - zerok
            elif variable_id in ['rsut', 'rlut']:
                # change to era5 convention, downward positive
                dset[variable_id] = dset[variable_id] * (-1)
            elif variable_id in ['pr']:
                # change from mm/s to mm/day
                dset[variable_id] = dset[variable_id] * seconds_per_d
            
            dset = dset.compute()
            # print('calculate mon_sea_ann')
            # cmip6_data_alltime[experiment_id][table_id][variable_id][source_id] = mon_sea_ann(var_monthly=dset.pipe(rename_cmip6).pipe(broadcast_lonlat)[variable_id], lcopy = False, mm=True, sm=True, am=True)
            
            print('calculate regridded mon_sea_ann')
            dsetr = cdo_regrid(dset)
            cmip6_data_regridded_alltime[experiment_id][table_id][variable_id][source_id] = mon_sea_ann(var_monthly=dsetr.pipe(rename_cmip6).pipe(broadcast_lonlat)[variable_id], lcopy = False, mm=True, sm=True, am=True)
            
            del dset, dsetr
            gc.collect()
            print(process.memory_info().rss / 2**30)
        
        # ofile1=f'/home/563/qg8515/scratch/data/sim/cmip6/{experiment_id}_{table_id}_{variable_id}_alltime.pkl'
        # if os.path.exists(ofile1): os.remove(ofile1)
        # with open(ofile1, 'wb') as f:
        #     pickle.dump(cmip6_data_alltime[experiment_id][table_id][variable_id], f)
        
        # ofile2=f'/home/563/qg8515/scratch/data/sim/cmip6/{experiment_id}_{table_id}_{variable_id}_regridded_alltime.pkl'
        # if os.path.exists(ofile2): os.remove(ofile2)
        # with open(ofile2, 'wb') as f:
        #     pickle.dump(cmip6_data_regridded_alltime[experiment_id][table_id][variable_id], f)
        
        source_ids = list(cmip6_data_regridded_alltime[experiment_id][table_id][variable_id].keys())
        source_da = xr.DataArray(source_ids, dims='source_id', coords={'source_id': source_ids})
        
        for ialltime in cmip6_data_regridded_alltime[experiment_id][table_id][variable_id][source_ids[0]].keys():
            # ialltime = 'mon'
            print(f'#-------- {ialltime}')
            cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id][ialltime] = xr.concat([cmip6_data_regridded_alltime[experiment_id][table_id][variable_id][source_id][ialltime] for source_id in source_ids], dim=source_da, coords='minimal', compat='override')
        
        ofile=f'/home/563/qg8515/scratch/data/sim/cmip6/{experiment_id}_{table_id}_{variable_id}_regridded_alltime_ens.pkl'
        if os.path.exists(ofile): os.remove(ofile)
        with open(ofile, 'wb') as f:
            pickle.dump(cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id], f)
        
        del cmip6_data[experiment_id][table_id][variable_id]
        del cmip6_data_alltime[experiment_id][table_id][variable_id]
        del cmip6_data_regridded_alltime[experiment_id][table_id][variable_id]
        del cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id]





'''
#-------------------------------- check
cmip6_data = {}
# cmip6_data_alltime = {}
# cmip6_data_regridded_alltime = {}
cmip6_data_regridded_alltime_ens = {}

ith_source_id=-1

for experiment_id in ['ssp585']:
    # experiment_id = 'historical'
    # ['piControl', 'abrupt-4xCO2', 'historical', 'amip', 'ssp585']
    print(f'#-------------------------------- {experiment_id}')
    cmip6_data[experiment_id] = {}
    # cmip6_data_alltime[experiment_id] = {}
    # cmip6_data_regridded_alltime[experiment_id] = {}
    cmip6_data_regridded_alltime_ens[experiment_id] = {}
    
    for table_id, variable_id in zip(['Omon'], ['tos']):
        # table_id = 'Amon'; variable_id = 'tas'
        # ['Amon'], ['tas']
        # ['Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Omon'], ['tas', 'rsut', 'rsdt', 'rlut', 'pr', 'tos']
        print(f'#---------------- {table_id} {variable_id}')
        cmip6_data[experiment_id][table_id]={}
        # cmip6_data_alltime[experiment_id][table_id] = {}
        # cmip6_data_regridded_alltime[experiment_id][table_id] = {}
        cmip6_data_regridded_alltime_ens[experiment_id][table_id] = {}
        
        with open(f'/home/563/qg8515/scratch/data/sim/cmip6/{experiment_id}_{table_id}_{variable_id}.pkl', 'rb') as f:
            cmip6_data[experiment_id][table_id][variable_id] = pickle.load(f)
        # with open(f'/home/563/qg8515/scratch/data/sim/cmip6/{experiment_id}_{table_id}_{variable_id}_alltime.pkl', 'rb') as f:
        #     cmip6_data_alltime[experiment_id][table_id][variable_id] = pickle.load(f)
        # with open(f'/home/563/qg8515/scratch/data/sim/cmip6/{experiment_id}_{table_id}_{variable_id}_regridded_alltime.pkl', 'rb') as f:
        #     cmip6_data_regridded_alltime[experiment_id][table_id][variable_id] = pickle.load(f)
        with open(f'/home/563/qg8515/scratch/data/sim/cmip6/{experiment_id}_{table_id}_{variable_id}_regridded_alltime_ens.pkl', 'rb') as f:
            cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id] = pickle.load(f)
        
        # for ialltime in cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id].keys():
        #     print(cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id][ialltime].shape)
        
        source_id = cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id]['am']['source_id'].values.astype('object')[ith_source_id]
        print(f'#-------- {source_id}')
        
        dset = cmip6_data[experiment_id][table_id][variable_id][source_id]
        if experiment_id in ['piControl']:
            dset = dset.sel(time=slice(dset.time[-150 * 12], dset.time[-1]))
            dset = dset.assign_coords(time=pd.date_range(start='1850-01-01', periods=150 * 12, freq='1ME'))
        elif experiment_id in ['abrupt-4xCO2']:
            dset = dset.sel(time=slice(dset.time[0], dset.time[150 * 12-1]))
            dset = dset.assign_coords(time=pd.date_range(start='1850-01-01', periods=150 * 12, freq='1ME'))
        elif experiment_id in ['historical']:
            dset = dset.sel(time=slice('1850', '2014'))
            dset = dset.assign_coords(time=pd.date_range(start='1850-01-01', periods=165 * 12, freq='1ME'))
        elif experiment_id in ['amip']:
            dset = dset.sel(time=slice('1979', '2014'))
            dset = dset.assign_coords(time=pd.date_range(start='1979-01-01', periods=36 * 12, freq='1ME'))
        elif experiment_id in ['ssp585']:
            dset = dset.sel(time=slice('2015', '2099'))
            dset = dset.assign_coords(time=pd.date_range(start='2015-01-01', periods=85 * 12, freq='1ME'))
        
        if variable_id in ['tas']:
            dset[variable_id] = dset[variable_id] - zerok
        elif variable_id in ['rsut', 'rlut']:
            dset[variable_id] = dset[variable_id] * (-1)
        elif variable_id in ['pr']:
            dset[variable_id] = dset[variable_id] * seconds_per_d
        
        dset = dset.compute()
        dsetr = cdo_regrid(dset).pipe(rename_cmip6).pipe(broadcast_lonlat)[variable_id]
        
        dset2 = cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id]['mon'].sel(source_id=source_id)
        
        print((dsetr.values[np.isfinite(dsetr.values)] == dset2.values[np.isfinite(dset2.values)]).all())
        del cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id]


# https://github.com/jbusecke/xMIP/blob/main/docs/tutorial.ipynb



#---------------- check
cmip6 = intake.open_catalog('/g/data/hh5/public/apps/nci-intake-catalogue/catalogue_new.yaml').esgf.cmip6
data_catalogue = cmip6.search(experiment_id='historical', table_id='Omon', variable_id='tos', source_id='AWI-CM-1-1-MR').df
if len(data_catalogue.member_id.unique()) > 1:
    print(f'Choose member_id: {data_catalogue.member_id.unique()}')
    member_id = sorted(data_catalogue.member_id.unique())[0]
    data_catalogue=data_catalogue[data_catalogue.member_id==member_id]
    print(f'{member_id} chosen')
dset = xr.open_mfdataset(sorted(data_catalogue.path.values[0:2]), use_cftime=True, parallel=True, data_vars='minimal')
print(dset)
cdo_regrid(dset.sel(time=slice(dset.time[0], dset.time[1])))

'''
# endregion


# region get global and zonal mean

cmip6_data_regridded_alltime_ens = {}
cmip6_data_regridded_alltime_ens_gzm = {}

for experiment_id in ['piControl']:
    # experiment_id = 'piControl'
    # ['piControl', 'abrupt-4xCO2', 'historical', 'amip', 'ssp585']
    print(f'#-------------------------------- {experiment_id}')
    cmip6_data_regridded_alltime_ens[experiment_id] = {}
    cmip6_data_regridded_alltime_ens_gzm[experiment_id] = {}
    
    for table_id, variable_id in zip(['Omon'], ['tos']):
        # table_id = 'Amon'; variable_id = 'tas'
        # ['Amon'], ['tas']
        # ['Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Omon'], ['tas', 'rsut', 'rsdt', 'rlut', 'pr', 'tos']
        print(f'#---------------- {table_id} {variable_id}')
        cmip6_data_regridded_alltime_ens[experiment_id][table_id] = {}
        cmip6_data_regridded_alltime_ens_gzm[experiment_id][table_id] = {}
        cmip6_data_regridded_alltime_ens_gzm[experiment_id][table_id][variable_id] = {}
        
        with open(f'/home/563/qg8515/scratch/data/sim/cmip6/{experiment_id}_{table_id}_{variable_id}_regridded_alltime_ens.pkl', 'rb') as f:
            cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id] = pickle.load(f)
        
        for ialltime in cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id].keys():
            # ialltime = 'ann'
            print(f'#-------- {ialltime}')
            cmip6_data_regridded_alltime_ens_gzm[experiment_id][table_id][variable_id][ialltime] = {}
            
            cmip6_data_regridded_alltime_ens_gzm[experiment_id][table_id][variable_id][ialltime]['zm'] = cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id][ialltime].mean(dim='x', skipna=True).compute()
            
            cmip6_data_regridded_alltime_ens_gzm[experiment_id][table_id][variable_id][ialltime]['gm'] = cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id][ialltime].weighted(np.cos(np.deg2rad(cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id][ialltime].lat))).mean(dim=['x', 'y'], skipna=True).compute().astype(np.float32)
        
        ofile=f'/home/563/qg8515/scratch/data/sim/cmip6/{experiment_id}_{table_id}_{variable_id}_regridded_alltime_ens_gzm.pkl'
        if os.path.exists(ofile): os.remove(ofile)
        with open(ofile, 'wb') as f:
            pickle.dump(cmip6_data_regridded_alltime_ens_gzm[experiment_id][table_id][variable_id], f)
        
        del cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id]
        del cmip6_data_regridded_alltime_ens_gzm[experiment_id][table_id][variable_id]




'''

#-------------------------------- check
cmip6_data_regridded_alltime_ens = {}
cmip6_data_regridded_alltime_ens_gzm = {}

ialltime = 'mon'
ith_source_id = -1
itime = -1

for experiment_id in ['piControl', 'historical', 'ssp585']:
    # experiment_id = 'historical'
    # ['piControl', 'abrupt-4xCO2', 'historical', 'amip', 'ssp585']
    print(f'#-------------------------------- {experiment_id}')
    cmip6_data_regridded_alltime_ens[experiment_id] = {}
    cmip6_data_regridded_alltime_ens_gzm[experiment_id] = {}
    
    for table_id, variable_id in zip(['Amon'], ['tas']):
        # table_id = 'Amon'; variable_id = 'tas'
        # ['Amon'], ['tas']
        # ['Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Omon'], ['tas', 'rsut', 'rsdt', 'rlut', 'pr', 'tos']
        print(f'#---------------- {table_id} {variable_id}')
        cmip6_data_regridded_alltime_ens[experiment_id][table_id] = {}
        cmip6_data_regridded_alltime_ens_gzm[experiment_id][table_id] = {}
        
        with open(f'/home/563/qg8515/scratch/data/sim/cmip6/{experiment_id}_{table_id}_{variable_id}_regridded_alltime_ens.pkl', 'rb') as f:
            cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id] = pickle.load(f)
        with open(f'/home/563/qg8515/scratch/data/sim/cmip6/{experiment_id}_{table_id}_{variable_id}_regridded_alltime_ens_gzm.pkl', 'rb') as f:
            cmip6_data_regridded_alltime_ens_gzm[experiment_id][table_id][variable_id] = pickle.load(f)
        
        data11 = cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id][ialltime].isel(source_id=ith_source_id, time=itime).mean(dim='x', skipna=True)
        data21 = cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id][ialltime].isel(source_id=ith_source_id, time=itime).weighted(np.cos(np.deg2rad(cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id][ialltime].isel(source_id=ith_source_id, time=itime)['lat']))).mean(dim=['x', 'y'], skipna=True)
        data12 = cmip6_data_regridded_alltime_ens_gzm[experiment_id][table_id][variable_id][ialltime]['zm'].isel(source_id=ith_source_id, time=itime)
        data22 = cmip6_data_regridded_alltime_ens_gzm[experiment_id][table_id][variable_id][ialltime]['gm'].isel(source_id=ith_source_id, time=itime)
        print((data11.values[np.isfinite(data11.values)] == data12.values[np.isfinite(data12.values)]).all())
        print(np.max(np.abs(data21 - data22)).values < 1e-4)



'''
# endregion




