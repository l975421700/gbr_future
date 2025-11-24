

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
import json
import time

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


# region get original data

# option
cmip_dir = {
    # 'cmip5': ['cmip5_al33', 'cmip5_rr3'],
    'cmip6': ['cmip6_fs38', 'cmip6_oi10'],}
experiment_ids  = ['abrupt-4xCO2']
# 'piControl', 'abrupt-4xCO2', 'historical', 'esm-hist', 'esm-piControl', 'amip', 'ssp585', 'esm-ssp585'
table_ids       = ['Amon']
variable_ids    = ['rlut']

for icmip in cmip_dir.keys():
    # icmip = 'cmip6'
    print(f'#-------------------------------- {icmip}')
    
    with open(f'data/sim/cmip/{icmip}_source_ids_member_ids.json', 'r') as f:
        source_ids_member_ids = json.load(f)
    
    for experiment_id in experiment_ids:
        # experiment_id = 'piControl'
        print(f'#---------------- {experiment_id}')
        
        for table_id in table_ids:
            # table_id = 'Amon'
            print(f'#-------- {table_id}')
            
            for variable_id in variable_ids:
                # variable_id = 'tas'
                print(f'#---- {variable_id}')
                
                start_time = time.perf_counter()
                ds = {}
                for source_id, member_id in source_ids_member_ids.items():
                    # source_id, member_id = next(iter(source_ids_member_ids.items()))
                    print(f'#-- {source_id}  {member_id}')
                    
                    catalogue = pd.concat([intake.cat.access_nri[idir].search(
                        experiment_id=experiment_id,
                        table_id=table_id,
                        variable_id=variable_id,
                        source_id=source_id,
                        member_id=member_id
                        ).df for idir in cmip_dir[icmip]], ignore_index=True)
                    
                    if len(catalogue) == 0:
                        print('Change to other member_id')
                        catalogue = pd.concat(
                            [intake.cat.access_nri[idir].search(
                                experiment_id=experiment_id,
                                table_id=table_id,
                                variable_id=variable_id,
                                source_id=source_id,
                                ).df for idir in cmip_dir[icmip]],
                            ignore_index=True)
                        if len(catalogue) == 0:
                            print('Warning no data found')
                            continue
                    
                    if len(catalogue.version.unique()) > 1:
                        print(f'Versions: {list(catalogue.version.unique())}')
                        version = sorted(catalogue.version.unique())[-1]
                        catalogue=catalogue[catalogue.version==version]
                        print(f'{version} chosen')
                    
                    if len(catalogue.grid_label.unique()) > 1:
                        print(f'grid_labels: {catalogue.grid_label.unique()}')
                        grid_label = sorted(catalogue.grid_label.unique())[0]
                        catalogue=catalogue[catalogue.grid_label==grid_label]
                        print(f'{grid_label} chosen')
                    
                    if len(catalogue.member_id.unique()) > 1:
                        print(f'member_ids: {catalogue.member_id.unique()}')
                        member_id = sorted(catalogue.member_id.unique())[0]
                        catalogue=catalogue[catalogue.member_id==member_id]
                        print(f'{member_id} chosen')
                    
                    if len(catalogue.experiment_id.unique()) > 1:
                        print(f'experiment_ids: {catalogue.experiment_id.unique()}')
                        experiment_id = sorted(catalogue.experiment_id.unique())[-1]
                        catalogue=catalogue[catalogue.experiment_id==experiment_id]
                        print(f'{experiment_id} chosen')
                    
                    try:
                        dset = xr.open_mfdataset(sorted(list(catalogue.path)), use_cftime=True, parallel=True, data_vars='minimal', compat='override', coords='minimal')
                        
                        if len(dset.time) < 120:
                            print('Warning simulation shorter than 10 years')
                            continue
                        
                        ds[source_id] = dset
                    except FileNotFoundError:
                        print('Warning no file found')
                    except ValueError:
                        print('Warning file opening error')
                
                if len(ds) > 0:
                    odir = f'data/sim/cmip/{icmip}/{experiment_id}/'
                    os.makedirs(odir, exist_ok=True)
                    ofile = f'{odir}/{table_id}_{variable_id}.pkl'
                    if os.path.exists(ofile): os.remove(ofile)
                    with open(ofile, 'wb') as f:
                        pickle.dump(ds, f)
                    del ds
                
                end_time = time.perf_counter()
                print(f"Execution time: {(end_time - start_time)/60:.1f} min")




'''
# check availabel data
cmip6 = intake.open_catalog('/g/data/hh5/public/apps/nci-intake-catalogue/catalogue_new.yaml').esgf.cmip6
with open('/home/563/qg8515/data/sim/cmip6/cmip6_ids.pkl', 'rb') as f:
    cmip6_ids = pickle.load(f)

data_catalogue = cmip6.search(experiment_id=['piControl', 'esm-piControl', 'abrupt-4xCO2', 'historical', 'esm-hist', 'amip', 'ssp585', 'esm-ssp585'], source_id=list(cmip6_ids.keys()), variable_id='rluscs').df
cmip6.search(variable_id='rluscs').df


#-------------------------------- check
cmip6 = intake.open_catalog('/g/data/hh5/public/apps/nci-intake-catalogue/catalogue_new.yaml').esgf.cmip6
with open('/home/563/qg8515/data/sim/cmip6/cmip6_ids.pkl', 'rb') as f:
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
        
        with open(f'/home/563/qg8515/data/sim/cmip6/{experiment_id[0]}_{table_id}_{variable_id}.pkl', 'rb') as f:
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
with open(f'data/sim/cmip6/{experiment_id[0]}_{table_id}_{variable_id}.pkl', 'rb') as f:
    cmip6_data[experiment_id[0]][table_id][variable_id] = pickle.load(f)


cmip6_data[experiment_id[0]][table_id][variable_id][source_id]



'''
# endregion

