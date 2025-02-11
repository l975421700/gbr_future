

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


# region get mon_sea_ann data

cmip6_data = {}
cmip6_data_alltime = {}
cmip6_data_regridded_alltime = {}

for experiment_id in ['historical']:
    # experiment_id = 'piControl'
    # ['piControl', 'abrupt-4xCO2', 'historical', 'amip', 'ssp585']
    print(f'#-------------------------------- {experiment_id}')
    cmip6_data[experiment_id] = {}
    cmip6_data_alltime[experiment_id] = {}
    cmip6_data_regridded_alltime[experiment_id] = {}
    
    for table_id, variable_id in zip(['Omon'], ['tos']):
        # table_id = 'Amon'; variable_id = 'tas'
        # ['Amon'], ['tas']
        # ['Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Omon'], ['tas', 'rsut', 'rsdt', 'rlut', 'pr', 'tos']
        print(f'#---------------- {table_id} {variable_id}')
        cmip6_data[experiment_id][table_id]={}
        cmip6_data_alltime[experiment_id][table_id] = {}
        cmip6_data_alltime[experiment_id][table_id][variable_id] = {}
        cmip6_data_regridded_alltime[experiment_id][table_id] = {}
        cmip6_data_regridded_alltime[experiment_id][table_id][variable_id] = {}
        
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
            print('calculate mon_sea_ann')
            cmip6_data_alltime[experiment_id][table_id][variable_id][source_id] = mon_sea_ann(var_monthly=dset.pipe(rename_cmip6).pipe(broadcast_lonlat)[variable_id], lcopy = False, mm=True, sm=True, am=True)
            
            print('calculate regridded mon_sea_ann')
            dsetr = cdo_regrid(dset)
            cmip6_data_regridded_alltime[experiment_id][table_id][variable_id][source_id] = mon_sea_ann(var_monthly=dsetr.pipe(rename_cmip6).pipe(broadcast_lonlat)[variable_id], lcopy = False, mm=True, sm=True, am=True)
            
            del dset, dsetr
            gc.collect()
            print(process.memory_info().rss / 2**30)
        
        ofile1=f'/home/563/qg8515/scratch/data/sim/cmip6/{experiment_id}_{table_id}_{variable_id}_alltime.pkl'
        if os.path.exists(ofile1): os.remove(ofile1)
        with open(ofile1, 'wb') as f:
            pickle.dump(cmip6_data_alltime[experiment_id][table_id][variable_id], f)
        
        ofile2=f'/home/563/qg8515/scratch/data/sim/cmip6/{experiment_id}_{table_id}_{variable_id}_regridded_alltime.pkl'
        if os.path.exists(ofile2): os.remove(ofile2)
        with open(ofile2, 'wb') as f:
            pickle.dump(cmip6_data_regridded_alltime[experiment_id][table_id][variable_id], f)
        
        del cmip6_data[experiment_id][table_id][variable_id]
        del cmip6_data_alltime[experiment_id][table_id][variable_id]
        del cmip6_data_regridded_alltime[experiment_id][table_id][variable_id]





'''
#-------------------------------- check
cmip6_data = {}
cmip6_data_alltime = {}
cmip6_data_regridded_alltime = {}

ith_source_id=-1

for experiment_id in ['historical']:
    # experiment_id = 'piControl'
    # ['piControl', 'abrupt-4xCO2', 'historical', 'amip', 'ssp585']
    print(f'#-------------------------------- {experiment_id}')
    cmip6_data[experiment_id] = {}
    cmip6_data_alltime[experiment_id] = {}
    cmip6_data_regridded_alltime[experiment_id] = {}
    
    for table_id, variable_id in zip(['Amon'], ['tas']):
        # table_id = 'Amon'; variable_id = 'tas'
        # ['Amon'], ['tas']
        # ['Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Omon'], ['tas', 'rsut', 'rsdt', 'rlut', 'pr', 'tos']
        print(f'#---------------- {table_id} {variable_id}')
        cmip6_data[experiment_id][table_id]={}
        cmip6_data_alltime[experiment_id][table_id] = {}
        cmip6_data_regridded_alltime[experiment_id][table_id] = {}
        
        with open(f'/home/563/qg8515/scratch/data/sim/cmip6/{experiment_id}_{table_id}_{variable_id}.pkl', 'rb') as f:
            cmip6_data[experiment_id][table_id][variable_id] = pickle.load(f)
        with open(f'/home/563/qg8515/scratch/data/sim/cmip6/{experiment_id}_{table_id}_{variable_id}_alltime.pkl', 'rb') as f:
            cmip6_data_alltime[experiment_id][table_id][variable_id] = pickle.load(f)
        with open(f'/home/563/qg8515/scratch/data/sim/cmip6/{experiment_id}_{table_id}_{variable_id}_regridded_alltime.pkl', 'rb') as f:
            cmip6_data_regridded_alltime[experiment_id][table_id][variable_id] = pickle.load(f)
        
        print(len(cmip6_data[experiment_id][table_id][variable_id].keys()))
        print(len(cmip6_data_alltime[experiment_id][table_id][variable_id].keys()))
        print(len(cmip6_data_regridded_alltime[experiment_id][table_id][variable_id].keys()))
        
        for source_id in cmip6_data_regridded_alltime[experiment_id][table_id][variable_id].keys():
            print(f'#-------- {source_id}')
            if (experiment_id in ['piControl', 'abrupt-4xCO2']) & ((len(cmip6_data_alltime[experiment_id][table_id][variable_id][source_id]['mon'].time) != 150 * 12) | (len(cmip6_data_regridded_alltime[experiment_id][table_id][variable_id][source_id]['mon'].time) != 150 * 12)):
                print('Warning: wrong simulation length')
            elif (experiment_id in ['historical']) & ((len(cmip6_data_alltime[experiment_id][table_id][variable_id][source_id]['mon'].time) != 165 * 12) | (len(cmip6_data_regridded_alltime[experiment_id][table_id][variable_id][source_id]['mon'].time) != 165 * 12)):
                print('Warning: wrong simulation length')
            elif (experiment_id in ['amip']) & ((len(cmip6_data_alltime[experiment_id][table_id][variable_id][source_id]['mon'].time) != 36 * 12) | (len(cmip6_data_regridded_alltime[experiment_id][table_id][variable_id][source_id]['mon'].time) != 36 * 12)):
                print('Warning: wrong simulation length')
            elif (experiment_id in ['ssp585']) & ((len(cmip6_data_alltime[experiment_id][table_id][variable_id][source_id]['mon'].time) != 85 * 12) | (len(cmip6_data_regridded_alltime[experiment_id][table_id][variable_id][source_id]['mon'].time) != 85 * 12)):
                print('Warning: wrong simulation length')
        
        source_id = list(cmip6_data_alltime[experiment_id][table_id][variable_id].keys())[ith_source_id]
        dset = cmip6_data[experiment_id][table_id][variable_id][source_id][variable_id]
        dset_alltime = cmip6_data_alltime[experiment_id][table_id][variable_id][source_id]['mon']
        dset_regridded_alltime = cmip6_data_regridded_alltime[experiment_id][table_id][variable_id][source_id]['mon']
        print(f'{np.max(np.abs(cdo_regrid(dset_alltime)[variable_id].values - dset_regridded_alltime.values))}')


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


                # if dset.time[0].dt.year != 2015:
                #     print('Shifting the time')
                #     dset = dset.sel(time=slice(dset.time[0], dset.time[85*12-1])).assign_coords(time=pd.date_range(start='2015-01-01', end='2100-01-01', freq='1ME'))
                # else:
                #     dset = dset.sel(time=slice('2015-01-01', '2100-01-01'))

# https://github.com/jbusecke/xMIP/blob/main/docs/tutorial.ipynb

'''
# endregion

