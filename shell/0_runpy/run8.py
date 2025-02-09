

# region import packages

# data analysis
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
import pandas as pd
import intake
from cdo import Cdo
cdo=Cdo()
import xarray as xr
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
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


# region get the mon_sea_ann data

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
    
    for table_id, variable_id in zip(['Amon', 'Amon', 'Omon'], ['rlut', 'pr', 'tos']):
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
            # source_id ='ACCESS-CM2'
            print(f'#-------- {source_id}')
            # print(np.min(cmip6_data[experiment_id][table_id][variable_id][source_id][variable_id].lon.values))
            # print(cmip6_data[experiment_id][table_id][variable_id][source_id][variable_id].units)
            # print(cmip6_data[experiment_id][table_id][variable_id][source_id][variable_id].time[-1].values)
            # print(len(cmip6_data[experiment_id][table_id][variable_id][source_id][variable_id].time))
            
            # ensure enough simulation length
            if (experiment_id in ['piControl', 'abrupt-4xCO2']) & (len(cmip6_data[experiment_id][table_id][variable_id][source_id][variable_id].time) < 150 * 12):
                print('Simulation length less than 150 yrs: ignored')
                continue
            elif (experiment_id in ['historical']) & (len(cmip6_data[experiment_id][table_id][variable_id][source_id][variable_id].time) < 165 * 12):
                print('Simulation length less than 165 yrs: ignored')
                continue
            elif (experiment_id in ['amip']) & (len(cmip6_data[experiment_id][table_id][variable_id][source_id][variable_id].time) < 36 * 12):
                print('Simulation length less than 36 yrs: ignored')
                continue
            elif (experiment_id in ['ssp585']) & (len(cmip6_data[experiment_id][table_id][variable_id][source_id][variable_id].time) < 85 * 12):
                print('Simulation length less than 85 yrs: ignored')
                continue
            
            # ensure the last month is Dec
            if cmip6_data[experiment_id][table_id][variable_id][source_id][variable_id].time[-1].dt.month != 12:
                print('Warning: last month is not December')
            
            dset = cmip6_data[experiment_id][table_id][variable_id][source_id]
            
            # ensure correct periods are selected
            if experiment_id in ['piControl', 'abrupt-4xCO2']:
                dset = dset.sel(time=slice(dset.time[-150 * 12], dset.time[-1]))
                if len(dset.time)/12 != 150:
                    print(f'Warning differred time length: {len(dset.time)/12}')
            elif experiment_id in ['historical']:
                dset = dset.sel(time=slice('1850-01-01', '2015-01-01'))
                if len(dset.time)/12 != 165:
                    print(f'Warning differred time length: {len(dset.time)/12}')
            elif experiment_id in ['amip']:
                dset = dset.sel(time=slice('1979-01-01', '2015-01-01'))
                if len(dset.time)/12 != 36:
                    print(f'Warning differred time length: {len(dset.time)/12}')
            elif experiment_id in ['ssp585']:
                if dset.time[0].dt.year != 2015:
                    print('Shifting the time')
                    dset = dset.sel(time=slice(dset.time[0], dset.time[85*12-1])).assign_coords(time=pd.date_range(start='2015-01-01', end='2100-01-01', freq='1ME'))
                else:
                    dset = dset.sel(time=slice('2015-01-01', '2100-01-01'))
                if len(dset.time)/12 != 85:
                    print(f'Warning differred time length: {len(dset.time)/12}')
            
            # ensure correct units
            if dset[variable_id].units != cmip6_units[variable_id]:
                print(f'Warning inconsistent units: {dset[variable_id].units} rather than {cmip6_units[variable_id]}')
            
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
            # mon_sea_ann(var_monthly=dset.sel(time=slice(dset.time[0], dset.time[11])).pipe(rename_cmip6).pipe(broadcast_lonlat)[variable_id], lcopy = False, mm=True, sm=True, am=True)
            
            print('calculate regridded mon_sea_ann')
            dsetr = cdo_regrid(dset)
            cmip6_data_regridded_alltime[experiment_id][table_id][variable_id][source_id] = mon_sea_ann(var_monthly=dsetr.pipe(rename_cmip6).pipe(broadcast_lonlat)[variable_id], lcopy = False, mm=True, sm=True, am=True)
            # mon_sea_ann(var_monthly=dsetr.sel(time=slice(dsetr.time[0], dsetr.time[11])).pipe(rename_cmip6).pipe(broadcast_lonlat)[variable_id], lcopy = False, mm=True, sm=True, am=True)
            
            del dset, dsetr
        
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




# https://github.com/jbusecke/xMIP/blob/main/docs/tutorial.ipynb
# https://github.com/coecms/xmip_nci/blob/main/intake_complex.ipynb
# https://github.com/coecms/nci-intake-catalogue/blob/main/docs/intake_cmip6_demo.ipynb
# https://cf-xarray.readthedocs.io/en/latest/examples/introduction.html
# https://forum.access-hive.org.au/t/analysing-cmip6-models-in-gadi-using-python/177/4
# https://github.com/DamienIrving/ocean-analysis/blob/master/cmip6_notes.md
# https://github.com/DamienIrving/ocean-analysis/blob/master/cmip6_notes.md
# https://forum.access-hive.org.au/t/intake-esm-and-3hr-data/2259/5

# https://projectpythia.org/cmip6-cookbook/notebooks/example-workflows/ecs-cmip6.html

Shifting the time: CIESM Amon rsdt, GISS-E2-1-H Omon tos,

'''
# endregion

