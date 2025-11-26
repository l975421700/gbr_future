

# qsub -I -q normal -P v46 -l walltime=4:00:00,ncpus=1,mem=20GB,jobfs=10GB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22+gdata/py18+gdata/gx60+gdata/xp65+gdata/qx55+gdata/rv74+gdata/al33+gdata/rr3+gdata/hr22+scratch/gx60+scratch/gb02+gdata/gb02


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


# endregion


# region get source_ids and member_ids

cmip_dir = {
    # 'cmip5': ['cmip5_al33', 'cmip5_rr3'],
    'cmip6': ['cmip6_fs38', 'cmip6_oi10'],
    }

for icmip in cmip_dir.keys():
    # icmip = 'cmip6'
    print(f'#-------------------------------- {icmip}')
    
    source_ids_member_ids = {}
    for idir in cmip_dir[icmip]:
        # idir = 'cmip6_oi10'
        # idir = 'cmip5_al33'
        print(f'#---------------- {idir}')
        
        source_ids = sorted(list(intake.cat.access_nri[idir].search(
            experiment_id=['piControl', 'abrupt-4xCO2'],
            table_id='Amon',
            variable_id=['tas', 'rsut', 'rsdt', 'rlut'],
            require_all_on=["source_id"]
            ).df.source_id.unique()))
        print(f'No. of source_ids: {len(source_ids)}')
        
        experiment_id_counts = intake.cat.access_nri[idir].search(
            source_id = source_ids,
            experiment_id = ['piControl', 'abrupt-4xCO2', 'amip', 'historical', 'ssp585', 'esm-piControl', 'esm-hist', 'esm-ssp585']
            ).df.groupby(['source_id', 'member_id']).experiment_id.nunique()
        
        source_ids_member_ids = source_ids_member_ids | {
            source_id: member_id
            for source_id, member_id in experiment_id_counts.groupby(level='source_id').idxmax().values
            }
    
    source_ids_member_ids = dict(sorted(source_ids_member_ids.items()))
    
    ofile = f'data/sim/cmip/{icmip}_source_ids_member_ids.json'
    if os.path.exists(ofile): os.remove(ofile)
    with open(ofile, 'w') as f:
        json.dump(source_ids_member_ids, f, indent=2)




'''
# python2_
# https://access-nri-intake-catalog.readthedocs.io/en/latest/usage/quickstart.html#
# https://research.csiro.au/access/wiki/access-model-output-archive-p73-wiki/
# print(sorted(intake.cat.access_nri.keys()))

#-------------------------------- check
cmip_dir = {
    # 'cmip5': ['cmip5_al33', 'cmip5_rr3'],
    'cmip6': ['cmip6_fs38', 'cmip6_oi10'],
    }

for icmip in cmip_dir.keys():
    # icmip = 'cmip6'
    print(f'#-------------------------------- {icmip}')
    
    with open(f'data/sim/cmip/{icmip}_source_ids_member_ids.json', 'r') as f:
        source_ids_member_ids = json.load(f)
    
    for source_id, member_id in source_ids_member_ids.items():
        # source_id, member_id = next(iter(source_ids_member_ids.items()))
        Nexp = np.sum([intake.cat.access_nri[idir].search(
            source_id=source_id,
            member_id=member_id,
            experiment_id=['piControl', 'amip', 'abrupt-4xCO2', 'historical', 'ssp585', 'esm-piControl', 'esm-hist', 'esm-ssp585']
            ).df.experiment_id.nunique() for idir in cmip_dir[icmip]])
        print(f'{source_id}  {member_id} {Nexp}')


'''
# endregion


# region get original data
# Memory Used: 23.19GB; Walltime Used: 00:12:41

# option
cmip_dir = {
    # 'cmip5': ['cmip5_al33', 'cmip5_rr3'],
    'cmip6': ['cmip6_fs38', 'cmip6_oi10'],}
experiment_ids  = ['piControl', 'abrupt-4xCO2']
# 'piControl', 'abrupt-4xCO2', 'historical', 'esm-hist', 'esm-piControl', 'amip', 'ssp585', 'esm-ssp585'
table_ids       = ['Amon']
variable_ids    = ['tas', 'rsut', 'rsdt', 'rlut']

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
#-------------------------------- check
isource_id = 5
cmip_dir = {
    # 'cmip5': ['cmip5_al33', 'cmip5_rr3'],
    'cmip6': ['cmip6_fs38', 'cmip6_oi10'],}
experiment_ids  = ['piControl', 'abrupt-4xCO2']
# 'piControl', 'abrupt-4xCO2', 'historical', 'esm-hist', 'esm-piControl', 'amip', 'ssp585', 'esm-ssp585'
table_ids       = ['Amon']
variable_ids    = ['tas', 'rsut', 'rsdt', 'rlut']

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
                
                ofile = f'data/sim/cmip/{icmip}/{experiment_id}/{table_id}_{variable_id}.pkl'
                with open(ofile, 'rb') as f:
                    ds = pickle.load(f)
                print(len(ds))
                
                # for source_id in ds.keys():
                #     # source_id = list(ds.keys())[0]
                #     print(f'#-- {source_id}')
                #     print(ds[source_id][variable_id].shape)
                
                source_id = list(ds.keys())[isource_id]
                catalogue = pd.concat([intake.cat.access_nri[idir].search(
                        experiment_id=experiment_id,
                        table_id=table_id,
                        variable_id=variable_id,
                        source_id=source_id,
                        member_id=source_ids_member_ids[source_id]
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
                    print(f'versions: {catalogue.version.unique()}')
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
                    exp_id = sorted(catalogue.experiment_id.unique())[-1]
                    catalogue=catalogue[catalogue.experiment_id==exp_id]
                    print(f'{exp_id} chosen')
                
                try:
                    dset = xr.open_mfdataset(sorted(catalogue.path.values), use_cftime=True, parallel=True)
                    if len(dset.time) < 120:
                        print('Warning simulation shorter than 10 years')
                        continue
                    
                    data1 = ds[source_id][variable_id].values
                    data2 = dset[variable_id].values
                    print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())
                    
                except FileNotFoundError:
                    print('Warning no file found')
                except ValueError:
                    print('Warning file opening error')




'''
# endregion


# region get alltime, regridded_alltime, and regridded_alltime_ens data
# Memory Used: 158.68GB, Walltime Used: 00:30:58, JobFS used: 893.75MB

# option
cmips = ['cmip6']
experiment_ids  = ['piControl', 'abrupt-4xCO2']
table_ids       = ['Amon']
variable_ids    = ['tas', 'rsut', 'rsdt', 'rlut']

# setting
min_length = {
    'piControl': 150,
    'abrupt-4xCO2': 150,
    'historical': 165,
    'esm-hist': 165,
    'esm-piControl': 150,
    'amip': 36,
    'ssp585': 85,
    'esm-ssp585': 85,
    }
years_yeare = {
    'historical':   ['1850', '2014'],
    'esm-hist':     ['1850', '2014'],
    'amip':         ['1979', '2014'],
    'ssp585':       ['2015', '2099'],
    'esm-ssp585':   ['2015', '2099'],
    }
start_dates = {
    'piControl':    '1850-01-01',
    'abrupt-4xCO2': '1850-01-01',
    'historical':   '1850-01-01',
    'esm-hist':     '1850-01-01',
    'esm-piControl':'1850-01-01',
    'amip':         '1979-01-01',
    'ssp585':       '2015-01-01',
    'esm-ssp585':   '2015-01-01',
    }

for icmip in cmips:
    # icmip = 'cmip6'
    print(f'#-------------------------------- {icmip}')
    for experiment_id in experiment_ids:
        # experiment_id = 'piControl'
        print(f'#---------------- {experiment_id}')
        for table_id in table_ids:
            # table_id = 'Amon'
            print(f'#-------- {table_id}')
            for variable_id in variable_ids:
                # variable_id = 'tas'
                print(f'#---- {variable_id}')
                
                ofile = f'data/sim/cmip/{icmip}/{experiment_id}/{table_id}_{variable_id}.pkl'
                with open(ofile, 'rb') as f:
                    ds = pickle.load(f)
                
                ds_alltime = {}
                ds_regridded_alltime = {}
                ds_regridded_alltime_ens = {}
                
                for source_id in ds.keys():
                    # source_id = 'MPI-ESM-1-2-HAM'
                    print(f'#-- {source_id}')
                    
                    dset = ds[source_id].copy()
                    
                    if (len(dset.time)/12) < min_length[experiment_id]:
                        print(f'Warning simulation length {len(dset.time)/12} less than required {min_length[experiment_id]} years')
                        continue
                    
                    if dset.time[-1].dt.month != 12:
                        print('Warning last month is not December')
                        continue
                    
                    if dset[variable_id].units != cmip6_units[variable_id]:
                        print(f'Warning inconsistent units: {dset[variable_id].units} rather than {cmip6_units[variable_id]}')
                        continue
                    
                    if experiment_id in ['piControl']:
                        yeare = dset.time[-1].dt.year.values
                        years = yeare - min_length[experiment_id] + 1
                        years_yeare[experiment_id] = [f'{years:04d}', f'{yeare:04d}']
                    elif experiment_id in ['abrupt-4xCO2']:
                        years = dset.time[0].dt.year.values
                        yeare = years + min_length[experiment_id] - 1
                        years_yeare[experiment_id] = [f'{years:04d}', f'{yeare:04d}']
                    
                    dset = dset.sel(time=slice(years_yeare[experiment_id][0],
                                               years_yeare[experiment_id][1]))
                    
                    if len(dset.time)/12 != min_length[experiment_id]:
                        print(f'Warning differred length {len(dset.time)/12} vs. {min_length[experiment_id]}')
                        continue
                    
                    dset = dset.assign_coords(time=pd.date_range(
                        start=start_dates[experiment_id],
                        periods=min_length[experiment_id] * 12,
                        freq='1ME'))
                    
                    if variable_id in ['tas']:
                        dset[variable_id] -= zerok
                    elif variable_id in ['rsut', 'rlut', 'hfls', 'hfss', 'rlus', 'rlutcs', 'rsus', 'rsuscs', 'rsutcs']:
                        dset[variable_id] *= (-1)
                    elif variable_id in ['pr', 'evspsbl']:
                        dset[variable_id] *= seconds_per_d
                    elif variable_id in ['psl']:
                        dset[variable_id] /= 100
                    
                    dset = dset.compute()
                    dsetr = cdo_regrid(dset)
                    
                    ds_alltime[source_id] = mon_sea_ann(
                        var_monthly=dset.pipe(rename_cmip6).pipe(broadcast_lonlat)[variable_id],
                        lcopy = True, mm=True, sm=True, am=True)
                    ds_regridded_alltime[source_id] = mon_sea_ann(
                        var_monthly=dsetr.pipe(rename_cmip6).pipe(broadcast_lonlat)[variable_id],
                        lcopy = True, mm=True, sm=True, am=True)
                
                source_ids = list(ds_regridded_alltime.keys())
                source_da = xr.DataArray(source_ids, dims='source_id', coords={'source_id': source_ids})
                
                for ialltime in ds_regridded_alltime[source_ids[0]].keys():
                    # ialltime = 'mon'
                    print(f'#-------- {ialltime}')
                    
                    ds_regridded_alltime_ens[ialltime] = xr.concat([ds_regridded_alltime[source_id][ialltime] for source_id in source_ids], dim=source_da, coords='minimal', compat='override')
                
                ofile1 = f'data/sim/cmip/{icmip}/{experiment_id}/{table_id}_{variable_id}_alltime.pkl'
                ofile2 = f'data/sim/cmip/{icmip}/{experiment_id}/{table_id}_{variable_id}_regridded_alltime.pkl'
                ofile3 = f'data/sim/cmip/{icmip}/{experiment_id}/{table_id}_{variable_id}_regridded_alltime_ens.pkl'
                if os.path.exists(ofile1): os.remove(ofile1)
                if os.path.exists(ofile2): os.remove(ofile2)
                if os.path.exists(ofile3): os.remove(ofile3)
                with open(ofile1, 'wb') as f:
                    pickle.dump(ds_alltime, f)
                with open(ofile2, 'wb') as f:
                    pickle.dump(ds_regridded_alltime, f)
                with open(ofile3, 'wb') as f:
                    pickle.dump(ds_regridded_alltime_ens, f)




'''
# https://github.com/jbusecke/xMIP/blob/main/docs/tutorial.ipynb
#-------------------------------- check
isource_id = -5
cmips = ['cmip6']
experiment_ids  = ['piControl', 'abrupt-4xCO2']
table_ids       = ['Amon']
variable_ids    = ['tas', 'rsut', 'rsdt', 'rlut']

min_length = {
    'piControl': 150,
    'abrupt-4xCO2': 150,
    'historical': 165,
    'esm-hist': 165,
    'esm-piControl': 150,
    'amip': 36,
    'ssp585': 85,
    'esm-ssp585': 85,
    }
years_yeare = {
    'historical':   ['1850', '2014'],
    'esm-hist':     ['1850', '2014'],
    'amip':         ['1979', '2014'],
    'ssp585':       ['2015', '2099'],
    'esm-ssp585':   ['2015', '2099'],
    }
start_dates = {
    'piControl':    '1850-01-01',
    'abrupt-4xCO2': '1850-01-01',
    'historical':   '1850-01-01',
    'esm-hist':     '1850-01-01',
    'esm-piControl':'1850-01-01',
    'amip':         '1979-01-01',
    'ssp585':       '2015-01-01',
    'esm-ssp585':   '2015-01-01',
    }

for icmip in cmips:
    # icmip = 'cmip6'
    print(f'#-------------------------------- {icmip}')
    for experiment_id in experiment_ids:
        # experiment_id = 'piControl'
        print(f'#---------------- {experiment_id}')
        for table_id in table_ids:
            # table_id = 'Amon'
            print(f'#-------- {table_id}')
            for variable_id in variable_ids:
                # variable_id = 'tas'
                print(f'#---- {variable_id}')
                
                ofile = f'data/sim/cmip/{icmip}/{experiment_id}/{table_id}_{variable_id}.pkl'
                ofile1 = f'data/sim/cmip/{icmip}/{experiment_id}/{table_id}_{variable_id}_alltime.pkl'
                ofile2 = f'data/sim/cmip/{icmip}/{experiment_id}/{table_id}_{variable_id}_regridded_alltime.pkl'
                ofile3 = f'data/sim/cmip/{icmip}/{experiment_id}/{table_id}_{variable_id}_regridded_alltime_ens.pkl'
                with open(ofile, 'rb') as f:
                    ds = pickle.load(f)
                with open(ofile1, 'rb') as f:
                    ds_alltime = pickle.load(f)
                with open(ofile2, 'rb') as f:
                    ds_regridded_alltime = pickle.load(f)
                with open(ofile3, 'rb') as f:
                    ds_regridded_alltime_ens = pickle.load(f)
                
                source_id = list(ds_alltime.keys())[isource_id]
                dset = ds[source_id].copy()
                
                if experiment_id in ['piControl']:
                    yeare = dset.time[-1].dt.year.values
                    years = yeare - min_length[experiment_id] + 1
                    years_yeare[experiment_id]=[f'{years:04d}', f'{yeare:04d}']
                elif experiment_id in ['abrupt-4xCO2']:
                    years = dset.time[0].dt.year.values
                    yeare = years + min_length[experiment_id] - 1
                    years_yeare[experiment_id]=[f'{years:04d}', f'{yeare:04d}']
                
                dset = dset.sel(time=slice(years_yeare[experiment_id][0],
                                           years_yeare[experiment_id][1]))
                dset = dset.assign_coords(time=pd.date_range(
                    start=start_dates[experiment_id],
                    periods=min_length[experiment_id] * 12,
                    freq='1ME'))
                
                if variable_id in ['tas']:
                    dset[variable_id] -= zerok
                elif variable_id in ['rsut', 'rlut', 'hfls', 'hfss', 'rlus', 'rlutcs', 'rsus', 'rsuscs', 'rsutcs']:
                    dset[variable_id] *= (-1)
                elif variable_id in ['pr', 'evspsbl']:
                    dset[variable_id] *= seconds_per_d
                elif variable_id in ['psl']:
                    dset[variable_id] /= 100
                
                dset = dset.compute()
                dsetr = cdo_regrid(dset).pipe(rename_cmip6).pipe(broadcast_lonlat)[variable_id]
                dset = dset.pipe(rename_cmip6).pipe(broadcast_lonlat)[variable_id]
                
                data10 = dset.values
                data20 = dsetr.values
                
                data11 = ds_alltime[source_id]['mon'].values
                data21 = ds_regridded_alltime[source_id]['mon'].values
                data22 = ds_regridded_alltime_ens['mon'].sel(source_id=source_id).values
                
                print((data10[np.isfinite(data10)] == data11[np.isfinite(data11)]).all())
                print((data20[np.isfinite(data20)] == data21[np.isfinite(data21)]).all())
                print((data20[np.isfinite(data20)] == data22[np.isfinite(data22)]).all())
                
                del ds, ds_alltime, ds_regridded_alltime, ds_regridded_alltime_ens


'''
# endregion


# region get global and zonal mean
# Memory Used: 76.61GB; Walltime Used: 00:02:30

# option
cmips = ['cmip6']
experiment_ids  = ['piControl', 'abrupt-4xCO2']
table_ids       = ['Amon']
variable_ids    = ['tas', 'rsut', 'rsdt', 'rlut']

for icmip in cmips:
    # icmip = 'cmip6'
    print(f'#-------------------------------- {icmip}')
    for experiment_id in experiment_ids:
        # experiment_id = 'piControl'
        print(f'#---------------- {experiment_id}')
        for table_id in table_ids:
            # table_id = 'Amon'
            print(f'#-------- {table_id}')
            for variable_id in variable_ids:
                # variable_id = 'tas'
                print(f'#---- {variable_id}')
                
                ofile3 = f'data/sim/cmip/{icmip}/{experiment_id}/{table_id}_{variable_id}_regridded_alltime_ens.pkl'
                with open(ofile3, 'rb') as f:
                    ds_regridded_alltime_ens = pickle.load(f)
                
                ds_regridded_alltime_ens_gzm = {}
                
                for ialltime in ds_regridded_alltime_ens.keys():
                    # ialltime = 'mon'
                    print(f'#-- {ialltime}')
                    ds_regridded_alltime_ens_gzm[ialltime] = {}
                    
                    ds_regridded_alltime_ens_gzm[ialltime]['zm'] = ds_regridded_alltime_ens[ialltime].mean(dim='x', skipna=True).compute()
                    ds_regridded_alltime_ens_gzm[ialltime]['gm'] = ds_regridded_alltime_ens[ialltime].weighted(np.cos(np.deg2rad(ds_regridded_alltime_ens[ialltime].lat))).mean(dim=['x', 'y'], skipna=True).compute()
                
                ofile4 = f'data/sim/cmip/{icmip}/{experiment_id}/{table_id}_{variable_id}_regridded_alltime_ens_gzm.pkl'
                if os.path.exists(ofile4): os.remove(ofile4)
                with open(ofile4, 'wb') as f:
                    pickle.dump(ds_regridded_alltime_ens_gzm, f)
        
        del ds_regridded_alltime_ens, ds_regridded_alltime_ens_gzm




'''
#-------------------------------- check
ialltime = 'mon'
isource_id = -10
itime = -1

cmips = ['cmip6']
experiment_ids  = ['piControl', 'abrupt-4xCO2']
table_ids       = ['Amon']
variable_ids    = ['tas', 'rsut', 'rsdt', 'rlut']

for icmip in cmips:
    # icmip = 'cmip6'
    print(f'#-------------------------------- {icmip}')
    for experiment_id in experiment_ids:
        # experiment_id = 'piControl'
        print(f'#---------------- {experiment_id}')
        for table_id in table_ids:
            # table_id = 'Amon'
            print(f'#-------- {table_id}')
            for variable_id in variable_ids:
                # variable_id = 'tas'
                print(f'#---- {variable_id}')
                
                ofile3 = f'data/sim/cmip/{icmip}/{experiment_id}/{table_id}_{variable_id}_regridded_alltime_ens.pkl'
                ofile4 = f'data/sim/cmip/{icmip}/{experiment_id}/{table_id}_{variable_id}_regridded_alltime_ens_gzm.pkl'
                with open(ofile3, 'rb') as f:
                    ds_regridded_alltime_ens = pickle.load(f)
                with open(ofile4, 'rb') as f:
                    ds_regridded_alltime_ens_gzm = pickle.load(f)
                
                data11 = ds_regridded_alltime_ens[ialltime].isel(source_id=isource_id, time=itime).mean(dim='x', skipna=True)
                data21 = ds_regridded_alltime_ens[ialltime].isel(source_id=isource_id, time=itime).weighted(np.cos(np.deg2rad(ds_regridded_alltime_ens[ialltime].isel(source_id=isource_id, time=itime)['lat']))).mean(dim=['x', 'y'], skipna=True)
                data12 = ds_regridded_alltime_ens_gzm[ialltime]['zm'].isel(source_id=isource_id, time=itime)
                data22 = ds_regridded_alltime_ens_gzm[ialltime]['gm'].isel(source_id=isource_id, time=itime)
                print((data11.values[np.isfinite(data11.values)] == data12.values[np.isfinite(data12.values)]).all())
                print((data21.values[np.isfinite(data21.values)] == data22.values[np.isfinite(data22.values)]).all())
                print(np.max(np.abs(data21 - data22)).values < 1e-4)




#-------------------------------- check 2
cmips = ['cmip6']
experiment_ids  = ['piControl', 'abrupt-4xCO2']
table_ids       = ['Amon']
variable_ids    = ['tas', 'rsut', 'rsdt', 'rlut']

for icmip in cmips:
    # icmip = 'cmip6'
    print(f'#-------------------------------- {icmip}')
    for experiment_id in experiment_ids:
        # experiment_id = 'piControl'
        print(f'#---------------- {experiment_id}')
        for table_id in table_ids:
            # table_id = 'Amon'
            print(f'#-------- {table_id}')
            for variable_id in variable_ids:
                # variable_id = 'tas'
                print(f'#---- {variable_id}')
                
                ofile4 = f'data/sim/cmip/{icmip}/{experiment_id}/{table_id}_{variable_id}_regridded_alltime_ens_gzm.pkl'
                with open(ofile4, 'rb') as f:
                    ds_regridded_alltime_ens_gzm = pickle.load(f)
                
                # print(ds_regridded_alltime_ens_gzm['am']['gm'].values.flatten())
                print(np.min(ds_regridded_alltime_ens_gzm['am']['gm'].values))
                print(np.max(ds_regridded_alltime_ens_gzm['am']['gm'].values))




'''
# endregion




# region check available datasets

cmip_dir = {
    # 'cmip5': ['cmip5_al33', 'cmip5_rr3'],
    'cmip6': ['cmip6_fs38', 'cmip6_oi10'],}
icmip = 'cmip6'
with open(f'data/sim/cmip/{icmip}_source_ids_member_ids.json', 'r') as f:
    source_ids_member_ids = json.load(f)

source_ids = list(source_ids_member_ids.keys())

catalogue = pd.concat([intake.cat.access_nri[idir].search(
    # experiment_id=experiment_id,
    # table_id=table_id,
    # variable_id=variable_id,
    # source_id=source_ids,
    # member_id=member_id,
    project_id='CFMIP'
    ).df for idir in cmip_dir[icmip]], ignore_index=True)

print('\n'.join(sorted(catalogue.experiment_id.unique())))
print('\n'.join(sorted(catalogue.table_id.unique())))
print('\n'.join(sorted(catalogue.variable_id.unique())))
print(catalogue.columns)

for icolumn in catalogue.columns:
    print(f'#-------------------------------- {icolumn}')
    print(catalogue[icolumn][:5])
    if len(catalogue[icolumn].unique()) < 100:
        print('\n'.join(sorted(catalogue[icolumn].unique())))



# endregion
