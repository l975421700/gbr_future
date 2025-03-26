

# region get BARRA-C2 hourly data

for var in ['clh', 'clm', 'cll', 'clt', 'pr', 'tas']:
    # var = 'cll'
    print(f'#-------------------------------- {var}')
    
    fl = sorted(glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/{var}/latest/*'))[:540]
    
    barra_c2_hourly = xr.open_mfdataset(fl, parallel=True)[var] #.sel(time=slice('1979', '2023'))
    if var in ['pr', 'evspsbl', 'evspsblpot']:
        barra_c2_hourly = barra_c2_hourly * seconds_per_d
    elif var in ['tas', 'ts']:
        barra_c2_hourly = barra_c2_hourly - zerok
    elif var in ['rlus', 'rluscs', 'rlut', 'rlutcs', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'hfls', 'hfss']:
        barra_c2_hourly = barra_c2_hourly * (-1)
    elif var in ['psl']:
        barra_c2_hourly = barra_c2_hourly / 100
    elif var in ['huss']:
        barra_c2_hourly = barra_c2_hourly * 1000
    
    ofile = f'data/sim/um/barra_c2/barra_c2_hourly_{var}.pkl'
    if os.path.exists(ofile): os.remove(ofile)
    with open(ofile,'wb') as f:
        pickle.dump(barra_c2_hourly, f)
    
    del barra_c2_hourly




'''
#-------------------------------- check
# 4TB data, 390GB memory storage, 40Gb storage
var = 'cll'
with open(f'data/sim/um/barra_c2/barra_c2_hourly_{var}.pkl','rb') as f:
    barra_c2_hourly = pickle.load(f)

fl = sorted(glob.glob(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/{var}/latest/*'))[:540]
ifile = -1
ds = xr.open_dataset(fl[ifile])

print((barra_c2_hourly[-744:, :, :] == ds[var]).all().values)




    barra_c2_hourly1 = barra_c2_hourly
    ofile = f'data/sim/um/barra_c2/barra_c2_hourly_{var}.nc'
    if os.path.exists(ofile): os.remove(ofile)
    barra_c2_hourly1.to_netcdf(ofile)
'''
# endregion


# region get mon_sea_ann data

cmip6_data = {}
cmip6_data_alltime = {}
cmip6_data_regridded_alltime = {}

for experiment_id in ['piControl']:
    # experiment_id = 'piControl'
    # ['piControl', 'abrupt-4xCO2', 'historical', 'amip', 'ssp585']
    print(f'#-------------------------------- {experiment_id}')
    cmip6_data[experiment_id] = {}
    cmip6_data_alltime[experiment_id] = {}
    cmip6_data_regridded_alltime[experiment_id] = {}
    
    for table_id, variable_id in zip(['Amon'], ['']):
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


# region get ens data

cmip6_data_regridded_alltime = {}
cmip6_data_regridded_alltime_ens = {}

for experiment_id in ['historical']:
    # experiment_id = 'piControl'
    # ['piControl', 'abrupt-4xCO2', 'historical', 'amip', 'ssp585']
    print(f'#-------------------------------- {experiment_id}')
    cmip6_data_regridded_alltime[experiment_id] = {}
    cmip6_data_regridded_alltime_ens[experiment_id] = {}
    
    for table_id, variable_id in zip(['Omon'], ['tos']):
        # table_id = 'Amon'; variable_id = 'tas'
        # ['Amon'], ['tas']
        # ['Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Omon'], ['tas', 'rsut', 'rsdt', 'rlut', 'pr', 'tos']
        print(f'#---------------- {table_id} {variable_id}')
        cmip6_data_regridded_alltime[experiment_id][table_id] = {}
        cmip6_data_regridded_alltime_ens[experiment_id][table_id] = {}
        cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id] = {}
        
        with open(f'/home/563/qg8515/scratch/data/sim/cmip6/{experiment_id}_{table_id}_{variable_id}_regridded_alltime.pkl', 'rb') as f:
            cmip6_data_regridded_alltime[experiment_id][table_id][variable_id] = pickle.load(f)
        
        source_ids = list(cmip6_data_regridded_alltime[experiment_id][table_id][variable_id].keys())
        source_da = xr.DataArray(source_ids, dims='source_id', coords={'source_id': source_ids})
        
        for ialltime in cmip6_data_regridded_alltime[experiment_id][table_id][variable_id][source_ids[0]].keys():
            # ialltime = 'mon'
            print(f'#-------- {ialltime}')
            cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id][ialltime] = xr.concat([cmip6_data_regridded_alltime[experiment_id][table_id][variable_id][source_id][ialltime].chunk() for source_id in source_ids], dim=source_da, coords='minimal', compat='override')
        
        ofile=f'/home/563/qg8515/scratch/data/sim/cmip6/{experiment_id}_{table_id}_{variable_id}_regridded_alltime_ens.pkl'
        if os.path.exists(ofile): os.remove(ofile)
        with open(ofile, 'wb') as f:
            pickle.dump(cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id], f)
        
        del cmip6_data_regridded_alltime[experiment_id][table_id][variable_id]
        del cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id]




'''
#-------------------------------- check
cmip6_data_regridded_alltime = {}
cmip6_data_regridded_alltime_ens = {}

ith_source_id = -1

for experiment_id in ['piControl']:
    # experiment_id = 'piControl'
    # ['piControl', 'abrupt-4xCO2', 'historical', 'amip', 'ssp585']
    print(f'#-------------------------------- {experiment_id}')
    cmip6_data_regridded_alltime[experiment_id] = {}
    cmip6_data_regridded_alltime_ens[experiment_id] = {}
    
    for table_id, variable_id in zip(['Amon'], ['tas']):
        # table_id = 'Amon'; variable_id = 'tas'
        # ['Amon'], ['tas']
        # ['Amon', 'Amon', 'Amon', 'Amon', 'Amon', 'Omon'], ['tas', 'rsut', 'rsdt', 'rlut', 'pr', 'tos']
        print(f'#---------------- {table_id} {variable_id}')
        cmip6_data_regridded_alltime[experiment_id][table_id] = {}
        cmip6_data_regridded_alltime_ens[experiment_id][table_id] = {}
        
        ifile1 = f'/home/563/qg8515/scratch/data/sim/cmip6/{experiment_id}_{table_id}_{variable_id}_regridded_alltime.pkl'
        print(f'File size: {np.round(os.path.getsize(ifile1) / 2**30, 1)}GB')
        with open(ifile1, 'rb') as f:
            cmip6_data_regridded_alltime[experiment_id][table_id][variable_id] = pickle.load(f)
        
        ifile2 = f'/home/563/qg8515/scratch/data/sim/cmip6/{experiment_id}_{table_id}_{variable_id}_regridded_alltime_ens.pkl'
        print(f'File size: {np.round(os.path.getsize(ifile2) / 2**30, 1)}GB')
        with open(ifile2, 'rb') as f:
            cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id] = pickle.load(f)
        
        for ialltime in cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id].keys():
            print(cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id][ialltime].shape)
        
        source_id = list(cmip6_data_regridded_alltime[experiment_id][table_id][variable_id].keys())[ith_source_id]
        print((cmip6_data_regridded_alltime[experiment_id][table_id][variable_id][source_id]['mon'].values == cmip6_data_regridded_alltime_ens[experiment_id][table_id][variable_id]['mon'].sel(source_id=source_id).values).all())


        # check
        for source_id in cmip6_data_regridded_alltime[experiment_id][table_id][variable_id].keys():
            print(cmip6_data_regridded_alltime[experiment_id][table_id][variable_id][source_id]['mon'].shape)
'''
# endregion


# region download data

year_str        = '2022'
month_str_list  = np.arange(2, 3, 1).astype(str).tolist()
day_str_list    = np.arange(6, 10, 1).astype(str).tolist()
area            = '10/122/-45/175'

output_filename = 'data/sim/wrf/input/era5/era5_sl_20220206_09.grib'
get_era5_for_wrf(
    year_str, month_str_list, day_str_list,
    output_filename, area=area,
    surface_only=True,
    )

output_filename = 'data/sim/wrf/input/era5/era5_pl_20220206_09.grib'
get_era5_for_wrf(
    year_str, month_str_list, day_str_list,
    output_filename, area=area,
    surface_only=False,
    )


'''
area            = '10/122/-45/175'
'''
# endregion


# region get CL_Frequency

ISCCP_types = {'Clear': 0,
               'Cirrus': 1, 'Cirrostratus': 2, 'Deep convection': 3,
               'Altocumulus': 4, 'Altostratus': 5, 'Nimbostratus':6,
               'Cumulus':7, 'Stratocumulus': 8, 'Stratus': 9,
               'Unknown':10}

# loop through each day
daterange = pd.date_range(start='1/1/2016', end='12/31/2016')

for idate in daterange:
    # idate = daterange[0]
    print(idate)
    
    year=str(idate)[:4]
    month=str(idate)[5:7]
    day=str(idate)[8:10]
    
    clp_fl = sorted(glob.glob(f'scratch/data/obs/jaxa/clp/{year}{month}/{day}/*/CLP_{year}{month}{day}????.nc'))
    
    clp_ds = xr.open_mfdataset(clp_fl)
    
    CLTYPE_values = clp_ds.CLTYPE.values
    
    CL_Frequency = xr.DataArray(
        name='CL_Frequency',
        data=np.zeros((1, 12, clp_ds.CLTYPE.shape[1], clp_ds.CLTYPE.shape[2])),
        dims=['time', 'types', 'latitude', 'longitude',],
        coords={
            'time': [datetime.strptime(f'{year}-{month}-{day}', '%Y-%m-%d')],
            'types': ['finite'] + list(ISCCP_types.keys()),
            'latitude': clp_ds.CLTYPE.latitude.values,
            'longitude': clp_ds.CLTYPE.longitude.values,
        }
    )
    
    CL_Frequency.loc[{'types': 'finite'}][0] = np.isfinite(CLTYPE_values).sum(axis=0)
    
    for itype in list(ISCCP_types.keys()):
        # print(itype)
        CL_Frequency.loc[{'types': itype}][0] = (CLTYPE_values == ISCCP_types[itype]).sum(axis=0)
    
    print((CL_Frequency[0, 0] == CL_Frequency[0, 1:].sum(axis=0)).all().values)
    print(CL_Frequency[0, 0].sum().values)
    
    ofile = f'scratch/data/obs/jaxa/clp/{year}{month}/{day}/CL_Frequency_{year}{month}{day}.nc'
    if os.path.exists(ofile): os.remove(ofile)
    CL_Frequency.to_netcdf(ofile)
    
    print("Current time:", datetime.now())


aaa = xr.open_dataset('scratch/data/obs/jaxa/clp/201601/01/CL_Frequency_20160101.nc')


'''
himawari_fl = sorted(glob.glob('data/obs/jaxa/clp/*/*/*/NC_*'))
clp_fl = sorted(glob.glob('scratch/data/obs/jaxa/clp/*/*/*/CLP_*'))
print(len(himawari_fl))
print(len(clp_fl))
'''
# endregion


# region animate sounding profiles (theta, RH)

start_date = datetime(2021, 1, 1, 0)
end_date = datetime(2021, 12, 31, 23)
station = '94299'
output_mp4 = 'figures/test1.mp4'

fig, axs = plt.subplots(1, 2, sharey=True, figsize=np.array([8.8, 6.4]) / 2.54)

ims = []
for date in pd.date_range(start_date, end_date, freq='12h'):
    try:
        df = WyomingUpperAir.request_data(date, station)
        df = df[['pressure', 'height', 'temperature', 'dewpoint', 'direction', 'speed',]].dropna(subset=('temperature', 'dewpoint', 'direction', 'speed'), how='any').reset_index(drop=True)
        
        p = df['pressure'].values * units.hPa
        T = (df['temperature'].values * units.degC).to(units('K'))
        Td = (df['dewpoint'].values * units.degC).to(units('K'))
        height = df['height'].values * units.m
        
        thta = mpcalc.potential_temperature(p, T)
        RH = mpcalc.relative_humidity_from_dewpoint(T, Td).to('percent')
        
        plt1 = axs[0].plot(thta, p, c='tab:blue')
        plt2 = axs[1].plot(RH, p, c='tab:blue')
        plt3 = plt.text(0.5, 0.95, str(date)[:13] + ' UTC', ha='center', fontsize=10, transform=fig.transFigure)
        
        ims.append(plt1+plt2 + [plt3])
        print(str(date)[:13])
    except:
        print('No data for ' + str(date)[:13])

axs[0].invert_yaxis()
axs[0].set_ylim(1000, 600)
axs[0].set_xlim(290, 330)
axs[1].set_xlim(0, 100)
axs[0].set_ylabel('Pressure [$hPa$]')
axs[0].set_xlabel(r'$\theta$ [$K$]')
axs[1].set_xlabel(r'RH [$\%$]')
axs[0].grid(lw=0.2, alpha=0.5, ls='--')
axs[1].grid(lw=0.2, alpha=0.5, ls='--')

# 2nd y-axis
height = np.round(pressure_to_height_std(
    pressure=np.arange(1000, 600-1e-4, -100) * units('hPa')), 1,)
ax2 = axs[1].twinx()
ax2.invert_yaxis()
ax2.set_ylim(1000, 600)
ax2.set_yticks(np.arange(1000, 600-1e-4, -100))
ax2.set_yticklabels(height.magnitude, c = 'gray')
ax2.set_ylabel('Altitude in a standard atmosphere [$km$]', c = 'gray')

fig.subplots_adjust(0.18, 0.18, 0.85, 0.88)
ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True)
ani.save(
    output_mp4,
    progress_callback=lambda iframe, n: print(f'Frame {iframe} of {n}'),)

# endregion


# region get and plot Wyoming sounding (theta and RH)

date = datetime(2021, 12, 8, 0)
station = '94299'
output_png='figures/test.png'

df = WyomingUpperAir.request_data(date, station)
plot_wyoming_sounding_vertical(df, date, output_png=output_png)



'''
print(df.columns)
'''
# endregion


# region get and plot Wyoming sounding (T, theta, RH)

date = datetime(2021, 9, 16, 0)
station = '94299'
output_png='figures/test.png'

df = WyomingUpperAir.request_data(date, station)
df = df[['pressure', 'height', 'temperature', 'dewpoint', 'direction', 'speed']].dropna(subset=('temperature', 'dewpoint', 'direction', 'speed'), how='all').reset_index(drop=True)

p = df['pressure'].values * units.hPa
T = (df['temperature'].values * units.degC).to(units('K'))
Td = (df['dewpoint'].values * units.degC).to(units('K'))
height = df['height'].values * units.m

thta = mpcalc.potential_temperature(p, T)
RH = mpcalc.relative_humidity_from_dewpoint(T, Td).to('percent')

fig, axs = plt.subplots(1, 3, sharey=True, figsize=np.array([8.8, 6.4]) / 2.54)

axs[0].plot(T, p)
axs[1].plot(thta, p)
axs[2].plot(RH, p)

axs[0].invert_yaxis()
axs[0].set_ylim(1000, 600)
axs[0].set_xlim(270, 310)
axs[1].set_xlim(290, 330)
axs[2].set_xlim(0, 100)
axs[0].set_ylabel('Pressure [$hPa$]')
axs[0].set_xlabel(r'T [$K$]')
axs[1].set_xlabel(r'$\theta$ [$K$]')
axs[2].set_xlabel(r'RH [$\%$]')
axs[0].grid(lw=0.2, alpha=0.5, ls='--')
axs[1].grid(lw=0.2, alpha=0.5, ls='--')
axs[2].grid(lw=0.2, alpha=0.5, ls='--')

# 2nd y-axis
height = np.round(pressure_to_height_std(
    pressure=np.arange(1000, 600-1e-4, -100) * units('hPa')), 1,)
ax2 = axs[2].twinx()
ax2.invert_yaxis()
ax2.set_ylim(1000, 600)
ax2.set_yticks(np.arange(1000, 600-1e-4, -100))
ax2.set_yticklabels(height.magnitude, c = 'gray')
ax2.set_ylabel('Altitude in a standard atmosphere [$km$]', c = 'gray')

plt.suptitle(str(date)[:13] + ' UTC', fontsize=10)
plt.subplots_adjust(0.18, 0.18, 0.85, 0.88)
plt.savefig(output_png)


# endregion


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


# endregion


# region plot data

with open('data/sim/cmip6/historical_Omon_tos.pkl', 'rb') as f:
    historical_Omon_tos = pickle.load(f)

models = list(historical_Omon_tos.keys())

output_png = 'figures/test.png'
cbar_label = r'CMIP6 $\mathit{historical}$' + ' monthly SST [$°C$]'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=28, cm_interval1=1, cm_interval2=2, cmap='viridis_r',)

nrow = 10
ncol = 6
fm_bottom = 2 / (4.4*nrow + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol, 4.4*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.Mollweide(central_longitude=180)},
    gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)

for irow in range(nrow):
    for jcol in range(ncol):
        if (jcol + ncol * irow < len(models)):
            panel_label = f'({string.ascii_lowercase[irow]}{jcol+1})'
            model = models[jcol + ncol * irow]
            
            axs[irow, jcol] = globe_plot(ax_org=axs[irow, jcol])
            
            axs[irow, jcol].text(
                -0.04, 1.04, model + '\n' + panel_label,
                ha='left', va='top', transform=axs[irow, jcol].transAxes,
                linespacing=1.6)
        else:
            axs[irow, jcol].set_visible(False)

# plot data
for irow in range(nrow):
    for jcol in range(ncol):
        if (jcol + ncol * irow < len(models)):
            model = models[jcol + ncol * irow]
            print(model)
            
            plot_data = historical_Omon_tos[model]['ann'].sel(
                time=slice('1979', '2014')).mean(dim='time')
            
            plt_mesh = axs[irow, jcol].contourf(
                lon, lat, plot_data, levels=pltlevel, extend='both',
                norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
            axs[irow, jcol].add_feature(
                cfeature.LAND, color='white', zorder=2, edgecolor=None,lw=0)

cbar = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp),
    ax=axs, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.5, ticks=pltticks, extend='both',
    anchor=(0.5, -0.84),)
cbar.ax.tick_params(length=2, width=0.4)
cbar.ax.set_xlabel(cbar_label)

fig.subplots_adjust(left=0.01, right = 0.996, bottom = fm_bottom, top = 0.99)
fig.savefig(output_png)
plt.close()

# endregion

