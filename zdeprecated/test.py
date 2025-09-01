

# region plot obs and sim data

year, month, day, hour = 2020, 6, 2, 4
suites = ['u-dr144', 'u-dr145', 'u-dr146', 'u-dr147', 'u-dr148', 'u-dr149'] # ['u-dq700', 'u-dq788', 'u-dq799', 'u-dq911', 'u-dq912', 'u-dq987', 'u-dr040', 'u-dr041', 'u-dr091', 'u-dr093', 'u-dr095', 'u-dr105', 'u-dr107', 'u-dr108', 'u-dr109', 'u-dr144', 'u-dr145', 'u-dr146', 'u-dr147', 'u-dr148', 'u-dr149']
var2s = ['rsut', 'rsutcs']
modes = ['original', 'difference']

suite_res = {
    'u-dq700': ['d11km', 'd4p4km'],
    'u-dq788': ['d11km', 'd4p4kms'],
    'u-dq799': ['d11km', 'd1p1km'],
    'u-dq911': ['d11km', 'd2p2km'],
    'u-dq912': ['d11km', 'd4p4kml'],
    'u-dq987': ['d11km', 'd4p4km'],
    'u-dr040': ['d11km', 'd4p4km'],
    'u-dr041': ['d11km', 'd4p4km'],
    'u-dr091': ['d11km', 'd1p1kmsa'],
    'u-dr093': ['d11km', 'd2p2kmsa'],
    'u-dr095': ['d11km', 'd4p4kmsa'],
    'u-dr105': ['d11km', 'd4p4km'],
    'u-dr107': ['d11km', 'd4p4km'],
    'u-dr108': ['d11km', 'd4p4km'],
    'u-dr109': ['d11km', 'd4p4km'],
    'u-dr144': ['d11km', 'd4p4kms'],
    'u-dr145': ['d11km', 'd1p1km'],
    'u-dr146': ['d11km', 'd2p2km'],
    'u-dr147': ['d11km', 'd1p1kmsa'],
    'u-dr148': ['d11km', 'd2p2kmsa'],
    'u-dr149': ['d11km', 'd4p4kmsa'],
}

ntime = pd.Timestamp(year,month,day,hour) + pd.Timedelta('1h')
year1, month1, day1, hour1 = ntime.year, ntime.month, ntime.day, ntime.hour
ptime = pd.Timestamp(year,month,day,hour) - pd.Timedelta('1h')
year0, month0, day0, hour0 = ptime.year, ptime.month, ptime.day, ptime.hour

min_lon1, max_lon1, min_lat1, max_lat1 = 80, 220, -70, 20
min_lon, max_lon, min_lat, max_lat = 110.58, 157.34, -43.69, -7.01
pwidth  = 7
pheight = pwidth * (max_lat1 - min_lat1) / (max_lon1 - min_lon1)
nrow = 1
fm_bottom = 1.6/(pheight*nrow+2.1)
fm_top = 1 - 0.5/(pheight*nrow+2.1)
regridder = {}


for isuite in suites:
    # isuite='u-dq700'
    # ['u-dq700', 'u-dq788', 'u-dq911', 'u-dq912', 'u-dq799', 'u-dq987', 'u-dr040', 'u-dr041']
    print(f'#-------------------------------- {isuite}')
    
    for var2 in var2s:
        # var2 = 'rsdt'
        var1 = cmip6_era5_var[var2]
        print(f'#---------------- {var1} vs. {var2}')
        
        if var2 == 'prw':
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=80, cm_interval1=5, cm_interval2=10, cmap='Blues_r')
            extend = 'max'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-10,cm_max=10,cm_interval1=1,cm_interval2=2,cmap='BrBG_r')
            extend2 = 'both'
        elif var2 in ['cll', 'clm', 'clh', 'clt']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=100, cm_interval1=5, cm_interval2=10, cmap='Blues_r')
            extend = 'neither'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-100,cm_max=100,cm_interval1=10,cm_interval2=20,cmap='BrBG_r')
            extend2 = 'neither'
        elif var2 in ['clwvi']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=0.6, cm_interval1=0.05, cm_interval2=0.1, cmap='Blues_r')
            extend = 'max'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-0.6,cm_max=0.6,cm_interval1=0.1,cm_interval2=0.1,cmap='BrBG_r')
            extend2 = 'both'
        elif var2 in ['clivi']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=1, cm_interval1=0.1, cm_interval2=0.2, cmap='Blues_r')
            extend = 'max'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-1,cm_max=1,cm_interval1=0.1,cm_interval2=0.2,cmap='BrBG_r')
            extend2 = 'both'
        elif var2=='pr':
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=8, cm_interval1=1, cm_interval2=1, cmap='Blues_r')
            extend = 'max'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-4,cm_max=4,cm_interval1=1,cm_interval2=1,cmap='BrBG_r')
            extend2 = 'both'
        elif var2 in ['evspsbl', 'evspsblpot']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=1, cm_interval1=0.1, cm_interval2=0.2, cmap='Blues_r')
            extend = 'max'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-1,cm_max=1,cm_interval1=0.1,cm_interval2=0.2,cmap='BrBG_r')
            extend2 = 'both'
        elif var2 in ['hfls']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=-600, cm_max=300, cm_interval1=50, cm_interval2=100, cmap='PRGn', asymmetric=True)
            extend = 'both'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-30,cm_max=30,cm_interval1=5,cm_interval2=10,cmap='BrBG')
            extend2 = 'both'
        elif var2 in ['hfss']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=-400, cm_max=400, cm_interval1=50, cm_interval2=100, cmap='PRGn')
            extend = 'both'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-30,cm_max=30,cm_interval1=5,cm_interval2=10,cmap='BrBG')
            extend2 = 'both'
        elif var2 in ['sfcWind']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=24, cm_interval1=2, cm_interval2=4, cmap='Purples_r')
            extend = 'max'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-4,cm_max=4,cm_interval1=1,cm_interval2=1,cmap='BrBG_r')
            extend2 = 'both'
        elif var2 in ['tas', 'ts']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=-20, cm_max=35, cm_interval1=2.5, cm_interval2=5, cmap='PuOr', asymmetric=True)
            extend = 'both'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-4,cm_max=4,cm_interval1=1,cm_interval2=1,cmap='BrBG')
            extend2 = 'both'
        elif var2 in ['das']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=-20, cm_max=35, cm_interval1=2.5, cm_interval2=5, cmap='PuOr', asymmetric=True)
            extend = 'both'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-4,cm_max=4,cm_interval1=1,cm_interval2=1,cmap='BrBG')
            extend2 = 'both'
        elif var2 in ['huss']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=24, cm_interval1=2, cm_interval2=4, cmap='Blues_r')
            extend = 'max'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-4,cm_max=4,cm_interval1=0.5,cm_interval2=1,cmap='BrBG_r')
            extend2 = 'both'
        elif var2 in ['hurs']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=20, cmap='Blues_r')
            extend = 'neither'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-10,cm_max=10,cm_interval1=1,cm_interval2=2,cmap='BrBG_r')
            extend2 = 'both'
        elif var2 in ['rsut']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=-800, cm_max=0, cm_interval1=50, cm_interval2=100, cmap='Greens')
            extend = 'min'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-400,cm_max=400,cm_interval1=50,cm_interval2=100,cmap='BrBG')
            extend2 = 'both'
        elif var2 in ['rsutcs']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=-100, cm_max=0, cm_interval1=10, cm_interval2=10, cmap='Greens')
            extend = 'min'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10, cmap='BrBG')
            extend2 = 'both'
        elif var2 in ['rlut', 'rlutcs']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=-360, cm_max=0, cm_interval1=20, cm_interval2=40, cmap='Greens')
            extend = 'min'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-180,cm_max=180,cm_interval1=20,cm_interval2=40,cmap='BrBG')
            extend2 = 'both'
        elif var2 in ['rlds', 'rldscs']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=80, cm_max=430, cm_interval1=10, cm_interval2=20, cmap='viridis_r',)
            extend = 'both'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10, cmap='BrBG')
            extend2 = 'both'
        elif var2=='rsdt':
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=150, cm_max=490, cm_interval1=10, cm_interval2=40, cmap='viridis_r',)
            extend = 'both'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-1, cm_max=1, cm_interval1=0.2, cm_interval2=0.2, cmap='BrBG',)
            extend2 = 'both'
        elif var2 in ['rsds', 'rsdscs']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=40, cm_max=400, cm_interval1=10, cm_interval2=40, cmap='viridis_r',)
            extend = 'both'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10, cmap='BrBG')
            extend2 = 'both'
        elif var2 in ['rlns', 'rlnscs']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=-150, cm_max=0, cm_interval1=10, cm_interval2=20, cmap='viridis',)
            extend = 'both'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10, cmap='BrBG')
            extend2 = 'both'
        elif var2 in ['blh']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=2000, cm_interval1=100, cm_interval2=400, cmap='viridis')
            extend = 'max'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-600,cm_max=600,cm_interval1=50,cm_interval2=200,cmap='BrBG')
            extend2 = 'both'
        elif var2 in ['orog']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=2000, cm_interval1=100, cm_interval2=400, cmap='viridis')
            extend = 'max'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-300,cm_max=300,cm_interval1=50,cm_interval2=100,cmap='BrBG')
            extend2 = 'both'
        elif var2=='psl':
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=1005, cm_max=1022, cm_interval1=1, cm_interval2=2, cmap='viridis_r',)
            extend = 'both'
            pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
                cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='BrBG')
            extend2 = 'both'
        else:
            print('Warning: no colorbar specified')
        
        ds = {}
        
        if var2 in ['rsdt', 'rsut', 'rsutcs', 'rlut', 'rlutcs']:
            ds['CERES'] = xr.open_dataset(f'data/obs/CERES/CERES_SYN1deg-1H_Terra-Aqua-NOAA20_Ed4.2_Subset_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc').rename({
                'toa_sw_clr_1h':    'rsutcs',
                'toa_sw_all_1h':    'rsut',
                'toa_lw_clr_1h':    'rlutcs',
                'toa_lw_all_1h':    'rlut',
                'toa_solar_all_1h': 'rsdt',})
            ds['CERES'] = ds['CERES'][var2].sel(time=pd.Timestamp(year,month,day,hour) + pd.Timedelta('30min'), method='nearest').sel(lon=slice(min_lon1, max_lon1), lat=slice(min_lat1, max_lat1))
            if var2 in ['rsut', 'rsutcs', 'rlut', 'rlutcs']:
                ds['CERES'] *= (-1)
        
        if var1 in ['t2m', 'd2m', 'u10', 'v10', 'u100', 'v100']:
            if var1 == 't2m': vart = '2t'
            if var1 == 'd2m': vart = '2d'
            if var1 == 'u10': vart = '10u'
            if var1 == 'v10': vart = '10v'
            if var1 == 'u100': vart = '100u'
            if var1 == 'v100': vart = '100v'
            ds['ERA5'] = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/{vart}/{year}/{vart}_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')[var1].sel(time=pd.Timestamp(year,month,day,hour))
        elif var1=='orog':
            ds['ERA5'] = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/z/{year}/z_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['z'].sel(time=pd.Timestamp(year,month,day,hour))
        elif var1=='rh2m':
            era5_t2m = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/2t/{year}/2t_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['t2m'].sel(time=pd.Timestamp(year,month,day,hour))
            era5_d2m = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/2d/{year}/2d_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['d2m'].sel(time=pd.Timestamp(year,month,day,hour))
            ds['ERA5'] = relative_humidity_from_dewpoint(era5_t2m * units.K, era5_d2m * units.K) * 100
        elif var1=='q2m':
            era5_sp = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/sp/{year}/sp_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['sp'].sel(time=pd.Timestamp(year,month,day,hour))
            era5_d2m = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/2d/{year}/2d_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['d2m'].sel(time=pd.Timestamp(year,month,day,hour))
            ds['ERA5'] = specific_humidity_from_dewpoint(era5_sp * units.Pa, era5_d2m * units.K) * 1000
        elif var1=='mtuwswrf':
            era5_mtnswrf = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/mtnswrf/{year1}/mtnswrf_era5_oper_sfc_{year1}{month1:02d}01-{year1}{month1:02d}{calendar.monthrange(year1, month1)[1]}.nc')['mtnswrf'].sel(time=pd.Timestamp(year1,month1,day1,hour1))
            era5_mtdwswrf = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/mtdwswrf/{year1}/mtdwswrf_era5_oper_sfc_{year1}{month1:02d}01-{year1}{month1:02d}{calendar.monthrange(year1, month1)[1]}.nc')['mtdwswrf'].sel(time=pd.Timestamp(year1,month1,day1,hour1))
            ds['ERA5'] = era5_mtnswrf - era5_mtdwswrf
        elif var1=='mtuwswrfcs':
            era5_mtnswrfcs = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/mtnswrfcs/{year1}/mtnswrfcs_era5_oper_sfc_{year1}{month1:02d}01-{year1}{month1:02d}{calendar.monthrange(year1, month1)[1]}.nc')['mtnswrfcs'].sel(time=pd.Timestamp(year1,month1,day1,hour1))
            era5_mtdwswrf = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/mtdwswrf/{year1}/mtdwswrf_era5_oper_sfc_{year1}{month1:02d}01-{year1}{month1:02d}{calendar.monthrange(year1, month1)[1]}.nc')['mtdwswrf'].sel(time=pd.Timestamp(year1,month1,day1,hour1))
            ds['ERA5'] = era5_mtnswrfcs - era5_mtdwswrf
        elif var1=='si10':
            era5_u10 = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/10u/{year}/10u_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['u10'].sel(time=pd.Timestamp(year,month,day,hour))
            era5_v10 = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/10v/{year}/10v_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')['v10'].sel(time=pd.Timestamp(year,month,day,hour))
            ds['ERA5'] = (era5_u10**2 + era5_v10**2)**0.5
        elif var1 in ['tp', 'e', 'pev', 'mslhf', 'msshf', 'mtnlwrf', 'msdwlwrf', 'mtdwswrf', 'msdwswrfcs', 'msdwswrf', 'msnlwrf', 'mtnlwrfcs', 'msdwlwrfcs', ]:
            ds['ERA5'] = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/{var1}/{year1}/{var1}_era5_oper_sfc_{year1}{month1:02d}01-{year1}{month1:02d}{calendar.monthrange(year1, month1)[1]}.nc')[var1].sel(time=pd.Timestamp(year1,month1,day1,hour1))
        elif var1 in ['skt', 'blh', 'msl', 'tcwv', 'hcc', 'mcc', 'lcc', 'tcc', 'tclw', 'tciw']:
            ds['ERA5'] = xr.open_dataset(f'/g/data/rt52/era5/single-levels/reanalysis/{var1}/{year}/{var1}_era5_oper_sfc_{year}{month:02d}01-{year}{month:02d}{calendar.monthrange(year, month)[1]}.nc')[var1].sel(time=pd.Timestamp(year,month,day,hour))
        else:
            print('Warning: var file not found')
        
        ds['ERA5']['longitude'] = ds['ERA5']['longitude'] % 360
        ds['ERA5'] = ds['ERA5'].sortby(['longitude', 'latitude']).rename({'latitude': 'lat', 'longitude': 'lon'}).sel(lon=slice(min_lon1, max_lon1), lat=slice(min_lat1, max_lat1))
        
        if var1 in ['tp', 'e', 'cp', 'lsp', 'pev']:
            ds['ERA5'] *= 24000 / 24
        elif var1 in ['msl']:
            ds['ERA5'] /= 100
        elif var1 in ['sst', 't2m', 'd2m', 'skt']:
            ds['ERA5'] -= zerok
        elif var1 in ['hcc', 'mcc', 'lcc', 'tcc']:
            ds['ERA5'] *= 100
        elif var1 in ['z']:
            ds['ERA5'] /= 9.80665
        elif var1 in ['mper']:
            ds['ERA5'] *= seconds_per_d / 24
        
        if var1 in ['e', 'pev', 'mper']:
            ds['ERA5'] *= (-1)
        
        for ires in suite_res[isuite]:
            # ires = 'd11km'
            # ['d11km', 'd4p4km', 'd4p4kms', 'd1p1km']
            print(f'#-------- {ires}')
            
            if var2 in ['orog']:
                # instanteneous variable
                ds[ires] = xr.open_dataset(sorted(glob.glob(f'/home/563/qg8515/cylc-run/{isuite}/share/cycle/{year}{month:02d}{day:02d}T0000Z/Australia/{ires}/*/um/umnsaa_pa000.nc'))[0]).pipe(preprocess_umoutput)[var2stash[var2]]
            elif var2 in ['ts', 'blh', 'tas', 'huss', 'hurs', 'das', 'clslw', 'psl', 'CAPE', 'prw']:
                # instanteneous variable
                ds[ires] = xr.open_dataset(sorted(glob.glob(f'/home/563/qg8515/cylc-run/{isuite}/share/cycle/{year}{month:02d}{day:02d}T0000Z/Australia/{ires}/*/um/umnsaa_pa000.nc'))[0]).pipe(preprocess_umoutput)[var2stash[var2]].sel(time=pd.Timestamp(year,month,day,hour))
            elif var2 in ['rlds', 'rlu_t_s', 'rsut', 'rsdt', 'rsutcs', 'rsdscs', 'rsds', 'rlns', 'rlut', 'rlutcs', 'rldscs', 'hfss', 'hfls', 'rain', 'snow', 'pr', 'cll', 'clm', 'clh', 'clt', 'clwvi', 'clivi']:
                # time mean
                ds[ires] = xr.open_dataset(sorted(glob.glob(f'/home/563/qg8515/cylc-run/{isuite}/share/cycle/{year1}{month1:02d}{day1:02d}T0000Z/Australia/{ires}/*/um/umnsaa_pa000.nc'))[0]).pipe(preprocess_umoutput)[var2stash[var2]].sel(time=pd.Timestamp(year1,month1,day1,hour1))
            
            if var2 in ['pr', 'evspsbl', 'evspsblpot']:
                ds[ires] *= seconds_per_d / 24
            elif var2 in ['tas', 'ts']:
                ds[ires] -= zerok
            elif var2 in ['rlus', 'rluscs', 'rlut', 'rlutcs', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'hfls', 'hfss']:
                ds[ires] *= (-1)
            elif var2 in ['psl']:
                ds[ires] /= 100
            elif var2 in ['huss']:
                ds[ires] *= 1000
            elif var2 in ['cll', 'clm', 'clh', 'clt']:
                ds[ires] *= 100
        
        for imode in modes:
            # imode = 'difference'
            print(f'#---- {imode}')
            
            plt_colnames = list(ds.keys())
            if imode=='difference':
                plt_colnames = [plt_colnames[0]] + [f'{ids} - {plt_colnames[0]}' for ids in plt_colnames[1:]]
            
            opng = f"figures/4_um/4.1_access_ram3/4.1.1_sim_obs/4.1.1.0_{year}-{month:02d}-{day:02d}-{hour:02d} {var2} in {isuite} {', '.join(suite_res[isuite])}, ERA5, and {plt_colnames[0]}, {imode}, {min_lon1}_{max_lon1}_{min_lat1}_{max_lat1}.png"
            
            ncol = len(plt_colnames)
            fig, axs = plt.subplots(
                nrow, ncol,
                figsize=np.array([pwidth*ncol, pheight*nrow+2.1])/2.54,
                subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
                gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)
            
            for jcol in range(ncol):
                axs[jcol] = regional_plot(extent=[min_lon1, max_lon1, min_lat1, max_lat1], central_longitude=180, ax_org=axs[jcol], lw=0.1)
                axs[jcol].add_patch(Rectangle(
                    (min_lon, min_lat), max_lon-min_lon, max_lat-min_lat,
                    ec='red', color='None', lw=0.5,
                    transform=ccrs.PlateCarree(), zorder=2))
                axs[jcol].add_patch(Rectangle(
                    (ds[suite_res[isuite][0]].lon[0], ds[suite_res[isuite][0]].lat[0]),
                    ds[suite_res[isuite][0]].lon[-1] - ds[suite_res[isuite][0]].lon[0],
                    ds[suite_res[isuite][0]].lat[-1] - ds[suite_res[isuite][0]].lat[0],
                    ec='red', color='None', lw=0.5, linestyle='--',
                    transform=ccrs.PlateCarree(), zorder=2))
                axs[jcol].add_patch(Rectangle(
                    (ds[suite_res[isuite][1]].lon[0], ds[suite_res[isuite][1]].lat[0]),
                    ds[suite_res[isuite][1]].lon[-1] - ds[suite_res[isuite][1]].lon[0],
                    ds[suite_res[isuite][1]].lat[-1] - ds[suite_res[isuite][1]].lat[0],
                    ec='red', color='None', lw=0.5, linestyle=':',
                    transform=ccrs.PlateCarree(), zorder=2))
            
            if imode=='original':
                for jcol, ids in enumerate(ds.keys()):
                    # print(f'#---- {jcol} {ids}')
                    plt_mesh = axs[jcol].pcolormesh(
                        ds[ids].lon,
                        ds[ids].lat,
                        ds[ids],
                        norm=pltnorm, cmap=pltcmp,
                        transform=ccrs.PlateCarree(), zorder=1)
                    cbar = fig.colorbar(
                        plt_mesh, #cm.ScalarMappable(norm=pltnorm1, cmap=pltcmp1), #
                        format=remove_trailing_zero_pos,
                        orientation="horizontal", ticks=pltticks, extend=extend,
                        cax=fig.add_axes([1/3, fm_bottom-0.115, 1/3, 0.03]))
                    cbar.ax.set_xlabel(era5_varlabels[var1].replace('day^{-1}', 'hour^{-1}'), fontsize=9, labelpad=1)
                    cbar.ax.tick_params(labelsize=9, pad=1)
            elif imode=='difference':
                plt_mesh = axs[0].pcolormesh(
                    ds[list(ds.keys())[0]].lon,
                    ds[list(ds.keys())[0]].lat,
                    ds[list(ds.keys())[0]],
                    norm=pltnorm, cmap=pltcmp,
                    transform=ccrs.PlateCarree(), zorder=1)
                for jcol, ids1 in zip(range(1, ncol), list(ds.keys())[1:]):
                    ids2 = list(ds.keys())[0]
                    # print(f'#-------- {jcol} {ids1} {ids2}')
                    if not f'{ids1} - {ids2}' in regridder.keys():
                        regridder[f'{ids1} - {ids2}'] = xe.Regridder(
                            ds[ids1],
                            ds[ids2].sel(lon=slice(ds[ids1].lon[0], ds[ids1].lon[-1]), lat=slice(ds[ids1].lat[0], ds[ids1].lat[-1])),
                            method='bilinear')
                    plt_data = regridder[f'{ids1} - {ids2}'](ds[ids1]) - ds[ids2].sel(lon=slice(ds[ids1].lon[0], ds[ids1].lon[-1]), lat=slice(ds[ids1].lat[0], ds[ids1].lat[-1]))
                    rmse = np.sqrt(np.square(plt_data).weighted(np.cos(np.deg2rad(plt_data.lat))).mean()).values
                    plt_colnames[jcol] = f'{plt_colnames[jcol]}, RMSE: {str(np.round(rmse, 2))}'
                    plt_mesh2 = axs[jcol].pcolormesh(
                        plt_data.lon, plt_data.lat, plt_data,
                        norm=pltnorm2, cmap=pltcmp2,
                        transform=ccrs.PlateCarree(), zorder=1)
                cbar = fig.colorbar(
                    plt_mesh, #cm.ScalarMappable(norm=pltnorm1, cmap=pltcmp1), #
                    format=remove_trailing_zero_pos,
                    orientation="horizontal", ticks=pltticks, extend=extend,
                    cax=fig.add_axes([0.05, fm_bottom-0.115, 0.4, 0.03]))
                cbar.ax.set_xlabel(era5_varlabels[var1].replace('day^{-1}', 'hour^{-1}'), fontsize=9, labelpad=1)
                cbar.ax.tick_params(labelsize=9, pad=1)
                cbar2 = fig.colorbar(
                    plt_mesh2, #cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), #
                    format=remove_trailing_zero_pos,
                    orientation="horizontal", ticks=pltticks2, extend=extend2,
                    cax=fig.add_axes([0.55, fm_bottom-0.115, 0.4, 0.03]))
                cbar2.ax.set_xlabel(f"Difference in {era5_varlabels[var1].replace('day^{-1}', 'hour^{-1}')}",
                                    fontsize=9, labelpad=1)
                cbar2.ax.tick_params(labelsize=9, pad=1)
            
            for jcol in range(ncol):
                axs[jcol].text(0, 1.02, f'({string.ascii_lowercase[jcol]}) {plt_colnames[jcol]}', ha='left', va='bottom', transform=axs[jcol].transAxes)
            
            fig.text(0.5, fm_bottom-0.02, f'{year}-{month:02d}-{day:02d} {hour:02d}:00 UTC', ha='center', va='top')
            fig.subplots_adjust(left=0.005, right=0.995, bottom=fm_bottom, top=fm_top)
            fig.savefig(opng)
        
        del ds



'''
            for ivar in ds[ires].data_vars:
                print(f'{ivar}: {ds[ires][ivar].attrs['cell_methods']}')
'''
# endregion


# region plot MODIS Terra and Aqua QKM and HKM


year, month, day, hour = 2020, 6, 2, 4
doy = datetime(year, month, day).timetuple().tm_yday

fl = {}
for iproduct in ['MOD02QKM', 'MYD02QKM', 'MOD02HKM', 'MYD02HKM']:
    # 'MOD021KM', 'MYD021KM'
    # print(f'#-------------------------------- {iproduct}')
    fl[iproduct] = sorted(glob.glob(f'scratch/data/obs/MODIS/{iproduct}/{year}/{doy:03d}/*.{hour:02d}??.061.*.hdf'))

sat_product = {'Terra': ['MOD02QKM', 'MOD02HKM'],
               'Aqua':  ['MYD02QKM', 'MYD02HKM']}

# fig, ax = globe_plot()

for isat in ['Terra', 'Aqua']:
    # isat = 'Terra'
    print(f'#---------------- {isat}: {sat_product[isat]}')
    if (len(fl[sat_product[isat][0]]) != len(fl[sat_product[isat][1]])):
        print('Warning: File not matching')
        continue
    
    for ifileQ, ifileH in zip(fl[sat_product[isat][0]], fl[sat_product[isat][1]]):
        # ifileQ=fl[sat_product[isat][0]][0]; ifileH=fl[sat_product[isat][1]][0]
        if ifileQ.split('.')[2] != ifileH.split('.')[2]:
            print('Warning: Time not matching')
            continue
        
        hdf = SD(ifileQ, SDC.READ)
        EV_RefSB = hdf.select('EV_250_RefSB')
        red_reflectance = si2reflectance(
            EV_RefSB[0],
            scales=EV_RefSB.attributes()['reflectance_scales'][0],
            offsets=EV_RefSB.attributes()['reflectance_offsets'][0])
        
        hdf = SD(ifileH, SDC.READ)
        EV_RefSB = hdf.select('EV_500_RefSB')
        green_reflectance = si2reflectance(
            EV_RefSB[1],
            scales=EV_RefSB.attributes()['reflectance_scales'][1],
            offsets=EV_RefSB.attributes()['reflectance_offsets'][1])
        blue_reflectance = si2reflectance(
            EV_RefSB[0],
            scales=EV_RefSB.attributes()['reflectance_scales'][0],
            offsets=EV_RefSB.attributes()['reflectance_offsets'][0])
        green_reflectance = np.kron(green_reflectance, np.ones((2, 2)))
        blue_reflectance = np.kron(blue_reflectance, np.ones((2, 2)))
        
        rgb = np.dstack([red_reflectance, green_reflectance, blue_reflectance])
        color_tuples = np.array([red_reflectance.flatten(),
                                 green_reflectance.flatten(),
                                 blue_reflectance.flatten()]).transpose()
        
        scn = Scene(filenames={'modis_l1b': [ifileQ]})
        scn.load(["longitude", "latitude"])
        lon = scn["longitude"].values
        lat = scn["latitude"].values
        
        fig, ax = globe_plot()
        ax.pcolormesh(lon, lat, red_reflectance, color=color_tuples,
                      transform=ccrs.PlateCarree())
        fig.savefig('figures/test.png')

# fig.savefig('figures/test1.png')




'''


varnames = {'MOD02QKM': 'EV_250_RefSB', 'MYD02QKM': 'EV_250_RefSB',
            'MOD02HKM': 'EV_500_RefSB', 'MYD02HKM': 'EV_500_RefSB',
            'MOD021KM': 'EV_1KM_RefSB', 'MYD021KM': 'EV_1KM_RefSB'}

datetime(2020, 6, 1).timetuple().tm_yday
datetime(2020, 6, 30).timetuple().tm_yday


for iproduct in ['MOD02QKM', 'MYD02QKM', 'MOD02HKM', 'MYD02HKM']:
    # iproduct = 'MOD02QKM'
    # iproduct = 'MOD02HKM'
    # 'MOD021KM', 'MYD021KM'
    print(f'#-------------------------------- {iproduct}')
    
    for ifile in fl[iproduct]:
        # ifile = fl[iproduct][0]
        print(f'{ifile}')
        
        hdf = SD(ifile, SDC.READ)
        scn = Scene(filenames={'modis_l1b': [ifile]})
        scn.load(["longitude", "latitude"])
        
        if iproduct in ['MOD02QKM', 'MYD02QKM']:
            EV_RefSB = hdf.select('EV_250_RefSB')
            red_reflectance = si2reflectance(
                EV_RefSB[0],
                scales=EV_RefSB.attributes()['reflectance_scales'][0],
                offsets=EV_RefSB.attributes()['reflectance_offsets'][0])
        elif iproduct in ['MOD02HKM', 'MYD02HKM']:
            EV_RefSB = hdf.select('EV_500_RefSB')
            green_reflectance = si2reflectance(
                EV_RefSB[1],
                scales=EV_RefSB.attributes()['reflectance_scales'][1],
                offsets=EV_RefSB.attributes()['reflectance_offsets'][1])
            blue_reflectance = si2reflectance(
                EV_RefSB[0],
                scales=EV_RefSB.attributes()['reflectance_scales'][0],
                offsets=EV_RefSB.attributes()['reflectance_offsets'][0])
            green_reflectance = np.kron(green_reflectance, np.ones((2, 2)))
            blue_reflectance = np.kron(blue_reflectance, np.ones((2, 2)))


'''
# endregion


# region plot 'MOD021KM' 'MYD021KM': Calibrated Radiances


year, month, day, hour = 2020, 6, 2, 3
doy = datetime(year, month, day).timetuple().tm_yday

fl = {}
for iproduct in ['MOD021KM', 'MYD021KM']:
    # 'MOD02QKM', 'MYD02QKM', 'MOD02HKM', 'MYD02HKM', 'MOD021KM', 'MYD021KM'
    # print(f'#-------------------------------- {iproduct}')
    fl[iproduct] = sorted(glob.glob(f'scratch/data/obs/MODIS/{iproduct}/{year}/{doy:03d}/*.{hour:02d}??.061.*.hdf'))

sat_product = {'Terra': 'MOD021KM', 'Aqua':  'MYD021KM'}


fig, ax = globe_plot(figsize=np.array([88, 44]) / 2.54, lw=1)

for isat in ['Aqua']:
    # isat = 'Terra'
    # ['Terra', 'Aqua']
    print(f'#-------------------------------- {isat}')
    for ifile in fl[sat_product[isat]]:
        # ifile = fl[sat_product[isat]][0]
        print(f'#---- {ifile}')
        
        hdf = SD(ifile, SDC.READ)
        scn = Scene(filenames={'modis_l1b': [ifile]})
        scn.load(["longitude", "latitude"])
        lon = scn["longitude"].values
        lat = scn["latitude"].values
        
        
        EV_RefSB = hdf.select('EV_250_Aggr1km_RefSB')
        red_reflectance = si2reflectance(
            EV_RefSB[0],
            scales=EV_RefSB.attributes()['reflectance_scales'][0],
            offsets=EV_RefSB.attributes()['reflectance_offsets'][0])
        
        EV_RefSB = hdf.select('EV_500_Aggr1km_RefSB')
        green_reflectance = si2reflectance(
            EV_RefSB[1],
            scales=EV_RefSB.attributes()['reflectance_scales'][1],
            offsets=EV_RefSB.attributes()['reflectance_offsets'][1])
        blue_reflectance = si2reflectance(
            EV_RefSB[0],
            scales=EV_RefSB.attributes()['reflectance_scales'][0],
            offsets=EV_RefSB.attributes()['reflectance_offsets'][0])
        
        rgb = np.dstack([red_reflectance, green_reflectance, blue_reflectance])
        color_tuples = rgb.reshape(-1, 3)
        
        
        # hdf.select('Band_1KM_Emissive')[:][8:12]
        # EV_Emissive = hdf.select('EV_1KM_Emissive')
        # radiance32 = si2radiance(
        #     EV_Emissive[11],
        #     scales=EV_Emissive.attributes()['radiance_scales'][11],
        #     offsets=EV_Emissive.attributes()['radiance_offsets'][11])
        # radiance31 = si2radiance(
        #     EV_Emissive[10],
        #     scales=EV_Emissive.attributes()['radiance_scales'][10],
        #     offsets=EV_Emissive.attributes()['radiance_offsets'][10])
        # radiance29 = si2radiance(
        #     EV_Emissive[8],
        #     scales=EV_Emissive.attributes()['radiance_scales'][8],
        #     offsets=EV_Emissive.attributes()['radiance_offsets'][8])
        # red_radiance = radiance32 - radiance31
        # green_radiance = radiance31 - radiance29
        # blue_radiance = radiance31
        # red_radiance = np.clip((red_radiance+1)/(1+1), 0, 1)
        # green_radiance = np.clip((green_radiance+0)/(2+0), 0, 1)
        # blue_radiance = np.clip((blue_radiance+0)/(10+0), 0, 1)
        # rgb = np.dstack([red_radiance, green_radiance, blue_radiance])
        # color_tuples = rgb.reshape(-1, 3)
        
        # fig, ax = globe_plot(figsize=np.array([88, 44]) / 2.54, lw=1)
        ax.pcolormesh(lon, lat, np.zeros_like(lat), color=color_tuples,
                      transform=ccrs.PlateCarree())
        # fig.savefig('figures/test.png')

# fig.savefig('figures/test1.png')
fig.savefig('figures/test2.png')



'''
'''
# endregion


# region plot 'MOD02HKM', 'MYD02HKM': Calibrated Radiances

product_sat = {
    'MOD021KM': 'Terra', 'MYD021KM': 'Aqua',
    'MOD02HKM': 'Terra', 'MYD02HKM': 'Aqua',
}

starttime = datetime(2020, 6, 2, 6)
endtime = datetime(2020, 6, 2, 6)
timeseries = pd.date_range(start=starttime, end=endtime, freq='h')
products = ['MOD021KM', 'MYD021KM'] # ['MOD021KM', 'MYD021KM', 'MOD02HKM', 'MYD02HKM']
regions = ['global'] # ['global', 'BARRA-C2']

for itime in timeseries:
    print(f'#-------------------------------- {itime}')
    
    year, month, day, hour = itime.year, itime.month, itime.day, itime.hour
    doy = datetime(year, month, day).timetuple().tm_yday
    
    for iproduct in products:
        print(f'#---------------- {iproduct}')
        fl = sorted(glob.glob(f'scratch/data/obs/MODIS/{iproduct}/{year}/{doy:03d}/*.{hour:02d}??.061.*.hdf'))
        lats, lons, rgbs = get_modis_latlonrgbs(fl)
        
        if lats.shape[0] == 2 * rgbs.shape[0]:
            lats = block_reduce(lats, block_size=(2, 2), func=np.min)
            lons = block_reduce(lons, block_size=(2, 2), func=np.min)
        
        # # trim nan values
        # row_mask = np.all(np.isfinite(lons), axis=1) & np.all(np.isfinite(lats), axis=1)
        # col_mask = np.all(np.isfinite(lons), axis=0) & np.all(np.isfinite(lats), axis=0)
        # lons = lons[:, col_mask][row_mask]
        # lats = lats[:, col_mask][row_mask]
        # rgbs = rgbs[:, col_mask, :][row_mask]
        
        for iregion in regions:
            print(f'#-------- {iregion}')
            
            opng = f'figures/3_satellites/3.2_modis/3.2.0_images/3.2.0.0 {iproduct} {iregion} {str(itime)[:13]} UTC.png'
            label = f'MODIS {product_sat[iproduct]} {iproduct} {str(itime)[:16]} UTC'
            if iregion == 'global':
                fig, ax = globe_plot(
                    figsize=np.array([24, 13]) / 2.54, lw=0.1,
                    projections = ccrs.Robinson(central_longitude=180))
                ax.pcolormesh(lons, lats, rgbs, transform=ccrs.PlateCarree())
                fig.text(0.5, 0.01, label, ha='center', va='bottom')
                fig.subplots_adjust(left=0.01,right=0.99,bottom=0.05,top=0.99)
            elif iregion == 'BARRA-C2':
                min_lon, max_lon, min_lat, max_lat = [110.58, 157.34, -43.69, -7.01]
                mask = (lons>=min_lon) & (lons<=max_lon) & (lats>=min_lat) & (lats<=max_lat)
                rgbscopy = rgbs.copy()
                rgbscopy[np.broadcast_to(~mask[:, :, np.newaxis], rgbs.shape)] = np.nan
                fig, ax = regional_plot(
                    extent=[min_lon, max_lon, min_lat, max_lat],
                    central_longitude=180,
                    figsize = np.array([8.8, 7.4]) / 2.54)
                ax.pcolormesh(lons, lats, rgbscopy, transform=ccrs.PlateCarree())
                fig.text(0.5, 0.01, label, ha='center', va='bottom')
                fig.subplots_adjust(left=0.01,right=0.99,bottom=0.06,top=0.99)
            
            fig.savefig(opng, dpi=1200)



'''
    # not necessary
    lon = lon % 360
    
    # does not work
    import pyproj
    lont, latt = pyproj.transform('epsg:4326', ccrs.Mollweide(central_longitude=180), lon, lat, always_xy=True)
    ax.pcolormesh(lont, latt, rgb, transform=ccrs.Mollweide(central_longitude=180))
    
    # does not work
    mask = lon>180
    rgbcopy = rgb.copy()
    rgbcopy[np.broadcast_to(mask[:, :, np.newaxis], rgb.shape)] = np.nan
    ax.pcolormesh(lon, lat, rgbcopy, transform=ccrs.PlateCarree())
    
    # mask = lat>60 # works
    mask = lon<=0 # does not work with default/'nearest'/'gouraud' shading
    rgbcopy = rgb.copy()
    rgbcopy[np.broadcast_to(mask[:, :, np.newaxis], rgb.shape)] = np.nan
    ax.pcolormesh(lon, lat, rgbcopy, transform=ccrs.PlateCarree())
    
    #---- works ugly
    # , projections = ccrs.PlateCarree(central_longitude=180),
    # not working:
    # projections = ccrs.Mollweide(central_longitude=-180)
    
    #---- not working
    # ax.pcolormesh(lon, lat, np.zeros_like(lat), color=rgb.reshape(-1, 3),
    #               transform=ccrs.PlateCarree())
'''
# endregion


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

