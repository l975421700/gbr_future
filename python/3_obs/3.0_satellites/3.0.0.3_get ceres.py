

# qsub -I -q normal -l walltime=5:00:00,ncpus=1,mem=60GB,jobfs=100MB,storage=gdata/v46+gdata/rt52+gdata/ob53+gdata/zv2+scratch/v46


# region import packages

# data analysis
import xarray as xr
import pickle

# management
import os
import sys  # print(sys.path)
sys.path.append(os.getcwd() + '/code/gbr_future/module')

from calculations import mon_sea_ann

# endregion


# region get ceres data

years = '2001'
yeare = '2023'

# TOA
ceres_ebaf = xr.open_dataset('data/obs/CERES/CERES_EBAF-TOA_Ed4.2.1_Subset_200003-202411.nc').sel(time=slice(years, yeare))
ceres_ebaf = ceres_ebaf.rename({
    'toa_sw_all_mon': 'mtuwswrf',
    'toa_lw_all_mon': 'mtnlwrf',
    'solar_mon': 'mtdwswrf'})
ceres_ebaf['mtuwswrf'] *= (-1)
ceres_ebaf['mtnlwrf'] *= (-1)


# Surface
ceres_ebaf1 = xr.open_dataset('data/obs/CERES/CERES_EBAF_Ed4.2_Subset_200003-202407 (1).nc').sel(time=slice(years, yeare))
ceres_ebaf1 = ceres_ebaf1.rename({
    'sfc_sw_down_all_mon': 'msdwswrf',
    'sfc_sw_up_all_mon': 'msuwswrf',
    'sfc_lw_down_all_mon': 'msdwlwrf',
    'sfc_lw_up_all_mon': 'msuwlwrf',
    'sfc_net_sw_all_mon': 'msnswrf',
    'sfc_net_lw_all_mon': 'msnlwrf',
    # 'sfc_net_tot_all_mon':
})
ceres_ebaf1['msuwswrf'] *= (-1)
ceres_ebaf1['msuwlwrf'] *= (-1)


ceres_ebaf = xr.merge((ceres_ebaf, ceres_ebaf1), compat='override')


for var in ['mtuwswrf', 'mtnlwrf', 'mtdwswrf', 'msdwswrf', 'msuwswrf', 'msdwlwrf', 'msuwlwrf', 'msnswrf', 'msnlwrf']:
    # var = 'mtuwswrf'
    print(f'#-------------------------------- {var}')
    
    ceres_mon = ceres_ebaf[var]
    ceres_mon_alltime = mon_sea_ann(
        var_monthly=ceres_mon, lcopy=False, mm=True, sm=True, am=True)
    
    ofile = f'data/obs/CERES/ceres_mon_alltime_{var}.pkl'
    if os.path.exists(ofile): os.remove(ofile)
    with open(ofile, 'wb') as f:
        pickle.dump(ceres_mon_alltime, f)
    
    del ceres_mon, ceres_mon_alltime



'''
#-------------------------------- check
years = '2001'
yeare = '2023'
# TOA
ceres_ebaf = xr.open_dataset('data/obs/CERES/CERES_EBAF-TOA_Ed4.2.1_Subset_200003-202411.nc').sel(time=slice(years, yeare))
ceres_ebaf = ceres_ebaf.rename({
    'toa_sw_all_mon': 'mtuwswrf',
    'toa_lw_all_mon': 'mtnlwrf',
    'solar_mon': 'mtdwswrf'})
ceres_ebaf['mtuwswrf'] *= (-1)
ceres_ebaf['mtnlwrf'] *= (-1)
# Surface
ceres_ebaf1 = xr.open_dataset('data/obs/CERES/CERES_EBAF_Ed4.2_Subset_200003-202407 (1).nc').sel(time=slice(years, yeare))
ceres_ebaf1 = ceres_ebaf1.rename({
    'sfc_sw_down_all_mon': 'msdwswrf',
    'sfc_sw_up_all_mon': 'msuwswrf',
    'sfc_lw_down_all_mon': 'msdwlwrf',
    'sfc_lw_up_all_mon': 'msuwlwrf',
    'sfc_net_sw_all_mon': 'msnswrf',
    'sfc_net_lw_all_mon': 'msnlwrf',
    # 'sfc_net_tot_all_mon':
})
ceres_ebaf1['msuwswrf'] *= (-1)
ceres_ebaf1['msuwlwrf'] *= (-1)
ceres_ebaf = xr.merge((ceres_ebaf, ceres_ebaf1), compat='override')

ilat = 100
ilon = 100
for var in ['mtuwswrf', 'mtnlwrf', 'mtdwswrf', 'msdwswrf', 'msuwswrf', 'msdwlwrf', 'msuwlwrf', 'msnswrf', 'msnlwrf']:
    # var = 'mtuwswrf'
    print(f'#-------------------------------- {var}')
    with open(f'data/obs/CERES/ceres_mon_alltime_{var}.pkl', 'rb') as f:
        ceres_mon_alltime = pickle.load(f)
    
    print((ceres_mon_alltime['mon'][:, ilat, ilon] == ceres_ebaf[var][:, ilat, ilon]).all().values)


'''
# endregion


# region derive ceres data

for var1, vars in zip(['toa_albedo'], [['mtuwswrf', 'mtdwswrf']]):
    # var1='toa_albedo'; vars=['mtuwswrf', 'mtdwswrf']
    print(f'#-------------------------------- Derive {var1} from {vars}')
    
    ceres_mon_alltime = {}
    for var2 in vars:
        print(f'#---------------- {var2}')
        with open(f'data/obs/CERES/ceres_mon_alltime_{var2}.pkl', 'rb') as f:
            ceres_mon_alltime[var2] = pickle.load(f)
    
    ceres_mon_alltime[var1] = {}
    for ialltime in ceres_mon_alltime[vars[0]].keys():
        print(f'#-------- {ialltime}')
        ceres_mon_alltime[var1][ialltime] = (ceres_mon_alltime[vars[0]][ialltime] / ceres_mon_alltime[vars[1]][ialltime] * (-1)).compute()
    
    ofile = f'data/obs/CERES/ceres_mon_alltime_{var1}.pkl'
    if os.path.exists(ofile): os.remove(ofile)
    with open(ofile, 'wb') as f:
        pickle.dump(ceres_mon_alltime[var1], f)
    
    del ceres_mon_alltime




'''
#-------------------------------- check
ilat = 100
ilon = 100
for var1, vars in zip(['toa_albedo'], [['mtuwswrf', 'mtdwswrf']]):
    # var1='toa_albedo'; vars=['mtuwswrf', 'mtdwswrf']
    print(f'#-------------------------------- Derive {var1} from {vars}')
    
    ceres_mon_alltime = {}
    for var2 in [var1] + vars:
        print(f'#---------------- {var2}')
        with open(f'data/obs/CERES/ceres_mon_alltime_{var2}.pkl', 'rb') as f:
            ceres_mon_alltime[var2] = pickle.load(f)
    
    for ialltime in ceres_mon_alltime[var1].keys():
        print(f'#-------- {ialltime}')
        
        print((ceres_mon_alltime[var1][ialltime][:, ilat, ilon].values == (ceres_mon_alltime[vars[0]][ialltime][:, ilat, ilon] / ceres_mon_alltime[vars[1]][ialltime][:, ilat, ilon] * (-1)).values).all())


'''
# endregion

