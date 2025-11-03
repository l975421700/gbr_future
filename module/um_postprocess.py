

# region preprocess_umoutput


def preprocess_umoutput(ds_in):
    if 'T1HR_MN_rad_diag' in list(ds_in.dims):
        ds_in = ds_in.rename({'T1HR_MN_rad_diag': 'T1HR_MN_rad'})
    
    assert (ds_in['T1HR'].values == ds_in['T1HR_MN_rad'].values).all()
    assert (ds_in['T1HR'].values == ds_in['T1HR_MN'].values).all()
    assert (ds_in['grid_latitude_t'].values == ds_in['grid_latitude_cu'].values).all()
    assert (ds_in['grid_longitude_t'].values == ds_in['grid_longitude_cv'].values).all()
    
    time = ds_in['T1HR'].values
    lat = ds_in['grid_latitude_t'].values
    lon = ds_in['grid_longitude_t'].values
    rho80 = ds_in['TH_1_80_eta_rho'].values
    theta80 = ds_in['TH_1_80_eta_theta'].values
    
    ds_in = ds_in.rename_dims({
        'T1HR': 'time',
        'T1HR_MN_rad': 'time',
        'T1HR_MN': 'time',
        'grid_latitude_t': 'lat',
        'grid_longitude_t': 'lon',
        'grid_latitude_cu': 'lat',
        'grid_longitude_cv': 'lon',
        'TH_1_80_eta_rho': 'rho80',
        'TH_1_80_eta_theta': 'theta80',
        })
    
    ds_in = ds_in.squeeze().reset_coords(drop=True)
    ds_in = ds_in.drop_vars([
        'T1HR', 'T1HR_MN_rad', 'T1HR_MN',
        'grid_latitude_t', 'grid_longitude_t',
        'grid_latitude_cu', 'grid_longitude_cv',
        'TH_1_80_eta_rho', 'TH_1_80_eta_theta',
        'rotated_latitude_longitude',
        ] + [var for var in ds_in.data_vars if var.endswith("bounds")])
    
    ds_in = ds_in.assign_coords({
        'time': time,
        'lat': lat,
        'lon': lon,
        'rho80': rho80,
        'theta80': theta80,
        })
    
    ds_in = ds_in.isel(lon=slice(int(len(ds_in.lon)*0.05),
                                 int(len(ds_in.lon)*0.95)),
                       lat=slice(int(len(ds_in.lat)*0.05),
                                 int(len(ds_in.lat)*0.95)))
    
    return(ds_in)


# endregion


# region stash2var, stash2var_gal, stash2var_ral, var2stash_gal, var2stash_ral

stash2var = {
    'STASH_m01s09i203': 'cll',
    'STASH_m01s09i204': 'clm',
    'STASH_m01s09i205': 'clh',
    'STASH_m01s09i216': 'clt',
    'STASH_m01s30i461': 'prw',
    'STASH_m01s00i024': 'ts',
    'STASH_m01s03i236': 'tas',
    'STASH_m01s03i237': 'huss',
    'STASH_m01s03i245': 'hurs',
    'STASH_m01s00i010': 'hus',
    'STASH_m01s16i004': 'ta',
    'STASH_m01s00i002': 'ua',
    'STASH_m01s00i003': 'va',
    'STASH_m01s00i150': 'wa',
    'STASH_m01s30i008': 'wap',
    'STASH_m01s00i004': 'theta',
    'STASH_m01s00i012': 'qcf',
    'STASH_m01s00i254': 'qcl',
    'STASH_m01s16i206': 'qc',
    'STASH_m01s00i272': 'qr',
    'STASH_m01s16i207': 'qt',
    'STASH_m01s00i408': 'pa',
    'STASH_m01s03i234': 'hfls',
    'STASH_m01s03i217': 'hfss',
    'STASH_m01s02i205': 'rlut',
    'STASH_m01s00i238': 'rlds',
    'STASH_m01s01i205': 'rsut',
    'STASH_m01s01i207': 'rsdt',
    'STASH_m01s01i209': 'rsutcs',
    'STASH_m01s01i210': 'rsdscs',
    'STASH_m01s01i235': 'rsds',
    'STASH_m01s02i201': 'rlns',
    'STASH_m01s02i206': 'rlutcs',
    'STASH_m01s02i208': 'rldscs',
    'STASH_m01s16i222': 'psl',
    'STASH_m01s00i025': 'blh',
    'STASH_m01s00i030': 'iland',
    'STASH_m01s00i033': 'orog',
    'STASH_m01s00i239': 'rlu_t_s',
    'STASH_m01s03i250': 'das',
    'STASH_m01s04i224': 'clslw',
    'STASH_m01s20i114': 'CAPE',
    'STASH_m01s30i405': 'clwvi',
    'STASH_m01s30i406': 'clivi',
    'STASH_m01s03i209': 'uas',
    'STASH_m01s03i210': 'vas',
    'STASH_m01s03i230': 'sfcWind',
    'STASH_m01s16i201': 'zg',
    'STASH_m01s20i115': 'CIN',
    'STASH_m01s00i265': 'ACF',
    'STASH_m01s00i266': 'BCF',
    'STASH_m01s02i261': 'TCF',
    }

stash2var_gal = stash2var | {
    'STASH_m01s04i210': 'ncloud',
    'STASH_m01s05i216': 'pr',
    'STASH_m01s05i215': 'snow',
    'STASH_m01s05i214': 'rain',
    }

stash2var_ral = stash2var | {
    'STASH_m01s00i271': 'qs',
    'STASH_m01s00i273': 'qg',
    'STASH_m01s00i075': 'ncloud',
    'STASH_m01s00i076': 'nrain',
    'STASH_m01s00i078': 'nice',
    'STASH_m01s00i079': 'nsnow',
    'STASH_m01s00i081': 'ngraupel',
    'STASH_m01s04i203': 'rain',
    'STASH_m01s04i204': 'snow',
    'STASH_m01s04i212': 'graupel',
    'STASH_m01s04i304': 'snow_graupel',
    }

var2stash = {stash2var[ikey]: ikey  for ikey in stash2var.keys()}
var2stash_gal = {stash2var_gal[ikey]: ikey  for ikey in stash2var_gal.keys()}
var2stash_ral = {stash2var_ral[ikey]: ikey  for ikey in stash2var_ral.keys()}


'''
    # 'STASH_m01s00i391': 'mv',
    # 'STASH_m01s00i392': 'mcl',
    # 'STASH_m01s00i393': 'mcf',
    # 'STASH_m01s00i394': 'mr',
    # 'STASH_m01s00i395': 'mg',
    # 'STASH_m01s00i396': 'mcf2',
    # 'STASH_m01s03i513': 'blendingw',
    # 'STASH_m01s04i118': 'radar_reflectivity',
'''
# endregion


# region suite_res

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
    'u-dr789': ['d11km', 'd4p4km'],
    'u-dr922': ['d11km', 'd4p4km'],
    'u-ds714': ['d11km', 'd4p4km'],
    'u-ds718': ['d11km', 'd4p4km'],
    'u-ds717': ['d11km', 'd4p4km'],
    'u-ds719': ['d11km', 'd4p4km'],
    'u-ds722': ['d11km', 'd4p4km'],
    'u-ds724': ['d11km', 'd2p2km'],
    'u-ds726': ['d11km', 'd1p1km'],
    'u-ds728': ['d11km', 'd4p4km'],
    'u-ds730': ['d11km', 'd2p2km'],
    'u-ds732': ['d11km', 'd1p1km'],
    'u-ds919': ['d11km', 'd1p1km'],
    'u-ds920': ['d11km', 'd1p1km'],
    'u-ds921': ['d11km', 'd4p4km'],
    'u-ds922': ['d11km', 'd4p4km'],
    'u-dt038': ['d11km', 'd4p4km'],
    'u-dt039': ['d11km', 'd4p4km'],
    'u-dt040': ['d11km', 'd4p4km'],
    'u-dt020': ['d11km', 'd4p4km'],
    'u-dt041': ['d11km', 'd4p4kml'],
    'u-dt042': ['d11km', 'd4p4km'],
}

suite_label = {
    'u-dq700': r'$control$',
    'u-dq788': r'$NE4p4$',
    'u-dq799': r'$NE1p1$',
    'u-dq911': r'$NE2p2$',
    'u-dq912': r'$LD4p4$',
    'u-dq987': r'$CN150$',
    'u-dr040': r'$CN100$',
    'u-dr041': r'$CN50$',
    'u-dr091': r'$SA1p1$',
    'u-dr093': r'$SA2p2$',
    'u-dr095': r'$SA4p4$',
    'u-dr105': r'$Lev120$',
    'u-dr107': r'$Lev140$',
    'u-dr108': r'$RAL2M$',
    'u-dr109': r'$RAL2T$',
    'u-dr144': r'$NE4p4\_CN50$',
    'u-dr145': r'$NE1p1\_CN50$',
    'u-dr146': r'$NE2p2\_CN50$',
    'u-dr147': r'$SA1p1\_CN50$',
    'u-dr148': r'$SA2p2\_CN50$',
    'u-dr149': r'$SA4p4\_CN50$',
    'u-dr789': r'$cl\_inhomo0p5$',
    'u-dr922': r'$cl\_inhomo0p3$',
    'u-ds714': r'$control$',
    'u-ds718': r'$CN100$',
    'u-ds717': r'$CN50$',
    'u-ds719': r'$shortTS$',
    'u-ds722': r'$NE4p4s$',
    'u-ds724': r'$NE2p2s$',
    'u-ds726': r'$NE1p1s$',
    'u-ds728': r'$SA4p4s$',
    'u-ds730': r'$SA2p2s$',
    'u-ds732': r'$SA1p1s$',
    'u-ds919': r'$NE1p1\_CN50$',
    'u-ds920': r'$SA1p1\_CN50$',
    'u-ds921': r'$5dSpin$',
    'u-ds922': r'$10dSpin$',
    'u-dt038': r'$12hSpin$',
    'u-dt039': r'$6hSpin$',
    'u-dt040': r'$0hSpin$',
    'u-dt020': r'$CN5$',
    'u-dt041': r'$LD4p4$',
    'u-dt042': r'$CN500$',
}

# endregion


# region interp_to_pressure_levels

def interp_to_pressure_levels(var, pressure, plevs_hpa):
    import numpy as np
    import xarray as xr
    
    # Core interpolation function
    def _interp_column(p_target, p, v):
        # ensure ascending order for np.interp
        order = np.argsort(p)
        p_sorted = p[order]
        v_sorted = v[order]
        return np.interp(np.log(p_target), np.log(p_sorted), v_sorted, left=np.nan, right=np.nan)
    
    # Vectorized interpolation over latitude (and optionally time)
    result = xr.apply_ufunc(
        _interp_column,
        xr.DataArray(plevs_hpa, dims=['pressure']),
        pressure,
        var,
        input_core_dims=[['pressure'], ['theta80'], ['theta80']],
        output_core_dims=[['pressure']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[var.dtype],
    )
    
    # Assign coordinates and attributes
    result = result.assign_coords(pressure=('pressure', plevs_hpa))
    
    return result

# endregion
