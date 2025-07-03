

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
    'STASH_m01s00i391': 'mv',
    'STASH_m01s00i392': 'mcl',
    'STASH_m01s00i393': 'mcf',
    'STASH_m01s00i394': 'mr',
    'STASH_m01s00i395': 'mg',
    'STASH_m01s00i396': 'mcf2',
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
    'STASH_m01s03i513': 'blendingw',
    'STASH_m01s04i118': 'radar_reflectivity',
    'STASH_m01s04i224': 'clslw',
    'STASH_m01s20i114': 'CAPE',
    'STASH_m01s30i405': 'clwvi',
    'STASH_m01s30i406': 'clivi',
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

# endregion
