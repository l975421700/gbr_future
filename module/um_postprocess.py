

# region amstash2var, amvar2stash, preprocess_amoutput

amstash2var = {
    # monthly
    'fld_s00i002': 'ua', #'U COMPNT OF WIND AFTER TIMESTEP',
    'fld_s00i003': 'va', #'V COMPNT OF WIND AFTER TIMESTEP',
    'fld_s00i004': 'theta', #'THETA AFTER TIMESTEP',
    'fld_s00i010': 'hus', #'SPECIFIC HUMIDITY AFTER TIMESTEP',
    'fld_s00i012': 'qcf', #'QCF AFTER TIMESTEP',
    'fld_s00i031': 'seaice', #'FRAC OF SEA ICE IN SEA AFTER TSTEP',
    'fld_s00i150': 'wa', #'W COMPNT OF WIND AFTER TIMESTEP',
    'fld_s00i254': 'qcl', #'QCL AFTER TIMESTEP',
    'fld_s00i272': 'qr', #'RAIN AFTER TIMESTEP',
    'fld_s00i408': 'pa', #'PRESSURE AT THETA LEVELS AFTER TS',
    'fld_s00i507': 'sst', #'OPEN SEA SURFACE TEMP AFTER TIMESTEP',
    'fld_s01i217': 'rsu', #'UPWARD SW FLUX ON LEVELS',
    'fld_s01i218': 'rsd', #'DOWNWARD SW FLUX ON LEVELS',
    'fld_s01i219': 'rsucs', #'CLEAR-SKY UPWARD SW FLUX ON LEVELS',
    'fld_s01i220': 'rsdcs', #'CLEAR-SKY DOWNWARD SW FLUX ON LEVELS',
    'fld_s01i237': 'rsn_trop', #'NET DOWNWARD SW FLUX AT THE TROP.',
    'fld_s01i238': 'rsu_trop', #'UPWARD SW FLUX AT THE TROPOPAUSE',
    'fld_s02i217': 'rlu', #'UPWARD LW FLUX ON LEVELS',
    'fld_s02i218': 'rld', #'DOWNWARD LW FLUX ON LEVELS',
    'fld_s02i219': 'rlucs', #'CLEAR-SKY UPWARD LW FLUX ON LEVELS',
    'fld_s02i220': 'rldcs', #'CLEAR-SKY DOWNWARD LW FLUX ON LEVELS',
    'fld_s02i237': 'rln_trop', #'NET DOWNWARD LW FLUX AT THE TROP.',
    'fld_s02i238': 'rld_trop', #'DOWNWARD LW FLUX AT THE TROPOPAUSE',
    'fld_s02i261': 'TCF', #'TOTAL CLOUD AMOUNT ON LEVELS',
    'fld_s02i261_0': 'TCF_hourly', #'TOTAL CLOUD AMOUNT ON LEVELS',
    'fld_s02i325': 'cosp_mask_cf', #'COSP: MASK FOR CALIPSO CF ON 40 LVLS',
    'fld_s02i327': 'cosp_mask359', #'COSP: MASK FOR 2.359',
    'fld_s02i335': 'cosp_isccp_Tb', #'COSP: ISCCP ALL-SKY 10.5 MICRON Tb',
    'fld_s02i336': 'cosp_isccp_Tbcs', #'COSP: ISCCP CLEAR-SKY 10.5 MICRON Tb',
    'fld_s02i337': 'cosp_isccp_ctp_tau', #'COSP: ISCCP CTP-TAU HISTOGRAM',
    'fld_s02i359': 'cosp_cc_cf', #'COSP: CALIPSO/CLOUDSAT CLOUD 40 LEV',
    'fld_s02i359_0': 'cosp_cc_cf_hourly', #'COSP: CALIPSO/CLOUDSAT CLOUD 40 LEV',
    'fld_s02i371': 'cosp_c_ca', #'COSP: CALIPSO CLOUD AREA 40 CSAT LVS',
    'fld_s02i374': 'cosp_c_c', #'COSP: CALIPSO-ONLY 3D CLOUD 40 LVLS',
    'fld_s02i473': 'cosp_c_cfl', #'COSP: CALIPSO CF 40 LVLS (LIQ)',
    'fld_s02i474': 'cosp_c_cfi', #'COSP: CALIPSO CF 40 LVLS (ICE)',
    'fld_s02i475': 'cosp_c_cfu', #'COSP: CALIPSO CF 40 LVLS (UNDET)',
    'fld_s03i261': 'GPP', #'GROSS PRIMARY PRODUCTIVITY KG C/M2/S',
    'fld_s03i662': 'PNPP', #'POTENTIAL NET PRIM PRODUC KG C/M2/S',
    'fld_s04i224': 'clslw', #'supercooled_liquid_water_content',
    'fld_s04i259': 'rain_evap', #'rate_of_loss_of_rain_mass_due_to_evaporation',
    'fld_s05i250': 'updraught_mf', #'UPDRAUGHT MASS FLUX (Pa/s)',
    'fld_s05i251': 'downdraught_mf', #'DOWNDRAUGHT MASS FLUX (PA/S)',
    'fld_s05i320': 'deepc_mf', #'MASS FLUX DEEP CONVECTION',
    'fld_s05i321': 'congestusc_mf', #'MASS FLUX CONGESTUS CONVECTION',
    'fld_s05i322': 'shallowc_mf', #'MASS FLUX SHALLOW CONVECTION',
    'fld_s05i323': 'midc_mf', #'MASS FLUX MID-LEVEL CONVECTION',
    'fld_s16i004': 'ta', #'TEMPERATURE ON THETA LEVELS',
    'fld_s16i201': 'zg', #'GEOPOTENTIAL HEIGHT ON THETA LEVELS',
    'fld_s16i206': 'qc', #'CLOUD WATER CONTENT (qc)',
    'fld_s16i206_0': 'qc_hourly', #'CLOUD WATER CONTENT (qc)',
    'fld_s16i207': 'qt', #'TOTAL SPECIFIC HUMIDITY (qT)',
    'fld_s20i033': 'zg_freeze', #'FREEZING LEVEL GEOPOTENTIAL HEIGHT',
    'fld_s20i034': 'p_freeze', #'FREEZING LEVEL PRESSURE',
    'fld_s20i084': 'p_trop', #'TROPOPAUSE PRESSURE',
    'fld_s20i085': 't_trop', #'TROPOPAUSE TEMPERATURE',
    'fld_s20i086': 'h_trop', #'TROPOPAUSE HEIGHT',
    'fld_s30i008': 'wap', #'OMEGA ON THETA LEVELS (C GRID)',
    'fld_s30i008_0': 'wap_hourly', #'OMEGA ON THETA LEVELS (C GRID)',
    'fld_s30i113': 'hur', #'RH on model levels at end timestep',
    'fld_s30i113_0': 'hur_hourly', #'RH on model levels at end timestep',
    'fld_s30i315': 'meridional_hf', #'MERIDIONAL HEAT FLUX',
    'fld_s30i316': 'meridional_mf', #'MERIDIONAL MOMENTUM FLUX',
    'fld_s34i071': 'DMS', #'DMS MASS MIXING RATIO AFTER TIMESTEP',
    
    # hourly
    'fld_s00i024': 'ts', #'SURFACE TEMPERATURE AFTER TIMESTEP',
    'fld_s00i025': 'blh', #'BOUNDARY LAYER DEPTH AFTER TIMESTEP',
    'fld_s00i238': 'rlds', #'SURFACE DOWNWARD LW RADIATION   W/M2',
    'fld_s00i239': 'rlu_t_s', #'TOA - SURF UPWARD LW RADIATION  W/M2',
    'fld_s00i409': 'ps', #'SURFACE PRESSURE AFTER TIMESTEP',
    'fld_s01i202': 'rsns', #'NET DOWN SURFACE SW FLUX: CORRECTED',
    'fld_s01i205': 'rsut', #'OUTGOING SW RAD FLUX (TOA):CORRECTED',
    'fld_s01i207': 'rsdt', #'INCOMING SW RAD FLUX (TOA): ALL TSS',
    'fld_s01i209': 'rsutcs', #'CLEAR-SKY (II) UPWARD SW FLUX (TOA)',
    'fld_s01i210': 'rsdscs', #'CLEAR-SKY (II) DOWN SURFACE SW FLUX',
    'fld_s01i211': 'rsuscs', #'CLEAR-SKY (II) UP SURFACE SW FLUX',
    'fld_s01i215': 'rss_dir', #'DIRECT SURFACE SW FLUX : CORRECTED',
    'fld_s01i216': 'rss_dif', #'DIFFUSE SURFACE SW FLUX : CORRECTED',
    'fld_s01i235': 'rsds', #'TOTAL DOWNWARD SURFACE SW FLUX',
    'fld_s02i201': 'rlns', #'NET DOWN SURFACE LW RAD FLUX',
    'fld_s02i205': 'rlut', #'OUTGOING LW RAD FLUX (TOA)',
    'fld_s02i206': 'rlutcs', #'CLEAR-SKY (II) UPWARD LW FLUX (TOA)',
    # 'fld_s02i207': 'rlds', #'DOWNWARD LW RAD FLUX: SURFACE',
    'fld_s02i208': 'rldscs', #'CLEAR-SKY (II) DOWN SURFACE LW FLUX',
    'fld_s02i321': 'cosp_mask_cll', #'COSP: MASK FOR CALIPSO LOW-LEVEL CF',
    'fld_s02i322': 'cosp_mask_clm', #'COSP: MASK FOR CALIPSO MID-LEVEL CF',
    'fld_s02i323': 'cosp_mask_clh', #'COSP: MASK FOR CALIPSO HIGH-LEVEL CF',
    'fld_s02i324': 'cosp_mask_clt', #'COSP: MASK FOR CALIPSO TOTAL CF',
    'fld_s02i330': 'cosp_isccp_weight', #'COSP: ISCCP/MISR/MODIS CLOUD WEIGHTS',
    'fld_s02i331': 'cosp_isccp_albedo', #'COSP: ISCCP WEIGHTED CLOUD ALBEDO',
    'fld_s02i332': 'cosp_isccp_tau', #'COSP: ISCCP WEIGHTED CLOUD TAU',
    'fld_s02i333': 'cosp_isccp_ctp', #'COSP: ISCCP WEIGHTED CLOUD TOP PRES.',
    'fld_s02i334': 'cosp_isccp_tcc', #'COSP: ISCCP TOTAL CLOUD AREA',
    'fld_s02i344': 'cosp_c_lcc', #'COSP: CALIPSO LOW-LEVEL CLOUD',
    'fld_s02i345': 'cosp_c_mcc', #'COSP: CALIPSO MID-LEVEL CLOUD',
    'fld_s02i346': 'cosp_c_hcc', #'COSP: CALIPSO HIGH-LEVEL CLOUD',
    'fld_s02i347': 'cosp_c_tcc', #'COSP: CALIPSO TOTAL CLOUD FRACTION',
    'fld_s03i209': 'uas', #'10 METRE WIND U-COMP',
    'fld_s03i210': 'vas', #'10 METRE WIND V-COMP',
    'fld_s03i217': 'hfss', #'SURFACE SENSIBLE HEAT FLUX     W/M2',
    'fld_s03i230': 'sfcWind', #'10 METRE WIND SPEED ON C GRID',
    'fld_s03i234': 'hfls', #'SURFACE LATENT HEAT FLUX        W/M2',
    'fld_s03i236': 'tas', #'TEMPERATURE AT 1.5M',
    'fld_s03i237': 'huss', #'SPECIFIC HUMIDITY  AT 1.5M',
    'fld_s03i245': 'hurs', #'RELATIVE HUMIDITY AT 1.5M',
    'fld_s03i248': 'fog2m', #'FOG FRACTION AT 1.5 M',
    'fld_s03i250': 'das', #'DEWPOINT AT 1.5M (K)',
    'fld_s03i255': 'qt2m', #'QT AT 1.5M',
    'fld_s03i258': 'hfms', #'SURFACE SNOWMELT HEAT FLUX      W/M2',
    'fld_s03i356': 'mlh', #'HEIGHT OF SURFACE MIXED LAYR TOP (M)',
    'fld_s03i362': 'mlentrain', #'ENTRAINMENT RATE FOR SURFACE ML',
    'fld_s03i363': 'blentrain', #'ENTRAINMENT RATE FOR BOUNDARY LAYER',
    'fld_s03i463': 'wind_gust', #'WIND GUST',
    'fld_s04i203': 'lsrf', #'LARGE SCALE RAINFALL RATE    KG/M2/S',
    'fld_s04i204': 'lssf', #'LARGE SCALE SNOWFALL RATE    KG/M2/S',
    'fld_s05i205': 'crf', #'CONVECTIVE RAINFALL RATE     KG/M2/S',
    'fld_s05i206': 'csf', #'CONVECTIVE SNOWFALL RATE     KG/M2/S',
    'fld_s05i214': 'rain', #'TOTAL RAINFALL RATE: LS+CONV KG/M2/S',
    'fld_s05i215': 'snow', #'TOTAL SNOWFALL RATE: LS+CONV KG/M2/S',
    'fld_s05i216': 'pr', #'TOTAL PRECIPITATION RATE     KG/M2/S',
    'fld_s05i277': 'deep_pr', #'deep convective precipitation rate',
    'fld_s09i202': 'clvl', #'VERY LOW CLOUD AMOUNT',
    'fld_s09i203': 'cll', #'LOW CLOUD AMOUNT',
    'fld_s09i204': 'clm', #'MEDIUM CLOUD AMOUNT',
    'fld_s09i205': 'clh', #'HIGH CLOUD AMOUNT',
    'fld_s09i216': 'clt', #'cloud_area_fraction_assuming_random_overlap',
    'fld_s16i222': 'psl', #'PRESSURE AT MEAN SEA LEVEL',
    'fld_s20i114': 'CAPE', #'UNKNOWN VARIABLE',
    'fld_s20i115': 'CIN', #'UNKNOWN VARIABLE',
    'fld_s30i403': 'dmvi', #'TOTAL COLUMN DRY MASS  RHO GRID',
    'fld_s30i404': 'wmvi', #'TOTAL COLUMN WET MASS  RHO GRID',
    'fld_s30i405': 'clwvi', #'TOTAL COLUMN QCL  RHO GRID',
    'fld_s30i406': 'clivi', #'TOTAL COLUMN QCF  RHO GRID',
    'fld_s30i461': 'prw', #'TOTAL COLUMN Q (WATER VAPOUR PATH)',
    
    # daily
    'fld_s04i210': 'ncloud', #'cloud_drop_number_concentration_where_cloud_is_present',
}
amvar2stash = {amstash2var[ikey]: ikey  for ikey in amstash2var.keys()}

def preprocess_amoutput(ds_in):
    
    if 'model_rho_level_number' in ds_in.dims:
        ds_in = ds_in.rename({'model_rho_level_number': 'rho85'})
    if 'model_theta_level_number' in ds_in.dims:
        ds_in = ds_in.rename({'model_theta_level_number': 'theta85'})
    if 'dim0' in ds_in.dims:
        ds_in = ds_in.rename({'dim0': 'hour'})
    
    ds_in = ds_in.reset_coords(drop=True)
    ds_in = ds_in.drop_vars([ivar for ivar in ds_in.data_vars if not ivar.startswith('fld_')])
    
    for ivar in ds_in.data_vars:
        # ivar = 'fld_s00i025'
        if ivar in amstash2var.keys():
            if ('hour' in ds_in[ivar].dims) & (not 'time' in ds_in[ivar].dims):
                ds_in[ivar] = ds_in[ivar].expand_dims(time=ds_in.time)
            if 'time_0' in ds_in[ivar].dims:
                ds_in[ivar] = ds_in[ivar].assign_coords(time=('time_0', ds_in.time.values)).swap_dims({'time_0': 'time'})
            ds_in = ds_in.rename({ivar: amstash2var[ivar]})
        else:
            # print(f'!!!! Warning: {ivar} not in amstash2var')
            ds_in = ds_in.drop_vars([ivar])
    
    if 'time_0' in ds_in.dims:
        ds_in = ds_in.drop_dims('time_0')
    
    return(ds_in)

amvargroups = {
    'a': [  # monthly
        'ua', 'va', 'theta', 'hus', 'qcf', 'seaice', 'wa', 'qcl', 'qr', 'pa',
        'sst', 'rsu', 'rsd', 'rsucs', 'rsdcs', 'rsn_trop', 'rsu_trop',
        'rlu', 'rld', 'rlucs', 'rldcs', 'rln_trop', 'rld_trop',
        'TCF', 'TCF_hourly',
        'cosp_mask_cf', 'cosp_mask359', 'cosp_isccp_Tb', 'cosp_isccp_Tbcs',
        'cosp_isccp_ctp_tau', 'cosp_cc_cf', 'cosp_cc_cf_hourly',
        'cosp_c_ca', 'cosp_c_c', 'cosp_c_cfl', 'cosp_c_cfi', 'cosp_c_cfu',
        'GPP', 'PNPP', 'clslw', 'rain_evap',
        'updraught_mf', 'downdraught_mf', 'deepc_mf', 'congestusc_mf',
        'shallowc_mf', 'midc_mf',
        'ta', 'zg', 'qc', 'qc_hourly', 'qt',
        'zg_freeze', 'p_freeze', 'p_trop', 't_trop', 'h_trop',
        'wap', 'wap_hourly', 'hur', 'hur_hourly',
        'meridional_hf', 'meridional_mf',
        'DMS'
    ],
    
    'b': [  # hourly
        'ts', 'blh', 'rlds', 'rlu_t_s', 'ps',
        'rsns', 'rsut', 'rsdt', 'rsutcs', 'rsdscs', 'rsuscs',
        'rss_dir', 'rss_dif', 'rsds',
        'rlns', 'rlut', 'rlutcs', 'rldscs',
        'cosp_mask_cll', 'cosp_mask_clm', 'cosp_mask_clh', 'cosp_mask_clt',
        'cosp_isccp_weight', 'cosp_isccp_albedo', 'cosp_isccp_tau',
        'cosp_isccp_ctp', 'cosp_isccp_tcc',
        'cosp_c_lcc', 'cosp_c_mcc', 'cosp_c_hcc', 'cosp_c_tcc',
        'uas', 'vas', 'hfss', 'sfcWind', 'hfls',
        'tas', 'huss', 'hurs', 'fog2m', 'das', 'qt2m',
        'hfms', 'mlh', 'mlentrain', 'blentrain', 'wind_gust',
        'lsrf', 'lssf', 'crf', 'csf', 'rain', 'snow', 'pr', 'deep_pr',
        'clvl', 'cll', 'clm', 'clh', 'clt',
        'psl', 'CAPE', 'CIN',
        'dmvi', 'wmvi', 'clwvi', 'clivi', 'prw'
    ],

    'c': [  # daily
        'ncloud'
    ]
}

# endregion


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

am3_label = {
    'access-am3-configs': r'$control$',
    'am3-plus4k': r'$SST{+}4K$',
    
    # obs
    'CERES': 'CERES',
    'ERA5': 'ERA5',
    'CM SAF': 'CM SAF',
    'Himawari': 'Himawari',
    'IMERG': 'IMERG',
    'OAFlux': 'OAFlux',
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
