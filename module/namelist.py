

import numpy as np


# region names

month_jan = np.array(
    ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
monthini = np.array(
    ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
month_num = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
month_dec = np.array(
    ['Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
     'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov',])
month_dec_num = np.array([12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

seasons = np.array(['DJF', 'MAM', 'JJA', 'SON'])
seasons_last_num = np.array([2, 5, 8, 11])

hours = ['00', '01', '02', '03', '04', '05',
         '06', '07', '08', '09', '10', '11',
         '12', '13', '14', '15', '16', '17',
         '18', '19', '20', '21', '22', '23',
         ]

months = ['01', '02', '03', '04', '05', '06',
          '07', '08', '09', '10', '11', '12']

month_days = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

seconds_per_d = 86400

nci_ds = {
    'era5': '/g/data/rt52/era5/',
    'BARRA-C2': '/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/',
    'ISCCP': '/g/data/ct11/access-nri/replicas/esmvaltool/obsdata-v2/Tier1/ISCCP',
    'AGCD': '/g/data/ct11/access-nri/replicas/esmvaltool/obsdata-v2/Tier2/AGCD',
    'CALIOP': '/g/data/ct11/access-nri/replicas/esmvaltool/obsdata-v2/Tier1/CALIOP',
    'ESACCI-CLOUD': '/g/data/ct11/access-nri/replicas/esmvaltool/obsdata-v2/Tier2/ESACCI-CLOUD',
    'IMERG': '/g/data/ia39/aus-ref-clim-data-nci/frogs/data/1DD_V1/IMERG_V07B_FC',
    'AuRa': '/g/data/rq0',
}

# endregion


# region physics

zerok = 273.15

# endregion


# region plot

panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)',
                '(h)', '(i)', '(j)', '(k)', '(l)', '(m)', '(n)',
                '(o)', '(p)', '(q)', '(r)', '(s)', '(t)', '(u)',
                '(v)', '(w)', '(x)', '(y)', '(z)',
                ]

ds_color = {
    'Himawari': 'black',
    'CERES': 'black',
    'ERA5': 'tab:blue',
    'BARRA-R2': 'tab:orange',
    'BARRA-C2': 'tab:green',
    'BARPA-C': 'tab:purple',
    'C2-RAL3.3': 'tab:red',
    'UM': 'tab:cyan',
}


# endregion


# region cmip6_units

cmip6_units = {
    'tas': 'K',
    'rsut': 'W m-2',
    'rsdt': 'W m-2',
    'rlut': 'W m-2',
    'pr': 'kg m-2 s-1',
    'tos': 'degC',
    'clt': '%',
    'evspsbl': 'kg m-2 s-1',
    'hfls': 'W m-2',
    'hfss': 'W m-2',
    'psl': 'Pa',
    'rlds': 'W m-2',
    'rldscs': 'W m-2',
    'rlus': 'W m-2',
    'rlutcs': 'W m-2',
    'rsds': 'W m-2',
    'rsdscs': 'W m-2',
    'rsus': 'W m-2',
    'rsuscs': 'W m-2',
    'rsutcs': 'W m-2',
}

era5_varlabels = {
    'tp': r'precipitation [$mm \; day^{-1}$]',
    'msl': r'sea level pressure [$hPa$]',
    'sst': r'SST [$°C$]',
    'hcc': r'high cloud cover [$\%$]',
    'mcc': r'middle cloud cover [$\%$]',
    'lcc': r'low cloud cover [$\%$]',
    'tcc': r'total cloud cover [$\%$]',
    'clear': r'clear sky [$\%$]',
    't2m': r'2 $m$ temperature [$°C$]',
    'msnlwrf': r'surface net LW radiation [$W \; m^{-2}$]',
    'msnswrf': r'surface net SW radiation [$W \; m^{-2}$]',
    'mtdwswrf': r'TOA downward SW radiation [$W \; m^{-2}$]',
    'mtnlwrf': r'TOA net LW radiation [$W \; m^{-2}$]',
    'mtnswrf': r'TOA net SW radiation [$W \; m^{-2}$]',
    'msdwlwrf': r'surface downward LW radiation [$W \; m^{-2}$]',
    'msdwswrf': r'surface downward SW radiation [$W \; m^{-2}$]',
    'msdwlwrfcs': r'clear-sky surface downward LW radiation [$W \; m^{-2}$]',
    'msdwswrfcs': r'clear-sky surface downward SW radiation [$W \; m^{-2}$]',
    'msnlwrfcs': r'clear-sky surface net LW radiation [$W \; m^{-2}$]',
    'msnswrfcs': r'clear-sky surface net SW radiation [$W \; m^{-2}$]',
    'mtnlwrfcs': r'clear-sky TOA net LW radiation [$W \; m^{-2}$]',
    'mtnswrfcs': r'clear-sky TOA net SW radiation [$W \; m^{-2}$]',
    'cbh': r'cloud base height [$m$]',
    'tciw': r'total column cloud ice water [$g \; m^{-2}$]',
    'tclw': r'total column cloud liquid water [$g \; m^{-2}$]',
    'e': r'evaporation [$mm \; day^{-1}$]',
    # 'z': r'orography [$m$]',
    'orog': r'orography [$m$]',
    'mslhf': r'surface latent heat flux [$W \; m^{-2}$]',
    'msshf': r'surface sensible heat flux [$W \; m^{-2}$]',
    'tcw': r'total column water [$kg \; m^{-2}$]',
    'tcwv': r'total column water vapour [$kg \; m^{-2}$]',
    'tcsw': r'total column snow water [$g \; m^{-2}$]',
    'tcrw': r'total column rain water [$g \; m^{-2}$]',
    'tcslw': r'total column supercooled liquid water [$g \; m^{-2}$]',
    'si10': r'10 $m$ wind speed [$m\;s^{-1}$]',
    'd2m': r'2 $m$ dew temperature [$°C$]',
    'cp': r'convective precipitation [$mm \; day^{-1}$]',
    'lsp': r'large-scale precipitation [$mm \; day^{-1}$]',
    'deg0l': r'0$°C$ isothermal level [$m$]',
    'mper': r'potential evaporation [$mm \; day^{-1}$]',
    'pev': r'potential evaporation [$mm \; day^{-1}$]',
    'skt': r'skin temperature [$°C$]',
    'u10': r'10 $m$ eastward wind speed [$m\;s^{-1}$]',
    'v10': r'10 $m$ northward wind speed [$m\;s^{-1}$]',
    'u100': r'100 $m$ eastward wind speed [$m\;s^{-1}$]',
    'v100': r'100 $m$ northward wind speed [$m\;s^{-1}$]',
    'msuwlwrf': r'surface upward LW radiation [$W \; m^{-2}$]',
    'msuwswrf': r'surface upward SW radiation [$W \; m^{-2}$]',
    'msuwlwrfcs': r'clear-sky surface upward LW radiation [$W \; m^{-2}$]',
    'msuwswrfcs': r'clear-sky surface upward SW radiation [$W \; m^{-2}$]',
    'msnlwrfcl': r'CRE on surface net LW radiation [$W \; m^{-2}$]',
    'msnswrfcl': r'CRE on surface net SW radiation [$W \; m^{-2}$]',
    'msdwlwrfcl': r'CRE on surface downward LW radiation [$W \; m^{-2}$]',
    'msdwswrfcl': r'CRE on surface downward SW radiation [$W \; m^{-2}$]',
    'msuwlwrfcl': r'CRE on surface upward LW radiation [$W \; m^{-2}$]',
    'msuwswrfcl': r'CRE on surface upward SW radiation [$W \; m^{-2}$]',
    'mtuwswrf': r'TOA upward SW radiation [$W \; m^{-2}$]',
    'mtuwswrfcs': r'clear-sky TOA upward SW radiation [$W \; m^{-2}$]',
    'mtnlwrfcl': r'CRE on TOA net LW radiation [$W \; m^{-2}$]',
    'mtnswrfcl': r'CRE on TOA net SW radiation [$W \; m^{-2}$]',
    'mtuwswrfcl': r'CRE on TOA upward SW radiation [$W \; m^{-2}$]',
    'toa_albedo': r'TOA albedo [$-$]',
    'toa_albedocs': r'clear-sky TOA albedo [$-$]',
    'toa_albedocl': r'CRE on TOA albedo [$-$]',
    'msnrf': r'surface net radiation [$W \; m^{-2}$]',
    'q': r'specific humidity [$g \; kg^{-1}$]',
    'pv': r'potential vorticity [$10^{-6} \; K \; m^{2} \; kg^{-1} \; s^{-1}$]',
    'r': r'relative humidity [$\%$]',
    't': r'temperature [$°C$]',
    'u': r'u [$m\;s^{-1}$]',
    'v': r'v [$m\;s^{-1}$]',
    'w': r'w [$Pa\;s^{-1}$]',
    'z': r'geopotential height [$m$]',
    'q2m': r'2 $m$ specific humidity [$g \; kg^{-1}$]',
    'rh2m': r'2 $m$ relative humidity [$\%$]',
    'S': r'$S$ [-]',
    'D': r'$D$ [-]',
    'S+D': r'$S+D$ [-]',
    'LTS': r'LTS [$K$]',
    'theta': r'potential temperature [$°C$]',
    'theta_e': r'equivalent potential temperature [$°C$]',
    'blh': r'boundary layer height [$m$]',
    'ncloud': r'cloud number concentration [$cm^{-3}$]',
}

era5_varlabels_sim = {
    'q': r'$q$ [$g \; kg^{-1}$]',
    'r': r'$RH$ [$\%$]',
    't': r'$T$ [$°C$]',
    'u': r'u [$m\;s^{-1}$]',
    'v': r'v [$m\;s^{-1}$]',
    'theta': r'$\theta$ [$°C$]',
    'theta_e': r'$\theta_e$ [$°C$]',
}

cmip6_era5_var = {
    'pr': 'tp',
    'clh': 'hcc',
    'clm': 'mcc',
    'cll': 'lcc',
    'clt': 'tcc',
    'evspsbl': 'e',
    'hfls': 'mslhf',
    'hfss': 'msshf',
    'psl': 'msl',
    'rlds': 'msdwlwrf',
    'rldscs': 'msdwlwrfcs',
    'rlus': 'msuwlwrf',
    'rluscs': 'msuwlwrfcs',
    'rlut': 'mtnlwrf',
    'rlutcs': 'mtnlwrfcs',
    'rsds': 'msdwswrf',
    'rsdscs': 'msdwswrfcs',
    'rsdt': 'mtdwswrf',
    'rsus': 'msuwswrf',
    'rsuscs': 'msuwswrfcs',
    'rsut': 'mtuwswrf',
    'rsutcs': 'mtuwswrfcs',
    'sfcWind': 'si10',
    'tas': 't2m',
    'das': 'd2m',
    'tos': 'sst',
    'sst': 'sst',
    'ts': 'skt',
    'evspsblpot': 'pev',
    'hurs': 'rh2m',
    'huss': 'q2m',
    'uas': 'u10',
    'vas': 'v10',
    'rlns': 'msnlwrf',
    'rsns': 'msnswrf',
    'rlnscs': 'msnlwrfcs',
    'rsnscs': 'msnswrfcs',
    'rlnscl': 'msnlwrfcl',
    'rsnscl': 'msnswrfcl',
    'rldscl': 'msdwlwrfcl',
    'rsdscl': 'msdwswrfcl',
    'rluscl': 'msuwlwrfcl',
    'rsuscl': 'msuwswrfcl',
    'rsnt': 'mtnswrf',
    'rsntcs': 'mtnswrfcs',
    'rlutcl': 'mtnlwrfcl',
    'rsntcl': 'mtnswrfcl',
    'rsutcl': 'mtuwswrfcl',
    'clwvi': 'tclw',
    'clivi': 'tciw',
    'prw': 'tcwv',
    'rns': 'msnrf',
    'hus': 'q',
    'hur': 'r',
    'ta': 't',
    'ua': 'u',
    'va': 'v',
    'wap': 'w',
    'zg': 'z',
    'theta': 'theta',
    'theta_e': 'theta_e',
    'blh': 'blh',
    'orog': 'orog',
    'ncloud': 'ncloud',
    # 'wa',
}
# tcw = tcwv + tcsw + tcrw + tclw + tciw
# 'cbh', 'tcw', 'tcsw', 'tcrw', 'tcslw', 'z', 'd2m', 'cp', 'lsp', 'deg0l', 'u100', 'v100'





# endregion
