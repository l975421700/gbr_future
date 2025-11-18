

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
    'ISCCP': '/g/data/ct11/access-nri/replicas/esmvaltool/obsdata-v2/Tier1/ISCCP',
    'AGCD': '/g/data/ct11/access-nri/replicas/esmvaltool/obsdata-v2/Tier2/AGCD',
    'CALIOP': '/g/data/ct11/access-nri/replicas/esmvaltool/obsdata-v2/Tier1/CALIOP',
    'ESACCI-CLOUD': '/g/data/ct11/access-nri/replicas/esmvaltool/obsdata-v2/Tier2/ESACCI-CLOUD',
    'AuRa': '/g/data/rq0',
    'BARPA-C': '/g/data/py18/BARPA/output/CMIP6/DD/AUST-04/BOM/ERA5/evaluation/r1i1p1f1/BARPA-C/v1-r1',
}

# endregion


# region physics

zerok = 273.15

# endregion


# region plot

ds_color = {
    'Himawari': 'black',
    'Radiosonde': 'black',
    'CERES': 'black',
    'ERA5': 'tab:blue',
    'BARRA-R2': 'tab:orange',
    'BARRA-C2': 'tab:green',
    'BARPA-R': 'tab:red',
    'BARPA-C': 'tab:purple',
    'C2-RAL3.3': 'tab:red',
    'UM': 'tab:cyan',
    r'$control$': 'tab:cyan',
    r'$CN50$': 'tab:blue',
    r'$NE4p4s$': 'tab:red',
    r'$NE1p1s$': 'tab:orange',
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
    'hcc': r'HCC [$\%$]',
    'mcc': r'MCC [$\%$]',
    'lcc': r'LCC [$\%$]',
    'cll_mol': r'LCC_MOL [$\%$]',
    'tcc': r'TCC [$\%$]',
    'ACF': r'ACF [$\%$]',
    'BCF': r'BCF [$\%$]',
    'TCF': r'TCF [$\%$]',
    'clear': r'clear sky [$\%$]',
    't2m': r'$t2m$ [$°C$]',
    'msnlwrf': r'surface net LW radiation [$W \; m^{-2}$]',
    'msnswrf': r'surface net SW radiation [$W \; m^{-2}$]',
    'mtdwswrf': r'TOA downward SW radiation [$W \; m^{-2}$]',
    'mtnlwrf': r'OLR [$W \; m^{-2}$]',
    'mtnswrf': r'TOA net SW radiation [$W \; m^{-2}$]',
    'msdwlwrf': r'surface downward LW radiation [$W \; m^{-2}$]',
    'msdwswrf': r'surface downward SW radiation [$W \; m^{-2}$]',
    'msdwlwrfcs': r'clear-sky surface downward LW radiation [$W \; m^{-2}$]',
    'msdwswrfcs': r'clear-sky surface downward SW radiation [$W \; m^{-2}$]',
    'msnlwrfcs': r'clear-sky surface net LW radiation [$W \; m^{-2}$]',
    'msnswrfcs': r'clear-sky surface net SW radiation [$W \; m^{-2}$]',
    'mtnlwrfcs': r'clear-sky OLR [$W \; m^{-2}$]',
    'mtnswrfcs': r'clear-sky TOA net SW radiation [$W \; m^{-2}$]',
    'cbh': r'cloud base height [$m$]',
    'tciw': r'IWP [$g \; m^{-2}$]',
    'tclw': r'LWP [$g \; m^{-2}$]',
    'cwp': r'CWP [$g \; m^{-2}$]',
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
    'si10': r'$vel10m$ [$m\;s^{-1}$]',
    'd2m': r'$d2m$ [$°C$]',
    'cp': r'convective precipitation [$mm \; day^{-1}$]',
    'lsp': r'large-scale precipitation [$mm \; day^{-1}$]',
    'deg0l': r'0$°C$ isothermal level [$m$]',
    'mper': r'potential evaporation [$mm \; day^{-1}$]',
    'pev': r'potential evaporation [$mm \; day^{-1}$]',
    'skt': r'skin temperature [$°C$]',
    'u10': r'$u10m$ [$m\;s^{-1}$]',
    'v10': r'$v10m$ [$m\;s^{-1}$]',
    'u100': r'$u100m$ [$m\;s^{-1}$]',
    'v100': r'$v100m$ [$m\;s^{-1}$]',
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
    'mtuwswrf': r'OSR [$W \; m^{-2}$]',
    'mtuwswrfcs': r'clear-sky OSR [$W \; m^{-2}$]',
    'mtnlwrfcl': r'CRE on OLR [$W \; m^{-2}$]',
    'mtnswrfcl': r'CRE on TOA net SW radiation [$W \; m^{-2}$]',
    'mtuwswrfcl': r'CRE on OSR [$W \; m^{-2}$]',
    'toa_albedo': r'TOA albedo [$-$]',
    'toa_albedocs': r'clear-sky TOA albedo [$-$]',
    'toa_albedocl': r'CRE on TOA albedo [$-$]',
    'msnrf': r'surface net radiation [$W \; m^{-2}$]',
    'q': r'$q$ [$g \; kg^{-1}$]',
    'ciwc': r'$IWC$ [$g \; kg^{-1}$]',
    'clwc': r'$LWC$ [$g \; kg^{-1}$]',
    'crwc': r'$RWC$ [$g \; kg^{-1}$]',
    'cswc': r'$SWC$ [$g \; kg^{-1}$]',
    'qc': r'$CWC$ [$g \; kg^{-1}$]',
    'qt': r'$qt$ [$g \; kg^{-1}$]',
    'clslw': r'supercooled liquid water [$g \; kg^{-1}$]',
    'qg': r'$GWC$ [$g \; kg^{-1}$]',
    'pv': r'$pv$ [$10^{-6} \; K \; m^{2} \; kg^{-1} \; s^{-1}$]',
    'r': r'$RH$ [$\%$]',
    't': r'$T$ [$°C$]',
    'u': r'$u$ [$m\;s^{-1}$]',
    'v': r'$v$ [$m\;s^{-1}$]',
    'w': r'$w$ [$Pa\;s^{-1}$]',
    'z': r'$z$ [$m$]',
    'q2m': r'$q2m$ [$g \; kg^{-1}$]',
    'rh2m': r'$rh2m$ [$\%$]',
    'S': r'$S$ [-]',
    'D': r'$D$ [-]',
    'S+D': r'$S+D$ [-]',
    'LTS': r'LTS [$K$]',
    'theta': r'$\theta$ [$°C$]',
    'theta_e': r'$\theta_e$ [$°C$]',
    'blh': r'BLH [$m$]',
    'ncloud': r'CDNC [$cm^{-3}$]',
    'CDNC': r'CDNC [$cm^{-3}$]',
    'inversionh': r'Inversion height [$m$]',
    'LCL': r'LCL [$m$]',
    'EIS': r'EIS [$K$]',
    'ECTEI': r'ECTEI [$K$]',
    'COT': r'COT [-]',
    'clphase': r'Cloud Top Phase [-]',
    'Reff': r'Cloud Top $r_{eff}$ [$\mu m$]',
    'CTH': r'CTH [$km$]',
    'CTP': r'CTP [$hPa$]',
    'CTT': r'CTT [$K$]',
}


cmip6_era5_var = {
    'pr': 'tp',
    'clh': 'hcc',
    'clm': 'mcc',
    'cll': 'lcc',
    'cll_mol': 'cll_mol',
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
    'zmla': 'blh',
    'orog': 'orog',
    'ncloud': 'ncloud',
    'CDNC': 'CDNC',
    'ps': 'sp',
    'inversionh': 'inversionh',
    'LCL': 'LCL',
    'LTS': 'LTS',
    'EIS': 'EIS',
    'ECTEI': 'ECTEI',
    'cwp': 'cwp',
    'qcf': 'ciwc',
    'qcl': 'clwc',
    'qr': 'crwc',
    'qs': 'cswc',
    'qc': 'qc',
    'qt': 'qt',
    'clslw': 'clslw',
    'qg': 'qg',
    'ACF': 'ACF',
    'BCF': 'BCF',
    'TCF': 'TCF',
    'COT': 'COT',
    'clphase': 'clphase',
    'Reff': 'Reff',
    'CTH': 'CTH',
    'CTP': 'CTP',
    'CTT': 'CTT',
    # 'wa',
}
# tcw = tcwv + tcsw + tcrw + tclw + tciw
# 'cbh', 'tcw', 'tcsw', 'tcrw', 'tcslw', 'z', 'd2m', 'cp', 'lsp', 'deg0l', 'u100', 'v100'





# endregion
