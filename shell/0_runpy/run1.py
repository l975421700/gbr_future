

# qsub -I -q normal -P nf33 -l walltime=3:00:00,ncpus=1,mem=20GB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22


# region import packages

# data analysis
import numpy as np
import xarray as xr
import pandas as pd
from metpy.calc import specific_humidity_from_dewpoint, relative_humidity_from_dewpoint, vertical_velocity_pressure, mixing_ratio_from_specific_humidity
from metpy.units import units
import calendar
import xesmf as xe
import pickle

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib import cm
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=10)
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})
mpl.use('Agg')
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as ticker
from matplotlib.colors import BoundaryNorm

# management
import os
import sys  # print(sys.path)
sys.path.append(os.getcwd() + '/code/gbr_future/module')
import psutil
process = psutil.Process()
# print(process.memory_info().rss / 2**30)
import string
import time
import glob

# self defined
from mapplot import (
    regional_plot,
    plot_maxmin_points,
    remove_trailing_zero_pos)

from namelist import (
    seconds_per_d,
    zerok,
    era5_varlabels,
    cmip6_era5_var,
    month_jan)

from component_plot import (
    plt_mesh_pars)

# endregion


# region plot am/sea/mon data


dss = ['ERA5', 'BARRA-R2', 'BARRA-C2']
regridder = {}
min_lon1, max_lon1, min_lat1, max_lat1 = 80, 220, -70, 20
min_lon, max_lon, min_lat, max_lat = 110.58, 157.34, -43.69, -7.01
pwidth  = 6.6
pheight = 6.6 * (max_lat1 - min_lat1) / (max_lon1 - min_lon1)
nrow = 1
ncol = len(dss)
fm_bottom = 1.6/(pheight*nrow+2.1)
fm_top = 1 - 0.5/(pheight*nrow+2.1)


era5_sl_mon_alltime = {}
barra_r2_mon_alltime = {}
barra_c2_mon_alltime = {}

periods = ['am', 'mon'] # ['am', 'sea', 'mon']
modes = ['org', 'diff'] # ['org', 'diff']
for var2 in ['clwvi', 'clivi', 'prw', 'pr', 'clh', 'clm', 'cll', 'clt', 'evspsbl', 'hfls', 'hfss', 'psl', 'rlds', 'rldscs', 'rlus', 'rluscs', 'rlut', 'rlutcs', 'rsds', 'rsdscs', 'rsdt', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'sfcWind', 'tas', 'ts', 'evspsblpot', 'uas', 'vas', 'rlns',  'rsns',  'rlnscs', 'rsnscs',  'rlnscl', 'rsnscl', 'rldscl', 'rsdscl',  'rluscl', 'rsuscl',  'rsnt',  'rsntcs',  'rlutcl', 'rsntcl', 'rsutcl']:
    # var2='clwvi'
    # ['clwvi', 'clivi', 'prw', 'pr', 'clh', 'clm', 'cll', 'clt', 'evspsbl', 'hfls', 'hfss', 'psl', 'rlds', 'rldscs', 'rlus', 'rluscs', 'rlut', 'rlutcs', 'rsds', 'rsdscs', 'rsdt', 'rsus', 'rsuscs', 'rsut', 'rsutcs', 'sfcWind', 'tas', 'ts', 'evspsblpot', 'uas', 'vas', 'rlns',  'rsns',  'rlnscs', 'rsnscs',  'rlnscl', 'rsnscl', 'rldscl', 'rsdscl',  'rluscl', 'rsuscl',  'rsnt',  'rsntcs',  'rlutcl', 'rsntcl', 'rsutcl']
    var1 = cmip6_era5_var[var2]
    print(f'#-------------------------------- {var1} in ERA5 vs. {var2} in BARRA-R2/C2')
    
    with open(f'data/obs/era5/mon/era5_sl_mon_alltime_{var1}.pkl', 'rb') as f:
        era5_sl_mon_alltime[var1] = pickle.load(f)
    with open(f'data/sim/um/barra_r2/barra_r2_mon_alltime_{var2}.pkl','rb') as f:
        barra_r2_mon_alltime[var2] = pickle.load(f)
    with open(f'data/sim/um/barra_c2/barra_c2_mon_alltime_{var2}.pkl','rb') as f:
        barra_c2_mon_alltime[var2] = pickle.load(f)
    
    if var1=='tp':
        pltlevel1 = np.array([0, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20,])
        pltticks1 = np.array([0, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20,])
        pltnorm1 = BoundaryNorm(pltlevel1, ncolors=len(pltlevel1)-1, clip=True)
        pltcmp1 = plt.get_cmap('viridis_r', len(pltlevel1)-1)
        extend1 = 'max'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-6,cm_max=6,cm_interval1=1,cm_interval2=1,cmap='BrBG_r')
    elif var1 in ['tcc', 'hcc', 'mcc', 'lcc']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=10, cmap='Blues_r',)
        extend1 = 'neither'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=10, cmap='BrBG_r',)
    elif var1 in ['e', 'pev']:
        pltlevel1 = np.array([0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8])
        pltticks1 = np.array([0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8])
        pltnorm1 = BoundaryNorm(pltlevel1, ncolors=len(pltlevel1)-1, clip=True)
        pltcmp1 = plt.get_cmap('viridis_r', len(pltlevel1)-1)
        extend1 = 'both'
        pltlevel2 = np.array([-5, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 5])
        pltticks2 = np.array([-5, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 5])
        pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
        pltcmp2 = plt.get_cmap('BrBG', len(pltlevel2)-1)
    elif var1=='mslhf':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-280, cm_max=40, cm_interval1=10, cm_interval2=40, cmap='PRGn', asymmetric=True)
        extend1 = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=10, cmap='BrBG')
    elif var1=='msshf':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-200, cm_max=80, cm_interval1=10, cm_interval2=40, cmap='PRGn', asymmetric=True,)
        extend1 = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=10, cmap='BrBG')
    elif var1=='msl':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=980, cm_max=1028, cm_interval1=2, cm_interval2=8, cmap='viridis_r',)
        extend1 = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-0.5, cm_max=0.5, cm_interval1=0.1, cm_interval2=0.1, cmap='BrBG')
    elif var1 in ['msdwlwrf', 'msdwlwrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=200, cm_max=440, cm_interval1=10, cm_interval2=40, cmap='viridis_r',)
        extend1 = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=10, cmap='BrBG')
    elif var1 in ['msuwlwrf', 'msuwlwrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-540, cm_max=-300, cm_interval1=10, cm_interval2=40, cmap='viridis',)
        extend1 = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=10, cmap='BrBG')
    elif var1 in ['mtnlwrf', 'mtnlwrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-310, cm_max=-180, cm_interval1=5, cm_interval2=20, cmap='viridis',)
        extend1 = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=10, cmap='BrBG')
    elif var1 in ['msdwswrf', 'msdwswrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=40, cm_max=400, cm_interval1=10, cm_interval2=40, cmap='viridis_r',)
        extend1 = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=10, cmap='BrBG')
    elif var1=='mtdwswrf':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=150, cm_max=490, cm_interval1=10, cm_interval2=40, cmap='viridis_r',)
        extend1 = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-0.6, cm_max=0.6, cm_interval1=0.1, cm_interval2=0.2, cmap='BrBG',)
    elif var1 in ['msuwswrf', 'msuwswrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-160, cm_max=0, cm_interval1=5, cm_interval2=20, cmap='viridis',)
        extend1 = 'min'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=10, cmap='BrBG')
    elif var1 in ['mtuwswrf', 'mtuwswrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-250, cm_max=-20, cm_interval1=10, cm_interval2=40, cmap='viridis',)
        extend1 = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=10, cmap='BrBG')
    elif var1 in ['si10']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=0, cm_max=14, cm_interval1=1, cm_interval2=2, cmap='viridis_r',)
        extend1 = 'max'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-3, cm_max=3, cm_interval1=0.5, cm_interval2=1, cmap='BrBG')
    elif var1=='t2m':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-20, cm_max=35, cm_interval1=2.5, cm_interval2=5, cmap='PuOr', asymmetric=True)
        extend1 = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-4, cm_max=4, cm_interval1=1, cm_interval2=1, cmap='BrBG')
    elif var1=='skt':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-20, cm_max=35, cm_interval1=2.5, cm_interval2=5, cmap='PuOr', asymmetric=True)
        extend1 = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-4, cm_max=4, cm_interval1=1, cm_interval2=1, cmap='BrBG')
    elif var1 in ['u10', 'v10']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-10, cm_max=10, cm_interval1=1, cm_interval2=2, cmap='PRGn',)
        extend1 = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-1, cm_max=1, cm_interval1=0.1, cm_interval2=0.2, cmap='BrBG',)
    elif var1 in ['msnlwrf', 'msnlwrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-150, cm_max=0, cm_interval1=10, cm_interval2=20, cmap='viridis',)
        extend1 = 'min'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=10, cmap='BrBG')
    elif var1 in ['msnswrf', 'msnswrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=30, cm_max=350, cm_interval1=10, cm_interval2=40, cmap='viridis',)
        extend1 = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=10, cmap='BrBG')
    elif var1=='msnlwrfcl':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-25, cm_max=70, cm_interval1=5, cm_interval2=10, cmap='PRGn', asymmetric=True)
        extend1 = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=10, cmap='BrBG')
    elif var1=='msnswrfcl':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-230, cm_max=30, cm_interval1=10, cm_interval2=40, cmap='PRGn', asymmetric=True)
        extend1 = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=10, cmap='BrBG')
    elif var1=='msdwlwrfcl':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=0, cm_max=70, cm_interval1=2.5, cm_interval2=10, cmap='viridis_r',)
        extend1 = 'max'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=10, cmap='BrBG')
    elif var1=='msdwswrfcl':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-270, cm_max=0, cm_interval1=10, cm_interval2=40, cmap='viridis',)
        extend1 = 'min'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=10, cmap='BrBG')
    elif var1=='msuwlwrfcl':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-40, cm_max=40, cm_interval1=5, cm_interval2=10, cmap='PRGn',)
        extend1 = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=10, cmap='BrBG')
    elif var1=='msuwswrfcl':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-40, cm_max=60, cm_interval1=5, cm_interval2=10, cmap='PRGn', asymmetric=True)
        extend1 = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=10, cmap='BrBG')
    elif var1 in ['mtnswrf', 'mtnswrfcs']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=70, cm_max=440, cm_interval1=10, cm_interval2=40, cmap='viridis_r',)
        extend1 = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=10, cmap='BrBG')
    elif var1=='mtnlwrfcl':
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=0, cm_max=80, cm_interval1=5, cm_interval2=10, cmap='viridis_r',)
        extend1 = 'max'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=10, cmap='BrBG')
    elif var1 in ['mtnswrfcl', 'mtuwswrfcl']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=-180, cm_max=30, cm_interval1=10, cm_interval2=20, cmap='PRGn', asymmetric=True)
        extend1 = 'both'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-30, cm_max=30, cm_interval1=5, cm_interval2=10, cmap='BrBG')
    elif var1 in ['tcwv']:
        pltlevel1, pltticks1, pltnorm1, pltcmp1 = plt_mesh_pars(
            cm_min=0, cm_max=80, cm_interval1=5, cm_interval2=10, cmap='Blues_r',)
        extend1 = 'max'
        pltlevel2, pltticks2, pltnorm2, pltcmp2 = plt_mesh_pars(
            cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='BrBG_r')
    elif var1 in ['tclw']:
        pltlevel1 = np.array([0, 0.01, 0.02, 0.04, 0.06, 0.1, 0.2, 0.3, 0.4])
        pltticks1 = np.array([0, 0.01, 0.02, 0.04, 0.06, 0.1, 0.2, 0.3, 0.4])
        pltnorm1 = BoundaryNorm(pltlevel1, ncolors=len(pltlevel1)-1, clip=True)
        pltcmp1 = plt.get_cmap('viridis_r', len(pltlevel1)-1)
        extend1 = 'max'
        pltlevel2 = np.array([-0.1, -0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06, 0.1])
        pltticks2 = np.array([-0.1, -0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06, 0.1])
        pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
        pltcmp2 = plt.get_cmap('BrBG', len(pltlevel2)-1)
    elif var1 in ['tciw']:
        pltlevel1 = np.array([0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1])
        pltticks1 = np.array([0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1])
        pltnorm1 = BoundaryNorm(pltlevel1, ncolors=len(pltlevel1)-1, clip=True)
        pltcmp1 = plt.get_cmap('viridis_r', len(pltlevel1)-1)
        extend1 = 'max'
        pltlevel2 = np.array([-1, -0.8, -0.6, -0.4, -0.2, -0.1, 0, 0.1, 0.2, 0.4, 0.6, 0.8, 1])
        pltticks2 = np.array([-1, -0.8, -0.6, -0.4, -0.2, -0.1, 0, 0.1, 0.2, 0.4, 0.6, 0.8, 1])
        pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
        pltcmp2 = plt.get_cmap('BrBG', len(pltlevel2)-1)
    else:
        print(f'Warning unspecified colorbar for {var1}')
    
    for iperiod in periods:
        # iperiod = 'am'
        # ['am', 'sea', 'mon']
        print(f'#---------------- {iperiod}')
        
        if iperiod=='am':
            ostr = '1979-2023'
        elif iperiod=='sea':
            iyear = 2020
            isea = 'JJA'
            ostr = f'{isea}-{iyear}'
        elif iperiod=='mon':
            iyear = 2020
            imon = 6
            ostr = f'{month_jan[imon-1]}-{iyear}'
        
        plt_data = {}
        if iperiod=='am':
            plt_data['ERA5'] = era5_sl_mon_alltime[var1]['am'].squeeze()
            plt_data['BARRA-R2'] = barra_r2_mon_alltime[var2]['am'].squeeze()
            plt_data['BARRA-C2'] = barra_c2_mon_alltime[var2]['am'].squeeze()
        elif iperiod=='sea':
            plt_data['ERA5'] = era5_sl_mon_alltime[var1]['sea'][(era5_sl_mon_alltime[var1]['sea'].time.dt.year == iyear) & (era5_sl_mon_alltime[var1]['sea'].time.dt.season == isea)].squeeze()
            plt_data['BARRA-R2'] = barra_r2_mon_alltime[var2]['sea'][(barra_r2_mon_alltime[var2]['sea'].time.dt.year == iyear) & (barra_r2_mon_alltime[var2]['sea'].time.dt.season == isea)].squeeze()
            plt_data['BARRA-C2'] = barra_c2_mon_alltime[var2]['sea'][(barra_c2_mon_alltime[var2]['sea'].time.dt.year == iyear) & (barra_c2_mon_alltime[var2]['sea'].time.dt.season == isea)].squeeze()
        elif iperiod=='mon':
            plt_data['ERA5'] = era5_sl_mon_alltime[var1]['mon'][(era5_sl_mon_alltime[var1]['mon'].time.dt.year == iyear) & (era5_sl_mon_alltime[var1]['mon'].time.dt.month == imon)].squeeze()
            plt_data['BARRA-R2'] = barra_r2_mon_alltime[var2]['mon'][(barra_r2_mon_alltime[var2]['mon'].time.dt.year == iyear) & (barra_r2_mon_alltime[var2]['mon'].time.dt.month == imon)].squeeze()
            plt_data['BARRA-C2'] = barra_c2_mon_alltime[var2]['mon'][(barra_c2_mon_alltime[var2]['mon'].time.dt.year == iyear) & (barra_c2_mon_alltime[var2]['mon'].time.dt.month == imon)].squeeze()
        
        plt_data['ERA5']['lon'] = plt_data['ERA5']['lon'] % 360
        plt_data['ERA5'] = plt_data['ERA5'].sortby(['lon', 'lat']).sel(lon=slice(min_lon1, max_lon1), lat=slice(min_lat1, max_lat1))
        
        for imode in modes:
            # imode = 'org'
            # ['org', 'diff']
            print(f'#-------- {imode}')
            
            if imode=='org':
                plt_colnames = dss
            elif imode=='diff':
                plt_colnames = [dss[0]] + [f'{ids1} - {ids2}' for ids1, ids2 in zip(dss[1:], dss[:-1])]
            
            opng = f'figures/4_um/4.0_barra/4.0.5_case_studies/4.0.5.3_{var2} in {', '.join(dss)} {imode} {min_lon1}_{max_lon1}_{min_lat1}_{max_lat1} {iperiod} {ostr}.png'
            
            fig, axs = plt.subplots(
                nrow, ncol, figsize=np.array([pwidth*ncol, pheight*nrow+2.1])/2.54,
                subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
                gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)
            
            for jcol in range(ncol):
                axs[jcol] = regional_plot(extent=[min_lon1, max_lon1, min_lat1, max_lat1], central_longitude=180, ax_org=axs[jcol], lw=0.1)
                axs[jcol].text(0, 1.02, f'({string.ascii_lowercase[jcol]}) {plt_colnames[jcol]}', ha='left', va='bottom', transform=axs[jcol].transAxes)
                axs[jcol].add_patch(Rectangle(
                    (min_lon, min_lat), max_lon-min_lon, max_lat-min_lat,
                    ec='red', color='None', lw=0.5,
                    transform=ccrs.PlateCarree(), zorder=2))
                axs[jcol].add_patch(Rectangle(
                    (barra_r2_mon_alltime[var2]['am'].lon[0],
                     barra_r2_mon_alltime[var2]['am'].lat[0]),
                    barra_r2_mon_alltime[var2]['am'].lon[-1] - barra_r2_mon_alltime[var2]['am'].lon[0],
                    barra_r2_mon_alltime[var2]['am'].lat[-1] - barra_r2_mon_alltime[var2]['am'].lat[0],
                    ec='red', color='None', lw=0.5, linestyle='--',
                    transform=ccrs.PlateCarree(), zorder=2))
                axs[jcol].add_patch(Rectangle(
                    (barra_c2_mon_alltime[var2]['am'].lon[0],
                     barra_c2_mon_alltime[var2]['am'].lat[0]),
                    barra_c2_mon_alltime[var2]['am'].lon[-1] - barra_c2_mon_alltime[var2]['am'].lon[0],
                    barra_c2_mon_alltime[var2]['am'].lat[-1] - barra_c2_mon_alltime[var2]['am'].lat[0],
                    ec='red', color='None', lw=0.5, linestyle=':',
                    transform=ccrs.PlateCarree(), zorder=2))
            
            if imode=='org':
                for jcol, ids in enumerate(dss):
                    plt_mesh = axs[jcol].pcolormesh(
                        plt_data[ids].lon, plt_data[ids].lat, plt_data[ids],
                        norm=pltnorm1, cmap=pltcmp1,
                        transform=ccrs.PlateCarree(), zorder=1)
                cbar = fig.colorbar(
                    plt_mesh, #cm.ScalarMappable(norm=pltnorm1, cmap=pltcmp1), #
                    format=remove_trailing_zero_pos,
                    orientation="horizontal", ticks=pltticks1, extend=extend1,
                    cax=fig.add_axes([1/3, fm_bottom-0.115, 1/3, 0.03]))
                cbar.ax.set_xlabel(f'{ostr} {era5_varlabels[var1]}', fontsize=9, labelpad=1)
                cbar.ax.tick_params(labelsize=9, pad=1)
            elif imode=='diff':
                plt_mesh = axs[0].pcolormesh(
                    plt_data['ERA5'].lon, plt_data['ERA5'].lat,
                    plt_data['ERA5'],
                    norm=pltnorm1, cmap=pltcmp1,
                    transform=ccrs.PlateCarree(), zorder=1)
                for jcol, ids1, ids2 in zip(range(1, len(dss)), dss[1:], dss[:-1]):
                    # jcol=1; ids1='BARRA-R2'; ids2='ERA5'
                    # print(f'#-------- {jcol} {ids1} {ids2}')
                    if not f'{ids1} - {ids2}' in regridder.keys():
                        regridder[f'{ids1} - {ids2}'] = xe.Regridder(
                            plt_data[ids1],
                            plt_data[ids2].sel(lon=slice(plt_data[ids1].lon[0], plt_data[ids1].lon[-1]), lat=slice(plt_data[ids1].lat[0], plt_data[ids1].lat[-1])),
                            method='bilinear')
                    plt_data_tem = regridder[f'{ids1} - {ids2}'](plt_data[ids1]) - plt_data[ids2].sel(lon=slice(plt_data[ids1].lon[0], plt_data[ids1].lon[-1]), lat=slice(plt_data[ids1].lat[0], plt_data[ids1].lat[-1]))
                    plt_mesh2 = axs[jcol].pcolormesh(
                        plt_data_tem.lon, plt_data_tem.lat, plt_data_tem,
                        norm=pltnorm2, cmap=pltcmp2,
                        transform=ccrs.PlateCarree(), zorder=1)
                cbar = fig.colorbar(
                    plt_mesh, #cm.ScalarMappable(norm=pltnorm1, cmap=pltcmp1), #
                    format=remove_trailing_zero_pos,
                    orientation="horizontal", ticks=pltticks1, extend=extend1,
                    cax=fig.add_axes([0.05, fm_bottom-0.115, 0.4, 0.03]))
                cbar.ax.set_xlabel(f'{ostr} {era5_varlabels[var1]}', fontsize=9, labelpad=1)
                cbar.ax.tick_params(labelsize=9, pad=1)
                cbar2 = fig.colorbar(
                    plt_mesh2, #cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), #
                    format=remove_trailing_zero_pos,
                    orientation="horizontal", ticks=pltticks2, extend='both',
                    cax=fig.add_axes([0.55, fm_bottom-0.115, 0.4, 0.03]))
                cbar2.ax.set_xlabel(f'Difference in {era5_varlabels[var1]}',
                                    fontsize=9, labelpad=1)
                cbar2.ax.tick_params(labelsize=9, pad=1)
            
            fig.subplots_adjust(left=0.005, right=0.995, bottom=fm_bottom, top=fm_top)
            fig.savefig(opng)
    
    del era5_sl_mon_alltime[var1], barra_r2_mon_alltime[var2], barra_c2_mon_alltime[var2]




# endregion
