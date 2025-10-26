

# qsub -I -q normal -P v46 -l walltime=3:00:00,ncpus=1,mem=96GB,jobfs=100MB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22


# region import packages

# data analysis
import numpy as np
import pandas as pd
import numpy.ma as ma
import glob
from datetime import datetime, timedelta
from pyhdf.SD import SD, SDC
from pyhdf.HDF import HDF
from pyhdf.VS import VS
from pyhdf.error import HDF4Error
from satpy.scene import Scene
from skimage.measure import block_reduce
import xarray as xr
import matplotlib.animation as animation

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=10)
mpl.rcParams['axes.linewidth'] = 0.2
plt.rcParams.update({"mathtext.fontset": "stix"})

# management
import os
import sys  # print(sys.path)
sys.path.append(os.getcwd() + '/code/gbr_future/module')
import psutil
process = psutil.Process()
# print(process.memory_info().rss / 2**30)

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
    month_jan,
    monthini,
    seasons,
    seconds_per_d,
    zerok,
    month_days,
    era5_varlabels,
    cmip6_era5_var,
    )

from component_plot import (
    rainbow_text,
    change_snsbar_width,
    cplot_wind_vectors,
    cplot_lon180,
    cplot_lon180_ctr,
    plt_mesh_pars,
    plot_loc,
    draw_polygon,
)

from calculations import (
    find_ilat_ilon,
    )

from metplot import si2reflectance, si2radiance, get_modis_latlonrgbs, get_modis_latlonvar, get_modis_latlonvars


# endregion


# region plot Calibrated Radiances

# option
products = ['MYD02HKM'] # 'MOD02HKM', 'MYD02HKM', 'MOD021KM' 'MYD021KM'
plt_regions = ['NE'] # 'global', 'c2_domain'
year, month, day, hour, minute = 2020, 6, 2, 3, 30
# year, month, day, hour, minute = 2020, 6, 2, 5, 00
doy = datetime(year, month, day).timetuple().tm_yday
plt_scene = 'hourly' # 'minutely'

for iproduct in products:
    # iproduct = 'MYD02HKM'
    print(f'#-------------------------------- {iproduct}')
    
    if plt_scene == 'hourly':
        fl = sorted(glob.glob(f'data/obs/MODIS/{iproduct}/{year}/{doy:03d}/*.{hour:02d}??.061.*.hdf'))
    elif plt_scene == 'minutely':
        fl = sorted(glob.glob(f'data/obs/MODIS/{iproduct}/{year}/{doy:03d}/*.{hour:02d}{minute:02d}.061.*.hdf'))
    if len(fl)==0: print('Warning: no file found')
    
    lats, lons, rgbs = get_modis_latlonrgbs(fl)
    if lats.shape[0] == 2 * rgbs.shape[0]:
        lats = block_reduce(lats, block_size=(2, 2), func=np.min)
        lons = block_reduce(lons, block_size=(2, 2), func=np.min)
    
    for plt_region in plt_regions:
        # plt_region = 'global'
        print(f'#---------------- {plt_region}')
        
        if plt_scene == 'hourly':
            opng = f'figures/3_satellites/3.2_modis/3.2.0_images/3.2.0.0 {iproduct} {plt_region} {plt_scene} {year}{month:02d}{day:02d} {hour:02d}.png'
            label = f'{iproduct} {year}-{month:02d}-{day:02d} {hour:02d}:00 UTC'
        elif plt_scene == 'minutely':
            opng = f'figures/3_satellites/3.2_modis/3.2.0_images/3.2.0.0 {iproduct} {plt_region} {plt_scene} {year}{month:02d}{day:02d} {hour:02d}{minute:02d}.png'
            label = f'{iproduct} {year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d} UTC'
        
        if plt_region == 'global':
            fig, ax = globe_plot(
                figsize=np.array([24, 13]) / 2.54, lw=0.1,
                projections = ccrs.PlateCarree(central_longitude=180))
            ax.pcolormesh(lons, lats, rgbs, transform=ccrs.PlateCarree())
            fig.text(0.5, 0.01, label, ha='center', va='bottom')
            fig.subplots_adjust(left=0.01,right=0.99,bottom=0.05,top=0.99)
        elif plt_region == 'c2_domain':
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
        elif plt_region == 'NE':
            min_lon, max_lon, min_lat, max_lat = [137.12, 157.3, -28.76, -7.05]
            mask = (lons>=min_lon) & (lons<=max_lon) & (lats>=min_lat) & (lats<=max_lat)
            rgbscopy = rgbs.copy()
            rgbscopy[np.broadcast_to(~mask[:, :, np.newaxis], rgbs.shape)] = np.nan
            fig, ax = regional_plot(
                extent=[min_lon, max_lon, min_lat, max_lat],
                central_longitude=180,
                figsize = np.array([4.4, 4.4]) / 2.54,
                lw=0.4, border_color='yellow')
            ax.pcolormesh(lons, lats, rgbscopy, transform=ccrs.PlateCarree())
            # fig.text(0.5, 0.01, label, ha='center', va='bottom')
            fig.subplots_adjust(left=0.01,right=0.99,bottom=0.01,top=0.99)
        
        fig.savefig(opng, dpi=1200)



'''
https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/science-domain/modis-L0L1/

https://modis.gsfc.nasa.gov/about/specifications.php
# Red: Band 1: 620 - 670 nm
# Green: Band 4: 545 - 565 nm
# Blue: Band 3: 459 - 479 nm

Band 32: 11.770 - 12.270
Band 31: 10.780 - 11.280
Band 29: 8.400 - 8.700

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

scn.available_dataset_names()
scn.available_composite_ids()
scn.load(["night_fog"])
image = np.asarray(scn["night_fog"]).transpose(1,2,0)
image = np.nan_to_num(image)
image = np.interp(image, (np.percentile(image,1), np.percentile(image,99)), (0, 1))
# ax.pcolormesh(lon, lat, np.zeros_like(lat), color=image.reshape(-1, 3),
#               transform=ccrs.PlateCarree())

print(hdf.datasets().keys())
print(hdf.select(varnames[iproduct])[:].shape)

'''
# endregion


# region animate Calibrated Radiances

# options
products = ['MYD021KM'] # 'MOD021KM' 'MYD021KM'
plt_regions = ['global'] # 'global'
plt_scene = 'minutely' # 'hourly', 'minutely'

starttime = datetime(2020, 6, 1, 0)
endtime   = datetime(2020, 6, 2, 23)

def update_frames(itime):
    # itime = 0
    global plt_objs
    for plt_obj in plt_objs:
        plt_obj.remove()
    plt_objs = []
    
    fl = fls[itime]
    lats, lons, rgbs = get_modis_latlonrgbs(fl)
    if lats.shape[0] == 2 * rgbs.shape[0]:
        lats = block_reduce(lats, block_size=(2, 2), func=np.min)
        lons = block_reduce(lons, block_size=(2, 2), func=np.min)
    
    flname = os.path.basename(fl[0]).split('.')
    iproduct, year, doy, hour, minute = flname[0], flname[1][1:5], flname[1][5:8], flname[2][:2], flname[2][2:4]
    dt = datetime.strptime(f'{year}{doy}', "%Y%j")
    month, day = dt.month, dt.day
    if len(fl) == 1:
        label = f'{iproduct} {year}-{month:02d}-{day:02d} {hour}:{minute} UTC'
    else:
        label = f'{iproduct} {year}-{month:02d}-{day:02d} {hour}:00 UTC'
    
    plt_mesh = ax.pcolormesh(lons, lats, rgbs, transform=ccrs.PlateCarree())
    plt_text = fig.text(0.5, 0.01, label, ha='center', va='bottom')
    
    plt_objs = [plt_mesh, plt_text]
    return(plt_objs)


for iproduct in products:
    # iproduct = 'MYD02HKM'
    print(f'#-------------------------------- {iproduct}')
    
    fls = []
    timeseries = pd.date_range(start=starttime, end=endtime, freq='h')
    for itime in timeseries:
        print(f'# {itime}')
        year, month, day, hour = itime.year, itime.month, itime.day, itime.hour
        doy = datetime(year, month, day).timetuple().tm_yday
        fl = sorted(glob.glob(f'scratch/data/obs/MODIS/{iproduct}/{year}/{doy:03d}/*.{hour:02d}??.061.*.hdf'))
        if plt_scene == 'hourly':
            fls.append(fl)
        elif plt_scene == 'minutely':
            fls += [[ifile] for ifile in fl]
    
    for plt_region in plt_regions:
        # plt_region = 'global'
        print(f'#---------------- {plt_region}')
        
        omp4 = f'figures/3_satellites/3.2_modis/3.2.0_images/3.2.0.0 {iproduct} {plt_region} {plt_scene} {str(starttime)[:13]} to {str(endtime)[:13]}.mp4'
        
        if plt_region == 'global':
            fig, ax = globe_plot(
                figsize=np.array([24, 13]) / 2.54, lw=0.1,
                projections = ccrs.PlateCarree(central_longitude=180))
            fig.subplots_adjust(left=0.01,right=0.99,bottom=0.05,top=0.99)
        else:
            print('Warning: unspecified region')
        
        plt_objs = []
        ani = animation.FuncAnimation(
            fig, update_frames, frames=len(fls), interval=500, blit=False)
        if os.path.exists(omp4): os.remove(omp4)
        ani.save(omp4, progress_callback=lambda iframe, n: print(f'Frame {iframe}/{n}'))




# endregion


# region plot L2 products

# 'MOD05_L2', 'MOD06_L2', 'MOD07_L2', 'MODATML2', 'MOD08_D3'
# 'MYD05_L2', 'MYD06_L2', 'MYD07_L2', 'MYDATML2', 'MYD08_D3'
products_vars = {
    # 'MYD05_L2': ['Water_Vapor_Near_Infrared', 'Water_Vapor_Infrared'],
    'MYD06_L2': ['Cloud_Water_Path'], # 'Cloud_Fraction', 'Cloud_Water_Path', 'Cloud_Water_Path_Uncertainty', 'Brightness_Temperature', 'Cloud_Top_Height', 'Cloud_Top_Pressure', 'Cloud_Top_Temperature', 'Cloud_Effective_Radius', 'Cloud_Optical_Thickness', 'Cloud_Phase_Infrared'
    # 'MYD07_L2': ['Water_Vapor'],
    # 'MYDATML2': ['Cloud_Water_Path'] # ['Cloud_Water_Path', 'Cloud_Fraction', 'Precipitable_Water_Infrared_ClearSky', 'Precipitable_Water_Near_Infrared_ClearSky']
    }
year, month, day, hour, minute = 2020, 6, 2, 3, 0
doy = datetime(year, month, day).timetuple().tm_yday
plt_regions = ['NE'] # 'global', 'c2_domain'
plt_scene = 'hourly' # 'minutely'


for iproduct in products_vars.keys():
    # iproduct = 'MYD06_L2'
    print(f'#-------------------------------- {iproduct}')
    
    if plt_scene == 'hourly':
        fl = sorted(glob.glob(f'data/obs/MODIS/{iproduct}/{year}/{doy:03d}/*.{hour:02d}??.061.*.hdf'))
    elif plt_scene == 'minutely':
        fl = sorted(glob.glob(f'data/obs/MODIS/{iproduct}/{year}/{doy:03d}/*.{hour:02d}{minute:02d}.061.*.hdf'))
    if len(fl)==0: print('Warning: no file found')
    
    for ivar in products_vars[iproduct]:
        # ivar = 'Cloud_Fraction'
        # ivar = 'Cloud_Water_Path'
        print(f'#---------------- {ivar}')
        
        lats, lons, vardata = get_modis_latlonvars(fl, ivar)
        
        if ivar in ['Cloud_Fraction']:
            vardata *= 100
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=100, cm_interval1=5, cm_interval2=10, cmap='viridis')
            var_labels = r'Cloud fraction [$\%$]'
            extend = 'neither'
        elif ivar in ['Cloud_Water_Path']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=600, cm_interval1=50, cm_interval2=100, cmap='viridis')
            extend = 'max'
            var_labels = r'CWP [$g \; m^{-2}$]'
        elif ivar in ['Cloud_Water_Path_Uncertainty']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=50, cm_interval1=5, cm_interval2=10, cmap='viridis')
            extend = 'max'
            var_labels = r'CWP uncertainty [$g \; m^{-2}$]'
        
        for plt_region in plt_regions:
            # plt_region = 'global'
            print(f'#-------- {plt_region}')
            
            if plt_scene == 'hourly':
                opng = f'figures/3_satellites/3.2_modis/3.2.1_cloud_properties/3.2.1.1 {iproduct} {ivar} {plt_region} {plt_scene} {year}{month:02d}{day:02d} {hour:02d}.png'
                label = f'{iproduct} {year}-{month:02d}-{day:02d} {hour:02d}:00 UTC\n{var_labels}'
            elif plt_scene == 'minutely':
                opng = f'figures/3_satellites/3.2_modis/3.2.1_cloud_properties/3.2.1.1 {iproduct} {ivar} {plt_region} {plt_scene} {year}{month:02d}{day:02d} {hour:02d}{minute:02d}.png'
                label = f'{iproduct} {year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d} UTC\n{var_labels}'
            
            if plt_region == 'global':
                fig, ax = globe_plot(
                    figsize=np.array([8.8, 6.6]) / 2.54, lw=0.1,
                    projections = ccrs.PlateCarree(central_longitude=180))
            elif plt_region == 'c2_domain':
                fig, ax = regional_plot(
                    extent=[110.58, 157.34, -43.69, -7.01],
                    central_longitude=180,
                    figsize = np.array([6.6, 6.6]) / 2.54)
            elif plt_region == 'NE':
                fig, ax = regional_plot(
                    extent=[137.12, 157.3, -28.76, -7.05],
                    central_longitude=180,
                    figsize = np.array([4.4, 6.6]) / 2.54)
            
            plt_mesh = ax.pcolormesh(
                lons, lats, vardata,
                norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree())
            cbar = fig.colorbar(
                plt_mesh, #cm.ScalarMappable(norm=pltnorm1, cmap=pltcmp1), #
                format=remove_trailing_zero_pos,
                orientation="horizontal", ticks=pltticks, extend=extend,
                cax=fig.add_axes([0.05, 0.24, 0.9, 0.03]))
            cbar.ax.set_xlabel(label, fontsize=10, labelpad=3, linespacing=1.5)
            fig.subplots_adjust(left=0.01,right=0.99,bottom=0.28,top=0.99)
            fig.savefig(opng, dpi=1200)





'''
#-------------------------------- check product variables
products = ['MYD04_L2', 'MYD05_L2', 'MYD06_L2', 'MYD07_L2', 'MYDATML2', 'MYD08_M3']
year, month, day, hour, minute = 2020, 6, 2, 3, 20
doy = datetime(year, month, day).timetuple().tm_yday

for iproduct in products:
    # iproduct = 'MYD06_L2'
    print(f'#-------------------------------- {iproduct}')
    
    ifile = sorted(glob.glob(f'scratch/data/obs/MODIS/{iproduct}/{year}/{doy:03d}/*.{hour:02d}{minute:02d}.061.*.hdf'))[0]
    hdf_sd = SD(ifile, SDC.READ)
    for ivar in hdf_sd.datasets().keys():
        print(f'{ivar}')



ifile = fl[0]
ivar = 'Cloud_Fraction'
ivar = 'Cloud_Water_Path'

get_modis_latlonvar(ifile, ivar)
get_modis_latlonvars(fl, ivar)

'''
# endregion


# region plot 'MOD08_M3', 'MYD08_M3': total column  q, qcl, qcf

products = ['MOD08_M3', 'MYD08_M3']
vars = ['prw', 'clivi', 'clwvi']
years = '2016'; yeare = '2023'

for iproduct in products:
    # iproduct = 'MOD08_M3'
    print(f'#---------------- {iproduct}')
    
    dss = xr.open_dataset(f'scratch/data/obs/MODIS/{iproduct}/{'_'.join(vars)}.nc').sel(time=slice(years, yeare))
    
    for ivar in ['clivi', 'clwvi']:
        # ivar = vars[2]
        print(f'#-------- {ivar}')
        
        plt_data = np.average(
            dss[ivar].groupby('time.month').mean('time', skipna=True),
            axis=0,
            weights=month_days)
        plt_data_gm = np.average(
            plt_data[np.isfinite(plt_data)],
            weights=np.cos(np.deg2rad(np.repeat(dss.lat.values[:, np.newaxis], len(dss.lon), axis=1)))[np.isfinite(plt_data)])
        
        # print(np.nanmean(plt_data))
        
        cbar_label = f'{iproduct} annual mean ({years}-{yeare}) {era5_varlabels[cmip6_era5_var[ivar]]}\nglobal mean: {str(np.round(plt_data_gm, 2))}'
        
        if ivar=='clivi':
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=240, cm_interval1=10, cm_interval2=20, cmap='viridis',)
        elif ivar=='clwvi':
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=160, cm_interval1=10, cm_interval2=20, cmap='viridis',)
        elif ivar in ['prw']:
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=60, cm_interval1=3, cm_interval2=6, cmap='viridis',)
        else:
            print(f'Warning unspecified colorbar for {ivar}')
        
        fig, ax = globe_plot(figsize=np.array([12, 8]) / 2.54, fm_bottom=0.13)
        plt_mesh1 = ax.pcolormesh(
            dss.lon, dss.lat, plt_data,
            norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),zorder=1,)
        cbar = fig.colorbar(
            plt_mesh1, ax=ax, aspect=40, format=remove_trailing_zero_pos,
            orientation="horizontal", shrink=0.8, ticks=pltticks, extend='max',
            pad=0.02, fraction=0.13,)
        cbar.ax.set_xlabel(cbar_label, ha='center', linespacing=1.3, labelpad=4)
        fig.savefig(f'figures/3_satellites/3.2_modis/3.2.1_cloud_properties/3.2.1.0 {iproduct} {ivar} global am.png')




# endregion

