

# qsub -I -q normal -P v46 -l walltime=3:00:00,ncpus=1,mem=192GB,jobfs=100MB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22


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
    panel_labels,
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

from metplot import si2reflectance, si2radiance, get_modis_latlonrgbs


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
https://modis.gsfc.nasa.gov/about/specifications.php
# Red: Band 1: 620 - 670 nm
# Green: Band 4: 545 - 565 nm
# Blue: Band 3: 459 - 479 nm

Band 32: 11.770 - 12.270
Band 31: 10.780 - 11.280
Band 29: 8.400 - 8.700

        scn.available_dataset_names()
        scn.available_composite_ids()
        scn.load(["night_fog"])
        image = np.asarray(scn["night_fog"]).transpose(1,2,0)
        image = np.nan_to_num(image)
        image = np.interp(image, (np.percentile(image,1), np.percentile(image,99)), (0, 1))
        # ax.pcolormesh(lon, lat, np.zeros_like(lat), color=image.reshape(-1, 3),
        #               transform=ccrs.PlateCarree())
        
'''
# endregion


# region plot L2 products

year, month, day, hour = 2020, 6, 2, 3
doy = datetime(year, month, day).timetuple().tm_yday
regions = ['global', 'BARRA-C2']

for iproduct in ['MYD06_L2']:
    print(f'#-------------------------------- {iproduct}')
    
    fl = sorted(glob.glob(f'scratch/data/obs/MODIS/{iproduct}/{year}/{doy:03d}/*.{hour:02d}??.061.*.hdf'))
    hdf_sd = SD(fl[0], SDC.READ)
    
    for ivar in hdf_sd.datasets().keys():
        # ivar = 'Cloud_Fraction'
        if ivar == 'Cloud_Fraction':
            pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
                cm_min=0, cm_max=100, cm_interval1=10, cm_interval2=10,
                cmap='Blues',)
            extend = 'neither'
        else:
            continue
        print(f'#---------------- {ivar}')
        
        lat = []
        lon = []
        time = []
        var_data = []
        for ifile in fl:
            # ifile = fl[1]
            print(ifile)
            hdf_sd = SD(ifile, SDC.READ)
            lat.append(hdf_sd.select('Latitude')[:])
            lon.append(hdf_sd.select('Longitude')[:])
            time.append(np.datetime64('1993-01-01T00:00:00') + hdf_sd.select('Scan_Start_Time')[:].astype('timedelta64[s]'))
            var_data.append(hdf_sd.select(ivar)[:])
        
        lat = np.concatenate(lat, axis=0)
        lon = np.concatenate(lon, axis=0)
        time = np.concatenate(time, axis=0)
        var_data = np.concatenate(var_data, axis=0)
        
        for iregion in regions:
            # iregion = 'global'
            print(f'#-------- {iregion}')
            
            if iregion == 'global':
                min_lon, max_lon, min_lat, max_lat = [-180, 180, -90, 90]
                fig, ax = globe_plot(figsize=np.array([88, 44]) / 2.54, lw=1)
                fm_bottom = 0.2
            elif iregion == 'BARRA-C2':
                min_lon, max_lon, min_lat, max_lat = [110.58, 157.34, -43.69, -7.01]
                fig, ax = regional_plot(extent=[min_lon, max_lon, min_lat, max_lat], central_longitude=180)
            
            mask = (lon >= min_lon) & (lon <= max_lon) & (lat >= min_lat) & (lat <= max_lat)
            lon = ma.masked_where(~mask, lon)
            lat = ma.masked_where(~mask, lat)
            var_data = ma.masked_where(~mask, var_data)
            
            plt_mesh = ax.pcolormesh(
                lon, lat, var_data,
                norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree())
            
            cbar = fig.colorbar(
                plt_mesh, #cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), #
                format=remove_trailing_zero_pos,
                orientation="horizontal", ticks=pltticks, extend=extend,
                cax=fig.add_axes([0.05, fm_bottom-0.12, 0.9, 0.03]))
            cbar.ax.set_xlabel(f'{iproduct} {ivar}', linespacing=1.5)
            fig.savefig('figures/test.png')




'''
        # np.concatenate([SD(fl[0], SDC.READ).select('Latitude')[:], SD(fl[1], SDC.READ).select('Latitude')[:]], axis=0)


    # 'MOD021KM' 'MYD021KM': Calibrated Radiances
    # 'MOD03' 'MYD03':       Geolocation Fields
    # 'MOD04_L2' 'MYD04_L2': Aerosol Product
    # 'MOD05_L2' 'MYD05_L2': Total Precipitable Water
    # 'MOD06_L2' 'MYD06_L2': Cloud Product
    # 'MOD07_L2' 'MYD07_L2': Atmospheric Profiles
    # 'MOD35_L2' 'MYD35_L2': Cloud Mask


    hdf_vs = HDF(fl[0]).vstart()
    for ivdata in hdf_vs.vdatainfo():
        print(f'#---------------- {ivdata}')
    try:
        scn = Scene(filenames={'modis_l1b': [fl[0]]})
    except ValueError:
        scn = Scene(filenames={'modis_l2': [fl[0]]})
    for ids in scn.available_dataset_names():
        print(f'#---------------- {ids}')
    print(scn.available_composite_ids())
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
https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/science-domain/modis-L0L1/


        scn = Scene(filenames={'modis_l1b': [ifileH]})
        scn.load(["longitude", "latitude"])
        print(scn["longitude"].shape)

varnames = {'MOD02QKM': 'EV_250_RefSB', 'MYD02QKM': 'EV_250_RefSB',
            'MOD02HKM': 'EV_500_RefSB', 'MYD02HKM': 'EV_500_RefSB',
            'MOD021KM': 'EV_1KM_RefSB', 'MYD021KM': 'EV_1KM_RefSB'}

    # 'MOD02HKM': 'EV_250_Aggr500_RefSB', 'MYD02HKM': 'EV_250_Aggr500_RefSB',
    # 'MOD021KM': 'EV_250_Aggr1km_RefSB', 'MYD021KM': 'EV_250_Aggr1km_RefSB',

print(hdf.datasets().keys())
print(hdf.select(varnames[iproduct])[:].shape)


print(hdf.select('Latitude')[:].shape)
lat = hdf.select('Latitude')[:]
lon = hdf.select('Longitude')[:]
print(lat.shape)
print(lon.shape)

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


