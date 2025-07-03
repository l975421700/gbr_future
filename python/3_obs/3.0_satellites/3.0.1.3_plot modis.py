

# qsub -I -q normal -P v46 -l walltime=3:00:00,ncpus=1,mem=40GB,jobfs=100MB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22


# region import packages

# data analysis
import numpy as np
import glob
from datetime import datetime
from pyhdf.SD import SD, SDC
from satpy.scene import Scene

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
    )

from metplot import si2reflectance, si2radiance

# endregion


# region plot MODIS Terra and Aqua 1KM


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




