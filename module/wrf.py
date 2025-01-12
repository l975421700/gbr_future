

# region get_era5_for_wrf

def get_era5_for_wrf(
    year_str, month_strs, day_strs,
    output_filename, area='10/125/-35/180', surface_only=True,
    fmt='grib', time_strs=None, var_names=None, plevels=None,):
    '''
    year_str: string of the year. cannot be a list of years.
    area: lat/lon of the domain. Default is set to Australia. Set to None for global. upperLeft_Lat/upperLeft_Lon/lowerRight_Lat/lowerRight_Lon.
    surface_only: True: surface; False: pressure levels.
    fmt: 'netcdf' (default) or 'grib'
    '''
    import cdsapi
    client = cdsapi.Client()
    
    if plevels is None:
        plevels = [
            '1', '2', '3',
            '5', '7', '10',
            '20', '30', '50',
            '70', '100', '125',
            '150', '175', '200',
            '225', '250', '300',
            '350', '400', '450',
            '500', '550', '600',
            '650', '700', '750',
            '775', '800', '825',
            '850', '875', '900',
            '925', '950', '975',
            '1000',
        ]
    
    if var_names is None:
        if surface_only:
            var_names = [
                'surface_pressure',
                'sea_surface_temperature',
                'skin_temperature',
                'land_sea_mask',
                
                'mean_sea_level_pressure',
                '10m_u_component_of_wind',
                '10m_v_component_of_wind',
                '2m_temperature',
                '2m_dewpoint_temperature',
                'snow_depth',
                'sea_ice_cover',
                'soil_type',
                'soil_temperature_level_1',
                'soil_temperature_level_2',
                'soil_temperature_level_3',
                'soil_temperature_level_4',
                'volumetric_soil_water_layer_1',
                'volumetric_soil_water_layer_2',
                'volumetric_soil_water_layer_3',
                'volumetric_soil_water_layer_4',
            ]
        else:
            var_names = [
                'geopotential',
                'relative_humidity',
                'temperature',
                'u_component_of_wind',
                'v_component_of_wind',
                
                'specific_humidity',
                'vertical_velocity',
                'specific_cloud_ice_water_content',
                'specific_cloud_liquid_water_content',
                'specific_rain_water_content',
                'specific_snow_water_content',
                'fraction_of_cloud_cover',
                'ozone_mass_mixing_ratio',
            ]
    
    if time_strs is None:
        time_strs = [
            '00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00'
            ]
    
    if surface_only:
        client.retrieve('reanalysis-era5-single-levels',
                         {
                             'variable':var_names,
                             "product_type": "reanalysis",
                             'year': year_str,
                             'month': month_strs,
                             'day': day_strs,
                             'time': time_strs,
                             "area": area,
                             "format": fmt
                         },
                         output_filename)
    else:
        client.retrieve("reanalysis-era5-pressure-levels",
                        {
                            'variable': var_names,
                            'pressure_level': plevels,
                            "product_type": "reanalysis",
                            'year': year_str,
                            'month': month_strs,
                            'day': day_strs,
                            'time': time_strs,
                            "area": area,
                            "format": fmt
                        },
                        output_filename)



'''

'''
# endregion



