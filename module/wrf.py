

# region get_era5_for_wrf

def get_era5_for_wrf(
    year_str, month_str_list, day_str_list,
    output_filename, area='10/125/-35/180',
    surface_only=True,
    format='grib', time_str_list=None, variable_name_list=None,
    pressure_level_list=None,):
    '''
    Download ERA5 data. Make sure you follow efficiency guidelines
    #---- Input
    year_str: string of the year. cannot be a list of years.
    month_str_list: list of strings of the months. e.g. ['08', '09']
    day_str_list:
    output_filename:
    surface_only: True: surface; False: pressure levels.
    area: lat/lon of the domain. Default is set to Australia. Set to None for global. upperLeft_Lat/upperLeft_Lon/lowerRight_Lat/lowerRight_Lon.
    format: 'netcdf' (default) or 'grib'
    time_str_list: hours strings in HH:MM format, default is full day hourly
    variable_name_list: list of variable name strings
    pressure_level_list: list of string of pressure levels
    '''
    
    import cdsapi
    
    client = cdsapi.Client()
    
    if pressure_level_list is None:
        pressure_level_list = [
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
    
    if variable_name_list is None:
        if surface_only:
            variable_name_list = [
                'surface_pressure',
                'mean_sea_level_pressure',
                '10m_u_component_of_wind',
                '10m_v_component_of_wind',
                '2m_temperature',
                'sea_surface_temperature',
                'skin_temperature',
                '2m_dewpoint_temperature',
                'snow_depth',
                'sea_ice_cover',
                'land_sea_mask',
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
            variable_name_list = [
                'geopotential',
                'relative_humidity',
                'specific_humidity',
                'temperature',
                'u_component_of_wind',
                'v_component_of_wind',
            ]
    
    if time_str_list is None:
        time_str_list = [
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
                             'variable':variable_name_list,
                             "product_type": "reanalysis",
                             'year': year_str,
                             'month': month_str_list,
                             'day': day_str_list,
                             'time': time_str_list,
                             "area": area,
                             "format": format
                         },
                         output_filename)
    else:
        client.retrieve("reanalysis-era5-pressure-levels",
                        {
                            'variable': variable_name_list,
                            'pressure_level': pressure_level_list,
                            "product_type": "reanalysis",
                            'year': year_str,
                            'month': month_str_list,
                            'day': day_str_list,
                            'time': time_str_list,
                            "area": area,
                            "format": format
                        },
                        output_filename)


# endregion