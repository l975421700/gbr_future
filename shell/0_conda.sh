
# source miniconda3/bin/activate
# conda create --name lowclouds -y
# source miniconda3/bin/activate lowclouds
# conda config --add channels conda-forge

#-------------------------------- Primary packages
conda install mamba -y
mamba install numba -y
mamba install xesmf -y
mamba install metpy -y
mamba install matplotlib -y
mamba install dask -y
mamba install cartopy -y
mamba install cdo -y
mamba install seaborn -y
mamba install cdsapi -y
mamba install windrose -y
mamba install ffmpeg -y
mamba install geopandas -y
mamba install fortls -y
mamba install mpl-scatter-density -y
mamba install intake -y
mamba install pycircstat -y
mamba install pingouin -y
mamba install ipython -y
mamba install ncview -y
mamba install xmip -y
mamba install siphon -y
mamba install scikit-image -y

#-------------------------------- Secondary packages
# mamba install mscorefonts -y
# mamba install xskillscore -y
# mamba install rasterio -y
# mamba install rioxarray -y
# mamba install line_profiler -y
# mamba install haversine -y
# mamba install nose -y
# mamba install openpyxl -y
# mamba install Pympler -y
# mamba install intake-esm -y
# mamba install intake-xarray -y
# mamba install dask-gateway -y
# mamba install xarrayutils -y
# mamba install xgcm -y
# mamba install gcsfs -y
# mamba install python-cdo -y
# mamba install numexpr -y
# mamba install pyhdf -y
# mamba install cfgrib -y
# mamba install geopy -y
# mamba install pytest -y
# mamba install pint-xarray -y
# mamba install ipykernel -y
# mamba install ipywidgets -y
# mamba install wget -y
# mamba install satpy -y
# mamba install libgdal-hdf5 -y
# mamba install gdal -y
# mamba install jupyter -y
# mamba install pywavelets -y
# mamba install pytables -y
# mamba install notebook -y
# mamba install pyfesom2 -y
# mamba install radian -y
# mamba install jupyterlab -y
# mamba install iris -y

# pip install SkillMetrics
# pip install findent
# pip install f90nml
# pip install igra

# pytest -v --pyargs xesmf


#--------------------------------
# conda clean -a
# conda env list
# conda env remove -n rcm_gbr
# conda config --show channels

# conda config --set auto_activate_base False   # remove (base) (base)
