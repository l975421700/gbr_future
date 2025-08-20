
# https://www.anaconda.com/docs/getting-started/miniconda/install
# source miniconda3/bin/activate
# conda create --name lowclouds -y
# source miniconda3/bin/activate lowclouds

#--------------------------------
# conda env list
# conda env remove -n lowclouds -y
# conda config --show channels
# conda config --add channels conda-forge
# conda config --set auto_activate_base False   # remove (base) (base)
# conda update -n base -c defaults conda

#-------------------------------- Primary packages
conda install mamba -y
mamba install python=3.12.7 -y
mamba install xarray=2024.10.0 -y
mamba install xesmf -y
mamba install python-cdo -y
mamba install matplotlib -y
mamba install cartopy -y
mamba install ipython -y
mamba install dask -y
mamba install intake -y
mamba install metpy -y
mamba install xmip -y
mamba install scikit-image -y
mamba install mscorefonts -y
mamba install ffmpeg -y
mamba install pyhdf -y
mamba install seaborn -y
mamba install haversine -y
mamba install zarr -y
mamba install ipykernel -y
mamba install h5netcdf -y
mamba install fortls -y
mamba install easygems -y
mamba install healpy -y
mamba install satpy -y
mamba install geopandas -y
conda clean -a -y -y -y -y -y

#-------------------------------- Other packages
# mamba install windrose -y
# mamba install pycircstat -y
# mamba install intake-xarray -y
# mamba install cfgrib -y
# mamba install statsmodels -y
# mamba install pingouin -y
# mamba install mpl-scatter-density -y
# mamba install iris -y
# mamba install cdsapi -y
# mamba install rioxarray -y
# mamba install siphon -y
# mamba install rasterio -y
# mamba install line_profiler -y
# mamba install notebook -y
# mamba install jupyterlab -y
# mamba install pytest -y
# pytest -v --pyargs xesmf

