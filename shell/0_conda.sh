
# source miniconda3/bin/activate
# conda create --name lowclouds -y
# source miniconda3/bin/activate lowclouds

#--------------------------------
# conda clean -a
# conda env list
# conda env remove -n rcm_gbr -y
# conda config --show channels
# conda config --add channels conda-forge
# conda config --set auto_activate_base False   # remove (base) (base)

#-------------------------------- Primary packages
conda install mamba -y
mamba install python=3.12.7 -y
mamba install xesmf -y
mamba install python-cdo -y
mamba install matplotlib -y
mamba install cartopy -y
mamba install ipython -y
mamba install dask -y
mamba install intake -y
mamba install metpy -y
mamba install xmip -y
mamba install siphon -y
mamba install scikit-image -y
mamba install geopandas -y
mamba install rioxarray -y
mamba install mscorefonts -y



#-------------------------------- Other packages
# mamba install seaborn -y
# mamba install fortls -y
# mamba install cdsapi -y
# mamba install mpl-scatter-density -y
# mamba install pingouin -y
# mamba install windrose -y
# mamba install pycircstat -y
# mamba install numba -y
# pip install cdo
# mamba install xskillscore -y
# mamba install rasterio -y
# mamba install line_profiler -y
# mamba install haversine -y
# mamba install nose -y
# mamba install intake-esm -y
# mamba install intake-xarray -y
# mamba install dask-gateway -y
# mamba install xarrayutils -y
# mamba install pyhdf -y
# mamba install cfgrib -y
# mamba install geopy -y
# mamba install ipykernel -y
# mamba install ipywidgets -y
# mamba install satpy -y
# mamba install pywavelets -y
# mamba install pytables -y
# mamba install notebook -y
# mamba install pyfesom2 -y
# mamba install radian -y
# mamba install jupyterlab -y
# mamba install iris -y
# mamba install pytest -y
# pytest -v --pyargs xesmf
# pip install SkillMetrics
# pip install findent
# pip install f90nml
# pip install igra

