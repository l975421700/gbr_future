
# source ~/miniconda3/bin/activate
# conda create --name rcm_gbr
# source ~/miniconda3/bin/activate rcm_gbr
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
mamba install haversine -y
mamba install windrose -y
mamba install ffmpeg -y
mamba install mscorefonts -y
mamba install line_profiler -y
mamba install rasterio -y
mamba install rioxarray -y
mamba install geopandas -y
mamba install pymannkendall -y
mamba install fortls -y
mamba install nose -y
mamba install cmip6_preprocessing -y
mamba install xskillscore -y
mamba install eofs -y
mamba install lazy_loader -y
mamba install siphon -y
mamba install mpl-scatter-density -y
mamba install intake -y
mamba install intake-xarray -y
mamba install dask-gateway -y
mamba install xarrayutils -y
mamba install xgcm -y
mamba install gcsfs -y
mamba install python-cdo -y
mamba install numexpr -y
mamba install pyhdf -y
mamba install cfgrib -y
mamba install geopy -y
mamba install pycircstat -y
mamba install openpyxl -y
mamba install Pympler -y
mamba install pingouin -y
mamba install intake-esm -y
mamba install ipython -y
mamba install pytest -y
mamba install pint-xarray -y
mamba install ipykernel -y
mamba install ipywidgets -y
mamba install wget -y
mamba install ncview -y

pip install SkillMetrics
pip install igra
pip install pint-xarray
pip install findent
pip install f90nml

#-------------------------------- Secondary packages

# pip install pycircstat failed

# mamba install jupyter -y
# mamba install pywavelets -y
# mamba install pytables -y
# mamba install notebook -y
# mamba install pyfesom2 -y
# mamba install radian -y
# mamba install jupyterlab -y
# mamba install iris -y
# pip install siphon

# pytest -v --pyargs xesmf
# conda clean -a


#--------------------------------
# conda env list
# conda env remove -n rcm_gbr
# conda config --show channels

# conda create --prefix /g/data/v46/qg8515/conda_envs/test
# conda activate /g/data/v46/qg8515/conda_envs/test
# conda config --set auto_activate_base False   # remove (base) (base)
