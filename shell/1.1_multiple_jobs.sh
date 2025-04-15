for idx in $(seq 1 12); do
    qsub -v idx=$idx <<EOF
#PBS -N qjob$idx
#PBS -q express
#PBS -l walltime=10:00:00
#PBS -l mem=192GB
#PBS -l jobfs=100MB
#PBS -l ncpus=1
#PBS -j oe
#PBS -l storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+gdata/zv2+gdata/ra22+gdata/py18
#PBS -P v46
#PBS -r y

cd ${HOME}

source ${HOME}/miniconda3/bin/activate lowclouds
python ${HOME}/code/gbr_future/shell/0_runpy/run${idx}.py
# python ${HOME}/code/gbr_future/shell/0_runpy/run1.py -y $idx

# module load python3/3.12.1
# /apps/python3/3.12.1/bin/python3 ${HOME}/code/gbr_future/shell/0_runpy/run${idx}.py

# ${HOME}/code/gbr_future/shell/2.1_check_jaxa.sh $idx

# Memory Used: 1.1GB; Walltime Used: 04:08:33
# sftp -ar gaoqg229ATgmail.com@www.cloudsat.cira.colostate.edu:Data/2B-CLDCLASS-LIDAR.P1_R05/${idx} /home/563/qg8515/scratch/data/obs/CloudSat_CALIPSO/2B-CLDCLASS-LIDAR.P1_R05

EOF
done


