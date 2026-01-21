for idx in $(seq 1 24); do
    qsub -v idx=$idx <<EOF
#PBS -N qjob$idx
#PBS -q normal
#PBS -l walltime=2:00:00
#PBS -l mem=24GB
#PBS -l jobfs=100MB
#PBS -l ncpus=1
#PBS -j oe
#PBS -l storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22+gdata/py18+gdata/gx60+gdata/xp65+gdata/qx55+gdata/rv74+gdata/al33+gdata/rr3
#PBS -P v46
#PBS -r y

cd ${HOME}

source miniconda3/bin/activate lowclouds
python code/gbr_future/shell/0_runpy/run${idx}.py

# code/gbr_future/shell/2.2_get_cloudsat.sh $idx 2B-GEOPROF.P1_R05 #2B-CWC-RO.P1_R05 #2B-GEOPROF-LIDAR.P2_R05
# python code/gbr_future/shell/0_runpy/run1.py -y $idx
# code/gbr_future/shell/2.1_check_jaxa.sh $idx
# sftp -ar gaoqg229ATgmail.com@www.cloudsat.cira.colostate.edu:Data/2B-CLDCLASS-LIDAR.P1_R05/${idx} scratch/data/obs/CloudSat_CALIPSO/2B-CLDCLASS-LIDAR.P1_R05 # Memory Used: 1.1GB; Walltime Used: 04:08:33
EOF
done


