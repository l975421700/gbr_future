for idx in $(seq 8 10); do
    qsub -v idx=$idx <<EOF
#PBS -N qjob$idx
#PBS -q express
#PBS -l walltime=10:00:00
#PBS -l mem=192GB
#PBS -l jobfs=192MB
#PBS -l ncpus=48
#PBS -j oe
#PBS -l storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+gdata/zv2+gdata/ra22
#PBS -P v46
#PBS -r y

cd ${HOME}

# module load python3/3.12.1
# /apps/python3/3.12.1/bin/python3 ${HOME}/code/gbr_future/shell/0_runpy/run${idx}.py

source ${HOME}/miniconda3/bin/activate lowclouds
python ${HOME}/code/gbr_future/shell/0_runpy/run${idx}.py

# ${HOME}/code/gbr_future/shell/2.1_check_jaxa.sh $idx

EOF
done


