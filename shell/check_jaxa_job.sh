#!/bin/bash

for month in $(seq 1 12); do
    month=$(printf '%02d' $month)
    qsub -v month=$month <<EOF
#!/bin/bash
#PBS -N qjob_$month
#PBS -q copyq
#PBS -l walltime=10:00:00
#PBS -l mem=4GB
#PBS -l ncpus=1
#PBS -j oe
#PBS -l storage=gdata/v46
#PBS -P v46

wget -r -nH --cut-dirs=5 --continue --no-remove-listing --ftp-user='gaoqg229_gmail.com' --ftp-password='SP+wari8' ftp://ftp.ptree.jaxa.jp/pub/himawari/L2/CLP/010/2016\${month}/* -P /g/data/v46/qg8515/data/obs/jaxa/clp

EOF
done
