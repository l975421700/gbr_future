#!/bin/bash

for day in {01..30}; do
    qsub -v day=$day <<EOF
#!/bin/bash
#PBS -N qjob_$day
#PBS -q copyq
#PBS -l walltime=10:00:00
#PBS -l mem=4GB
#PBS -l jobfs=4GB
#PBS -l ncpus=1
#PBS -j oe
#PBS -l storage=gdata/v46
#PBS -P v46
#PBS -r y

wget -r -nH --cut-dirs=5 --continue --no-remove-listing --ftp-user='gaoqg229_gmail.com' --ftp-password='SP+wari8' ftp://ftp.ptree.jaxa.jp/pub/himawari/L2/CLP/010/201601/${day}/* -P /home/563/qg8515/data/obs/jaxa/clp

EOF
done

# qsub -I -q copyq -l walltime=10:00:00,ncpus=1,mem=4GB,jobfs=4GB,storage=gdata/v46
# wget -r -nH --cut-dirs=5 --continue --no-remove-listing --ftp-user='gaoqg229_gmail.com' --ftp-password='SP+wari8' ftp://ftp.ptree.jaxa.jp/pub/himawari/L2/CLP/010/201601/08/* -P /g/data/v46/qg8515/data/obs/jaxa/clp

# lftp -u leongjiawen_outlook.com,SP+wari8 ftp://ftp.ptree.jaxa.jp <<EOF
# mirror --continue --no-empty-dirs /pub/himawari/L2/CLP/010/201601/08 /g/data/v46/qg8515/data/obs/jaxa/clp
# EOF

