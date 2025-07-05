
year=$1
ids=$2 #2B-CWC-RO.P1_R05 #2B-GEOPROF-LIDAR.P2_R05 #2B-GEOPROF.P1_R05
echo $ids

sftp -ar gaoqg229ATgmail.com@www.cloudsat.cira.colostate.edu:Data/${ids}/${year} /home/563/qg8515/scratch/data/obs/CloudSat_CALIPSO/${ids}/

