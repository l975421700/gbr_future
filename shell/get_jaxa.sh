#!/bin/bash

year=$1
month=$2
hour=$3

echo "Start processing: Year $year, Month $month, Hour $hour"

if [ $month -eq 01 ]; then
    ftp_user='gaoqg229_gmail.com'
elif [ $month -eq 02 ]; then
    ftp_user='gaoqg000_gmail.com'
elif [ $month -eq 03 ]; then
    ftp_user='gaoqgaaa_gmail.com'
elif [ $month -eq 04 ]; then
    ftp_user='gaoqgbbb_gmail.com'
elif [ $month -eq 05 ]; then
    ftp_user='gaoqgccc_gmail.com'
elif [ $month -eq 06 ]; then
    ftp_user='qinggang.gao_unimelb.edu.au'
elif [ $month -eq 07 ]; then
    ftp_user='qg229_cam.ac.uk'
elif [ $month -eq 08 ]; then
    ftp_user='qinggang.gao_awi.de'
elif [ $month -eq 09 ]; then
    ftp_user='gaoqgclub_gmail.com'
elif [ $month -eq 10 ]; then
    ftp_user='jiawenibgdrgn_gmail.com'
elif [ $month -eq 11 ]; then
    ftp_user='leongjiawen_icloud.com'
elif [ $month -eq 12 ]; then
    ftp_user='leongjiawen_outlook.com'
fi

ftp_server='ftp.ptree.jaxa.jp'
ftp_password='SP+wari8'

for day in $(seq 1 31); do
    day=$(printf '%02d' $day)
    echo 'Day' $day

    ftp_path=/pub/himawari/L2/CLP/010/$year$month/$day/$hour/*
    local_path=/g/data/v46/qg8515/data/obs/jaxa/clp

    wget -r -nH --cut-dirs=5 --continue --no-remove-listing --ftp-user=$ftp_user --ftp-password=$ftp_password ftp://$ftp_server$ftp_path -P $local_path
done
