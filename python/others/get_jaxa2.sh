
syear=$1
eyear=$2
smonth=$3
emonth=$4
sday=$5
eday=$6

ftp_server='ftp.ptree.jaxa.jp'
ftp_user='gaoqg229_gmail.com'
ftp_password='SP+wari8'

local_path=/home/563/qg8515/data/obs/jaxa/clp
himawari=08
parallel_jobs=1

generate_download_tasks() {
    for year in $(seq $syear $eyear); do
        for month in $(seq $smonth $emonth); do
            month=$(printf '%02d' $month)
            for day in $(seq $sday $eday); do
                day=$(printf '%02d' $day)
                day_path=$local_path/$year/$month/$day
                ftp_path=/pub/himawari/L2/CLP/010/$year$month/$day/*
                # echo 'Date:' $year $month $day
            done
        done
    done
}

download_file() {
    ftp_path=$1
    day_path=$2
    echo $ftp_path $day_path
    mkdir -p $day_path
    wget -r -nH --cut-dirs=100 --continue --no-remove-listing --ftp-user=$ftp_user --ftp-password=$ftp_password ftp://$ftp_server$ftp_path -P $day_path
    if ! [[ $? -eq 0 ]]; then
        echo 'Failed to download: $ftp_path' >&2
    fi
}

export -f download_file
export ftp_server ftp_user ftp_password himawari local_path
export syear eyear smonth emonth sday eday

generate_download_tasks | xargs -n 2 -P $parallel_jobs bash -c 'download_file "$@"' _

echo 'Download completed.'

# wget -r -nH --cut-dirs=5 --continue --no-remove-listing --ftp-user='gaoqg229_gmail.com' --ftp-password='SP+wari8' ftp://ftp.ptree.jaxa.jp/pub/himawari/L2/CLP/010/201612/01/00/NC_H08_20161201_0010_L2CLP010_FLDK.02401_02401.nc -P /home/563/qg8515/data/others
# wget -r -nH --cut-dirs=5 --continue --no-remove-listing --ftp-user='gaoqg229_gmail.com' --ftp-password='SP+wari8' ftp://ftp.ptree.jaxa.jp/pub/himawari/L2/CLP/010/* -P /Users/gao/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/research/data/obs/jaxa/clp

