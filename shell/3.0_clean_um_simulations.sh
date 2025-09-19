

echo '#-------------------------------- clean model output'
# output_dir='/home/563/qg8515/scratch/cylc-run'
output_dir='/home/563/qg8515/scratch/cylc-run1'
echo '#---------------- folder: ' ${output_dir}


for expid in 'u-ds714' 'u-ds717' 'u-ds718'; do
    # expid='u-ds718'
    # RNS 'u-dq700' 'u-dq788' 'u-dq799' 'u-dq911' 'u-dq912' 'u-dq987' 'u-dr040' 'u-dr041' 'u-dr091' 'u-dr093' 'u-dr095' 'u-dr105' 'u-dr107' 'u-dr108' 'u-dr109' 'u-dr144' 'u-dr145' 'u-dr146' 'u-dr147' 'u-dr148' 'u-dr149' 'u-dr789' 'u-dr922' 'u-dr923'
    # RAS: 'u-dq699' 'u-dq787' 'u-dq798' 'u-dq910' 'u-dq913' 'u-dr090' 'u-dr092' 'u-dr094' 'u-dr104' 'u-dr106'
    echo '#-------- exp: ' ${expid}
    cd ${output_dir}/${expid}
    # du -sh ./*

    size=$(du -sh "${output_dir}/${expid}" | cut -f1)
    # echo $size

    echo '#---- clean share folder'

    # du -sh share/cycle/*/Australia/*/*/um/umnsaa_cb*
    rm share/cycle/*/Australia/*/*/um/umnsaa_cb*

    # du -sh share/cycle/*/ec
    rm -rf share/cycle/*/ec

    # du -sh share/cycle/*/Australia/*/*/ics/*
    rm share/cycle/*/Australia/*/*/ics/*

    # du -sh share/cycle/*/Australia/*/*/lbcs/*
    rm share/cycle/*/Australia/*/*/lbcs/*

    # echo '#---- clean work folder'
    # du -sh work/*
    # rm -rf work/*

    size0=$(du -sh "${output_dir}/${expid}" | cut -f1)
    echo size reduced from ${size} to ${size0}

done
