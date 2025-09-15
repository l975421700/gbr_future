

echo '#-------------------------------- clean model output'
output_dir='/home/563/qg8515/scratch/cylc-run'
echo '#---------------- folder: ' ${output_dir}


for expid in 'u-dr789' 'u-dr922'; do
    # 'u-dr108' 'u-dr109', 'u-dr144' 'u-dr145' 'u-dr146' 'u-dr147' 'u-dr148' 'u-dr149', 'u-dq700' 'u-dq788' 'u-dq799' 'u-dq911' 'u-dq912' 'u-dq987' 'u-dr040' 'u-dr041' 'u-dr091' 'u-dr093' 'u-dr095' 'u-dr105' 'u-dr107'
    echo '#-------- exp: ' ${expid}
    cd ${output_dir}/${expid}
    # pwd

    size=$(du -sh "${output_dir}/${expid}" | cut -f1)
    # echo $size
    # du -sh *

    echo '#---- clean share folder'

    # du -sh share/cycle/*/Australia/*/*/um/umnsaa_cb*
    rm share/cycle/*/Australia/*/*/um/umnsaa_cb*

    # du -sh share/cycle/*/ec
    rm -rf share/cycle/*/ec

    # du -sh share/cycle/*/Australia/*/*/ics/*
    rm share/cycle/*/Australia/*/*/ics/*

    # du -sh share/cycle/*/Australia/*/*/lbcs/*
    rm share/cycle/*/Australia/*/*/lbcs/*

    echo '#---- clean work folder'

    # du -sh work/*
    rm -rf work/*

    size0=$(du -sh "${output_dir}/${expid}" | cut -f1)
    echo size reduced from ${size} to ${size0}

done
