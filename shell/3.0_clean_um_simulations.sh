

output_dir='/home/563/qg8515/scratch/cylc-run'

# RNS
for expid in 'u-dq700' 'u-dq788' 'u-dq799' 'u-dq911' 'u-dq912' 'u-dq987' 'u-dr040' 'u-dr041' 'u-dr091' 'u-dr093' 'u-dr095' 'u-dr105' 'u-dr107' 'u-dr108' 'u-dr109' 'u-dr144' 'u-dr145' 'u-dr146' 'u-dr147' 'u-dr148' 'u-dr149' 'u-dr789' 'u-dr922' 'u-ds717' 'u-ds718' 'u-ds719' 'u-ds722' 'u-ds724' 'u-ds726' 'u-ds728' 'u-ds730' 'u-ds732' 'u-ds920' 'u-ds921' 'u-ds922' 'u-dt020' 'u-dt038' 'u-dt039' 'u-dt040' 'u-dt042'; do
    # expid='u-dt040'
    # RNS 'u-dq700' 'u-dq788' 'u-dq799' 'u-dq911' 'u-dq912' 'u-dq987' 'u-dr040' 'u-dr041' 'u-dr091' 'u-dr093' 'u-dr095' 'u-dr105' 'u-dr107' 'u-dr108' 'u-dr109' 'u-dr144' 'u-dr145' 'u-dr146' 'u-dr147' 'u-dr148' 'u-dr149' 'u-dr789' 'u-dr922' 'u-ds714' 'u-ds717' 'u-ds718' 'u-ds719' 'u-ds722' 'u-ds724' 'u-ds726' 'u-ds728' 'u-ds730' 'u-ds732' 'u-ds920' 'u-ds921' 'u-ds922' 'u-dt020' 'u-dt038' 'u-dt039' 'u-dt040' 'u-dt042'
    echo '#---------------- expid: ' ${expid}
    cd ${output_dir}/${expid}
    size=$(du -sh "${output_dir}/${expid}" | cut -f1)
    echo $size
    # du -sh share/cycle/*/*/*/*/*

    echo '#-------- clean /share'

    # du -sh share/cycle/*/ec
    rm -rf share/cycle/*/ec

    # du -sh share/cycle/*/Australia/*/*/ics
    rm -rf share/cycle/*/Australia/*/*/ics

    # du -sh share/cycle/*/Australia/*/*/lbcs
    rm -rf share/cycle/*/Australia/*/*/lbcs

    # du -sh share/cycle/*/Australia/*/*/um/umnsaa_cb*
    rm share/cycle/*/Australia/*/*/um/umnsaa_cb*

    # du -sh share/cycle/*/Australia/d11km/GAL9/um/*.nc
    rm share/cycle/*/Australia/d11km/GAL9/um/*.nc

    echo '#-------- clean /work'
    # du -sh work/*
    rm -rf work/*

    size0=$(du -sh "${output_dir}/${expid}" | cut -f1)
    echo size reduced from ${size} to ${size0}
done


# RAS
for expid in 'u-dq699' 'u-dq787' 'u-dq798' 'u-dq910' 'u-dq913' 'u-dr090' 'u-dr092' 'u-dr094' 'u-dr104' 'u-dr106' 'u-ds721' 'u-ds723' 'u-ds725' 'u-ds727' 'u-ds729' 'u-ds731'; do
    # expid='u-dq787'
    # RAS: 'u-dq699' 'u-dq787' 'u-dq798' 'u-dq910' 'u-dq913' 'u-dr090' 'u-dr092' 'u-dr094' 'u-dr104' 'u-dr106' 'u-ds721' 'u-ds723' 'u-ds725' 'u-ds727' 'u-ds729' 'u-ds731'
    echo '#---------------- expid: ' ${expid}
    cd ${output_dir}/${expid}
    size=$(du -sh "${output_dir}/${expid}" | cut -f1)
    # echo $size
    # du -sh share/*

    # echo '#-------- clean /share'

    # echo '#-------- clean /work'
    # du -sh work/*
    rm -rf work/*

    size0=$(du -sh "${output_dir}/${expid}" | cut -f1)
    echo size reduced from ${size} to ${size0}
done

