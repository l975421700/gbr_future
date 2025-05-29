
import glob
qjobo_fl = sorted(glob.glob('/home/563/qg8515/data/others/run/*'))
for ifile in qjobo_fl:
    with open(ifile, 'r') as file:
        content = file.read()
        if ('Exit Status:        0' in content):
            pass
        elif not ('Exit Status:        ' in content):
            pass
        else:
            print(f'#-------------------------------- Warning')
            print(f'Not success: {ifile}')

len(sorted(glob.glob('/home/563/qg8515/scratch/data/sim/um/barra_r2/wap/wap_monthly_????09.nc')))


