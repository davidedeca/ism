import numpy as np
import os

# list output times of a simulation, looking at the output files
def output_time(nout, path='.'):

    path_output = os.path.join(path, 'output_'+str(nout).zfill(5))
    assert os.path.isdir(path_output)

    path_info = os.path.join(path_output, 'info_'+str(nout).zfill(5)+'.txt')
    info = open(path_info).read()
    i1 = info.find('time')
    i1 = info.find('=', i1)
    i1 += 2
    i2 = info.find('aexp', i1)
    return float(info[i1+1:i2])


def output_times(path='.'):

    assert os.path.isdir(path + '/output_00001'), 'output_00001 folder not found'

    nout = 0
    times=[]

    for filename in os.listdir(path):
        if os.path.isdir(path + '/' + filename) and 'output_' in filename:
            nout = int(filename[-5:])
            times.append(output_time(nout, path))

    times = np.array(times)
    times.sort()
    return times


# list computation time, looking at the log file

def cpu_times(path='.', logfile='log', step='N', walltime='timestep', unit='s'):

    assert time in ['N', 'time']
    assert walltime in ['timestep', 'totaltime', 'worktime']

    assert os.path.isfile(path + '/' + logfile), 'log file not found'

    string1 = 'Time elapsed since last coarse step:'
    string2 = 'Total running time:'
    string3 = ' t='
    len1 = len(string1)
    len2 = len(string2)
    len3 = len(string3)

    timestep  = np.array([])
    totaltime = np.array([])
    timesim   = np.array([])
    count = 0
    i1 = 0
    i2 = 0
    i3 = 0

    log = open(path + '/' + logfile).read()

    while (i1 != -1):
        i1 = log.find(string1, i1 + len1)
        i2 = log.find(string2, i2 + len2)
        i3 = log.find(string3, i3 + len3)
        if (i1 != -1):
            timestep  = np.append(timestep, float(log[i1+len1:i1+len1+8]))
            totaltime = np.append(totaltime, float(log[i2+len2:i2+len2+11]))
            timesim   = np.append(timesim, float(log[i3+len3:i3+len3+12]))
            count += 1

    worktime = totaltime - totaltime[0]
    nsteps = range(1, count+1)

    if step == 'N':
        x = nsteps
    elif step == 'time':
        x = timesim

    if walltime == 'timestep':
        y = timestep
    elif walltime == 'totaltime':
        y = totaltime
    elif walltime == 'worktime':  #remove setup time from total time
        y = worktime

    if unit == 'h':
        y /= 3600.

    return x, y
