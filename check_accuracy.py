import dask.bag as db
from os import cpu_count
from time import time

import numpy

from dask_arl.dask_average import load_msdata, time_average, channel_average
from dask_arl.tool_function import arl_path


def pick_source(casa, tag):
    casa_vis = load_msdata(casa)
    for vis in casa_vis:
        if vis.tag == tag:
            return [vis]


def delt_error(x, y):
    # check the vis variable
    delt = x.vis[0:7] - y.vis[0:7]
    real_delt = numpy.real(delt)
    imag_delt = numpy.imag(delt)
    return real_delt.mean(), imag_delt.mean(), real_delt.var(), imag_delt.var()


if __name__ == '__main__':
    print('cpu_count: %d' % cpu_count())

    s1 = time()
    avg_channel = [2, 4, 8, 16, 32, 64]
    length = len(avg_channel)
    casa_fmt = arl_path('source_data/CASA_vis/casaavg_time2_chan%d.ms')

    source_ms = arl_path('source_data/orignal_vis/day2_TDEM0003_10s_norx.0')
    casa_ms = db.from_sequence([casa_fmt % avg for avg in avg_channel])
    casa_data = casa_ms.map(load_msdata).flatten().filter(lambda x: x.tag == '0_7').compute()

    # average 0_7-source

    load = pick_source(source_ms, tag='0_7')
    time_avg = time_average(load, avg_time=2)  # equal to the split(timebin='20s') task in CASA

    avg_list = []
    for nch in avg_channel:
        tuple_res = channel_average(time_avg, nch)
        for res in tuple_res:
            avg_list.append(res)

    delt_result = []

    for x, y in zip(casa_data, avg_list):
        # print(x.sizes['nchan'])
        delt_result.append(delt_error(x, y))
    print(numpy.array(delt_result))

    s2 = time()

    print('check_accuracy operation costs %.2fs' % (s2 - s1))

'''
            rmean           imean           rval            ivar        
avg=2    [[-1.06102065e-10 -7.45795353e-13  7.50809805e-20  4.88266427e-20]
avg=4     [-7.26796540e-11 -1.04491925e-12  6.46637458e-20  5.43451200e-20]
avg=8     [-5.97330033e-11 -5.30443582e-13  7.17755927e-20  6.50234660e-20]
avg=16    [-5.39712545e-11 -1.49234162e-12  1.00684631e-19  9.15967142e-20]
avg=32    [-4.87273243e-11  2.35992785e-13  1.61991549e-19  1.44317077e-19]
avg=64    [-5.61921412e-11 -8.21669989e-12  3.07005138e-19  2.60739737e-19]]


'''
