import time as t
from shutil import copytree
from dask.distributed import Client
from dask.distributed import wait

from dask_arl.tool_function import arl_path

if __name__ == '__main__':
    c = Client('172.31.99.84:8786')
    total = 20
    avg_channel = [2, 4, 8, 16, 32, 64]  # 5 MS

    """
    It will copy 20+5 MS files, which occupies considerable disk space (about 25*2=50GB) 
    if an exception occurs you need to delete the output files in the modify_vis/ and orignal_vis/ directories,
    and then run again.
    """

    length = len(avg_channel)

    t1 = t.time()
    orignal = arl_path('source_data/day2_TDEM0003_10s_norx')
    ori_dirs = [orignal for i in range(total)]

    msname = arl_path('source_data/orignal_vis/day2_TDEM0003_10s_norx.%d')
    copy_dirs = [msname % i for i in range(total)]

    avg_orignal = [orignal for i in range(length)]

    avg_name = arl_path('source_data/modify_vis/day2_copy_avgchannel')
    avg_dsts = [avg_name + str(avg) for avg in avg_channel]

    print('Wait a moment ......')  # about 8 minutes depending on machine performance
    print('Multiple MS (%d MS files) are being copied....' % (total + length), end=' ')
    copy_ms = c.map(copytree, ori_dirs, copy_dirs)
    dst = c.map(copytree, avg_orignal, avg_dsts)

    wait(copy_ms)
    wait(dst)
    t2 = t.time()
    print('time consuming %.3fs' % (t2 - t1))
