"""
This script:
    1. copy orignal MS from the directory of orignal_vis
    2. perform the channel-average operation on these copied files in parallel and specify the parameters avg_channel=[2,4,8,16,64]

   The above steps are aim to generate the integral visibility and then use CASA package to  check the accuracy and reconstructe the sky image
"""

from time import time
from dask.distributed import Client
from dask_arl.tool_function import arl_path
from dask.distributed import wait

import numpy
import dask.array as da
from casacore.tables import table, makearrcoldesc, maketabdesc
from os import cpu_count


def modify_ms(dst_ms, avg_channel=4, ack=False):
    """
    :param dst_ms:
    :param avg_channel:int
    :param ack: ack=False` prohibit the printing of a message telling if the table was opened or created successfully.
    :return: DOES NOT flag data
    """
    assert avg_channel >= 1
    assert isinstance(avg_channel, int)

    print('parameter avg_channel = % d' % avg_channel)
    print('loading data ......', end=' ')
    with table(dst_ms, ack=ack, readonly=False) as tar_tab:

        t1 = time()
        flag_dim = tar_tab.getdminfo(columnname="FLAG")
        # flagcategory_dim = res_tab.getdminfo(columnname="FLAG_CATEGORY")
        data_dim = tar_tab.getdminfo(columnname="DATA")
        model_dim = tar_tab.getdminfo(columnname="MODEL_DATA")
        corr_dim = tar_tab.getdminfo(columnname="CORRECTED_DATA")
        imgwg_dim = tar_tab.getdminfo(columnname="IMAGING_WEIGHT")

        # capture needed data
        weight = da.asarray(tar_tab.getcol('WEIGHT'))  # (nrow, npol)
        vis = da.asarray(tar_tab.getcol('DATA'))  # (nrow,allchan,npol)
        flag_data = da.asarray(tar_tab.getcol("FLAG"))  # (nrow,allchan,npol)
        # flag_category=da.asarray(res_tab.getcol('FLAG_CATEGORY'))  # Invalid operation, iscelldefined ====> False
        model_data = da.asarray(tar_tab.getcol("MODEL_DATA"))  # (nrow,allchan,npol)
        # corr_data= res_tab.getcol("CORRECTED_DATA")  # the column "CORRECTED_DATA" before calibration is equal to "DATA",
        imgwg_data = da.asarray(tar_tab.getcol("IMAGING_WEIGHT"))  # (nrow,allchan)
        t2 = time()
        print('time consuming %.3fs' % (t2 - t1))

        allchan = vis.shape[1]
        assert avg_channel <= allchan
        count = allchan // avg_channel
        last = allchan % avg_channel
        start = 0
        i = 0
        exp_weight = weight[:, None, :]

        cat_vis = []
        cat_flag = []
        cat_model = []
        cat_imwg = []

        if last:
            print("\nWARNING: the number of channels(=%d) should be divided by avg_channels(=%d). "
                  "All channels' weight are stored in same " % (allchan, avg_channel))
            count = count + 1

        # loop event: 0.9s
        print('loop event ......', end=' ')
        t3 = time()
        write_weight = da.sum(exp_weight * da.ones(shape=(avg_channel, 1)), axis=-2)  # shape = (nrow,npol)
        while i < count:
            part_flag = flag_data[:, start, :]  # extract from first channel corresponding to each several continual channels
            part_model = model_data[:, start, :]
            part_imwg = imgwg_data[:, start:start + avg_channel]
            part_vis = vis[:, start:start + avg_channel, :]  # shape=(nrow, avg_channel, npol)

            sum_imwg = da.sum(part_imwg, axis=1)
            sum_vis = da.sum(part_vis * exp_weight, axis=-2)

            if i == count - 1 and last > 0:
                avg_vis = sum_vis / (weight * last + 1e-10)
            else:
                avg_vis = sum_vis / (write_weight + 1e-10)

            cat_vis.append(avg_vis)
            cat_flag.append(part_flag)
            cat_model.append(part_model)
            cat_imwg.append(sum_imwg)

            i = i + 1
            start = start + avg_channel

        write_vis = da.stack(cat_vis, axis=1)
        write_flag = da.stack(cat_flag, axis=1)
        write_model = da.stack(cat_model, axis=1)
        write_imwg = da.stack(cat_imwg, axis=1)

        t4 = time()
        print('time consuming %.3fs' % (t4 - t3))

        print('dask computing ......', end=' ')
        write_vis, write_flag, write_model, write_weight, write_imwg = da.compute(write_vis, write_flag, write_model, write_weight, write_imwg)
        write_corr = write_vis
        t5 = time()
        print('time consuming %.3fs' % (t5 - t4))

        print('modifying MAIN table......', end=' ')
        tar_tab.removecols(columnnames=["FLAG", "DATA", "MODEL_DATA", "CORRECTED_DATA", "IMAGING_WEIGHT"])  # "FLAG_CATEGORY",
        tar_tab.addcols(maketabdesc(makearrcoldesc(columnname="FLAG", value=False, ndim=2, datamanagertype='TiledShapeStMan',
                                                   datamanagergroup='TiledFlag', valuetype='boolean',
                                                   comment='The data flags, array of bools with same shape as data',
                                                   keywords={})),
                        flag_dim)

        tar_tab.addcols(maketabdesc(makearrcoldesc(columnname="DATA", value=0.0j, ndim=2, datamanagertype='TiledShapeStMan',
                                                   datamanagergroup='TiledDATA', valuetype='complex',
                                                   comment='The data column',
                                                   keywords={})),
                        data_dim)

        tar_tab.addcols(maketabdesc(makearrcoldesc(columnname="MODEL_DATA", value=0.0j, ndim=2, datamanagertype='TiledShapeStMan',
                                                   datamanagergroup='TiledMODEL_DATA', valuetype='complex',
                                                   comment='The model data column',
                                                   keywords={})),
                        model_dim)

        tar_tab.addcols(maketabdesc(makearrcoldesc(columnname="CORRECTED_DATA", value=0.0j, ndim=2, datamanagertype='TiledShapeStMan',
                                                   datamanagergroup='TiledCORRECTED_DATA', valuetype='complex',
                                                   comment='The corrected data column',
                                                   keywords={})),
                        corr_dim)

        tar_tab.addcols(maketabdesc(makearrcoldesc(columnname="IMAGING_WEIGHT", value=0.0, ndim=1, datamanagertype='TiledShapeStMan',
                                                   datamanagergroup='TiledImagingWeight', valuetype='float',
                                                   comment='Weight set by imaging task (e.g. uniform weighting)',
                                                   keywords={})),
                        imgwg_dim)

        tar_tab.putcol(columnname="FLAG", value=write_flag)
        tar_tab.putcol(columnname="WEIGHT", value=write_weight)
        tar_tab.putcol(columnname="DATA", value=write_vis)
        tar_tab.putcol(columnname="MODEL_DATA", value=write_model)
        tar_tab.putcol(columnname="CORRECTED_DATA", value=write_corr)
        tar_tab.putcol(columnname="IMAGING_WEIGHT", value=write_imwg)

        t6 = time()
    print('Done, time consuming %.3fs' % (t6 - t5))

    print('Modifying SPECTRAL_WINDOW table......', end=' ')
    with table('%s/SPECTRAL_WINDOW' % dst_ms, ack=False, readonly=False) as tar_spwtab:
        chanfreq_dim = tar_spwtab.getdminfo(columnname="CHAN_FREQ")
        chanfreq_dim["NAME"] = 'StandardStManCHAN_FREQ'

        chanwidth_dim = tar_spwtab.getdminfo(columnname="CHAN_WIDTH")
        chanwidth_dim["NAME"] = 'StandardStManCHAN_WIDTH'

        effectivebw_dim = tar_spwtab.getdminfo(columnname="EFFECTIVE_BW")
        effectivebw_dim["NAME"] = 'StandardStManEFFECTIVE_BW'

        resolution_dim = tar_spwtab.getdminfo(columnname="RESOLUTION")
        resolution_dim["NAME"] = 'StandardStManRESOLUTION'

        frequency = tar_spwtab.getcol('CHAN_FREQ')
        channel_bandwidth = tar_spwtab.getcol('CHAN_WIDTH')
        nspw = frequency.shape[0]

        j = 0
        start = 0
        cat_chanfreq = []
        cat_bandwidth = []

        write_ref = numpy.average(frequency[:, 0:avg_channel], axis=1)  # ref_frequency is equal to the array chan_freq[:,0] (the first column)
        # Use Numpy instead of dask.array to process small amounts of data
        while j < count:
            avg_chanfreq = numpy.average(frequency[:, start:start + avg_channel], axis=1)
            sum_chan = numpy.sum(channel_bandwidth[:, start:start + avg_channel], axis=1)
            cat_chanfreq.append(avg_chanfreq)
            cat_bandwidth.append(sum_chan)

            j = j + 1
            start = start + avg_channel

        write_chanfreq = numpy.stack(cat_chanfreq, axis=1)
        write_bandwidth = numpy.stack(cat_bandwidth, axis=1)

        tar_spwtab.removecols(["CHAN_FREQ", "CHAN_WIDTH", "EFFECTIVE_BW", "RESOLUTION"])
        tar_spwtab.addcols(maketabdesc(makearrcoldesc(columnname="CHAN_FREQ", value=0.0, ndim=1, datamanagertype='StandardStMan',
                                                      datamanagergroup='StandardStMan', valuetype='double',
                                                      comment='Center frequencies for each channel in the data matrix',
                                                      keywords={'QuantumUnits': ['Hz'],
                                                                'MEASINFO': {'type': 'frequency',
                                                                             'VarRefCol': 'MEAS_FREQ_REF',
                                                                             'TabRefTypes': ['REST', 'LSRK', 'LSRD', 'BARY', 'GEO', 'TOPO', 'GALACTO',
                                                                                             'LGROUP', 'CMB'],
                                                                             'TabRefCodes': [0, 1, 2, 3, 4, 5, 6, 7, 8]}
                                                                })),
                           chanfreq_dim)

        tar_spwtab.addcols(maketabdesc(makearrcoldesc(columnname="CHAN_WIDTH", value=0.0, ndim=1, datamanagertype='StandardStMan',
                                                      datamanagergroup='StandardStManCHAN_WIDTH', valuetype='double',
                                                      comment='Channel width for each channel',
                                                      keywords={'QuantumUnits': ['Hz']})),
                           chanwidth_dim)

        tar_spwtab.addcols(maketabdesc(makearrcoldesc(columnname="EFFECTIVE_BW", value=0.0, ndim=1, datamanagertype='StandardStMan',
                                                      datamanagergroup='StandardStManEFFECTIVE_BW', valuetype='double',
                                                      comment='Effective noise bandwidth of each channel',
                                                      keywords={'QuantumUnits': ['Hz']})),
                           effectivebw_dim)

        tar_spwtab.addcols(maketabdesc(makearrcoldesc(columnname="RESOLUTION", value=0.0, ndim=1, datamanagertype='StandardStMan',
                                                      datamanagergroup='StandardStManRESOLUTION', valuetype='double',
                                                      comment='The effective noise bandwidth for each channel',
                                                      keywords={'QuantumUnits': ['Hz', ]})),
                           resolution_dim)

        tar_spwtab.putcol(columnname="CHAN_FREQ", value=write_chanfreq)
        tar_spwtab.putcol(columnname="REF_FREQUENCY", value=write_ref)
        tar_spwtab.putcol(columnname="CHAN_WIDTH", value=write_bandwidth)
        tar_spwtab.putcol(columnname="EFFECTIVE_BW", value=write_bandwidth)
        tar_spwtab.putcol(columnname="RESOLUTION", value=write_bandwidth)
        tar_spwtab.putcol(columnname="NUM_CHAN", value=[count] * nspw)
        t7 = time()
        print('Done, time consuming %.3fs' % (t7 - t6))

    print('Finish operating channel-average on MS')


if __name__ == '__main__':
    c = Client('172.31.99.84:8786')
    """
    This program can ONLY be run once, 
    if an exception occurs you need to delete the files in the modify_vis directory,
    and re-copy the source files to that directory, then run again.
    """
    print('cpu_count: %d' % cpu_count())
    s1 = time()

    avg_channel = [2, 4, 8, 16, 32, 64]
    lenght = len(avg_channel)

    dst = arl_path('source_data/modify_vis/day2_copy_avgchannel')
    dsts = [dst + str(avg) for avg in avg_channel]

    avg_ms = c.map(modify_ms, dsts, avg_channel)
    wait(avg_ms)

    s2 = time()
    print('Total time: %.3fs' % (s2 - s1))
