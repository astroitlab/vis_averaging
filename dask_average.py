import time as t
import dask
import numpy
import gc
import os
import functools
import xarray as xr
import dask.bag as db
from astropy import units as u
from astropy.coordinates import SkyCoord
from dask import delayed
from dask.distributed import Client, wait
from xarray import Dataset
from casacore.tables import table
from os import cpu_count


def timer(func):
    """Calculate the execution time of a function
    :param func:
    :return:
    """

    @functools.wraps(func)
    def wrapper(*args, **kw):
        start = t.time()
        res = func(*args, **kw)
        end = t.time()
        print('Method:%s(...) costs %.2fs\n' % (func.__name__, end - start))
        return res

    return wrapper


def configuration_init(ms_file) -> Dataset:
    with table('%s/ANTENNA' % ms_file, ack=False) as anttab:
        mount = anttab.getcol('MOUNT')
        names = anttab.getcol('NAME')
        diameter = anttab.getcol('DISH_DIAMETER')
        xyz = anttab.getcol('POSITION')

        assert xyz.shape[1] == 3
        nants = xyz.shape[0]

        # nspos: antenna's spatial position(x,y,z)
        cf = xr.Dataset({'xyz': (['nant', 'nspos'], xyz),
                         'mount': (['nant'], mount),
                         'names': (['nant'], names),
                         'diameter': (['nant'], diameter)},
                        coords={'nant': numpy.arange(nants),
                                'nspos': ['x', 'y', 'z']})

        return cf


def phasecentre_init(ms_file, field):
    with table('%s/FIELD' % ms_file, ack=False) as phase_tab:
        pc = phase_tab.getcol('PHASE_DIR')[field, 0, :]  # [2.59594627 0.30936554]
        phasecentre = SkyCoord(ra=pc[0] * u.rad,
                               dec=pc[1] * u.rad,
                               frame='icrs',
                               equinox='J2000')
    return phasecentre


def polarisation_init(ms_file):
    with table('%s/POLARIZATION' % ms_file, ack=False) as poltab:
        corr_type = poltab.getcol('CORR_TYPE')

        return corr_type


def frequency_init(ms_file, dd) -> Dataset:
    with table('%s/SPECTRAL_WINDOW' % ms_file, ack=False) as spwtab:
        frequency = spwtab.getcol('CHAN_FREQ')[dd]
        channel_bandwidth = spwtab.getcol('CHAN_WIDTH')[dd]

        assert len(frequency) == len(channel_bandwidth)

        nchan = frequency.shape[0]

        fre_band = xr.Dataset(data_vars={'frequency': (['nchan'], frequency),
                                         'channel_bandwidth': (['nchan'], channel_bandwidth)},
                              coords={'nchan': numpy.arange(nchan)})

        return fre_band


def msdataset_init(query_ms, fre_band, phasecentre, configuration, corr_type, ms_tag) -> Dataset:
    time = query_ms.getcol('TIME')
    antenna1 = query_ms.getcol('ANTENNA1')
    antenna2 = query_ms.getcol('ANTENNA2')
    assert numpy.min(antenna2 - antenna1) > 0, 'ANTENNA1-ANTENNA2 are not sorted'

    nrow = query_ms.nrows()
    nants = configuration.sizes['nant']
    nbaseline = nants * (nants - 1) // 2

    baseline = []  # 'a1-a2'
    for a1 in range(nants - 1):
        for a2 in range(a1 + 1, nants):
            baseline.append('-'.join([str(a1), str(a2)]))

    utime, ucount = numpy.unique(time, return_counts=True)
    ntime = utime.shape[0]
    qmax = numpy.max(antenna2)
    baseline_index = (2 * qmax - antenna1 + 1) * antenna1 // 2 + antenna2 - antenna1 - 1  # Note: int type

    miss = ntime * nbaseline - nrow

    vis = query_ms.getcol('DATA')
    weight = query_ms.getcol('WEIGHT')
    uvw = query_ms.getcol('UVW')  # UVW is in units of the metre not wavelength

    assert uvw.shape[1] == 3
    _, nchan, npol = vis.shape
    assert len(fre_band.nchan) == nchan
    p_coords = ['p%d' % i for i in range(npol)]  # similar to [I,Q,U,V]
    assert corr_type.shape[1] == npol
    assert weight.shape[1] == npol

    if miss > 0:
        b_vis = numpy.zeros([ntime, nbaseline, nchan, npol], dtype='complex64')
        b_weight = numpy.zeros([ntime, nbaseline, npol])  # To save memory weight does not broadcast to the vis
        b_uvw = numpy.zeros([ntime, nbaseline, 3])

        rstart = 0
        for nt_index in range(ntime):
            rend = rstart + ucount[nt_index]

            nb_index = baseline_index[rstart:rend]
            b_vis[nt_index, nb_index, ...] = vis[rstart:rend, ...]
            b_weight[nt_index, nb_index, :] = weight[rstart:rend, :]
            b_uvw[nt_index, nb_index, :] = uvw[rstart:rend, :]

            rstart += ucount[nt_index]

    elif miss == 0:
        b_vis = vis.reshape((ntime, nbaseline, nchan, npol))
        b_weight = weight.reshape((ntime, nbaseline, npol))
        b_uvw = uvw.reshape((ntime, nbaseline, 3))

    else:
        b_uvw = None
        b_weight = None
        b_vis = None
        print('create an Empty MSData')

    del time, antenna1, antenna2, vis, weight, uvw
    gc.collect()
    # nfpos: visibility's Fourier position (u,v,w)
    main_table = xr.Dataset(data_vars={'vis': (['ntime', 'nbaseline', 'nchan', 'npol'], b_vis),
                                       'weight': (['ntime', 'nbaseline', 'npol'], b_weight),
                                       'uvw': (['ntime', 'nbaseline', 'nfpos'], b_uvw),
                                       'time': (['ntime'], utime),
                                       },
                            coords={'ntime': numpy.arange(ntime),
                                    'nbaseline': baseline,
                                    'nchan': numpy.arange(nchan),
                                    'npol': p_coords,
                                    'nfpos': ['u', 'v', 'w']})

    msvis = xr.merge([main_table, fre_band, configuration])

    msvis.attrs['tag'] = ms_tag
    msvis.attrs['phasecentre'] = phasecentre
    msvis.attrs['polarisation'] = corr_type
    # print('Reading %d rows from the MeasurementSet (the remaining %d row(s) are missing)' % (nrow, miss))
    return msvis


@timer
def load_msdata(ms_file, ack=False):
    """
    :param ms_file: msfile name
    :param ack: ack=False` prohibit the printing of a message telling if the table was opened or created successfully.
    :return: list of xarray
    """

    @delayed
    def pick_source(d, f, tab):

        fre_band = frequency_init(ms_file, d)
        query_ms = tab.query(query='DATA_DESC_ID==%d AND FIELD_ID==%d AND ANTENNA1!=ANTENNA2' % (d, f),
                             sortlist='TIME,ANTENNA1,ANTENNA2',
                             columns='UVW,WEIGHT,ANTENNA1,ANTENNA2,TIME,DATA')
        nrows = query_ms.nrows()

        assert nrows > 0, "Empty selection for FIELD_ID=%d and DATA_DESC_ID=%d" % (f, d)
        ms_tag = str(d) + '_' + str(f)
        phasecentre = phasecentre_init(ms_file, f)
        ms_vis = msdataset_init(query_ms, fre_band, phasecentre, configuration, corr_type, ms_tag)

        return ms_vis

    tmp = []

    configuration = configuration_init(ms_file)
    corr_type = polarisation_init(ms_file)

    with table(ms_file, ack=ack) as tab:

        dds = numpy.unique(tab.getcol('DATA_DESC_ID'))  # ndarray <class 'tuple'>: (0,1) ===> relate to Frequency info
        fields = numpy.unique(tab.getcol('FIELD_ID'))  # ndarray <class 'tuple'>: (2,3,5,7)  ===> relate to Field info

        # print("Found spectral window:%s, and fields:%s\n" % (str(dds), str(fields)))

        for dd in dds:
            for field in fields:
                msvis = pick_source(dd, field, tab)  # secondary split MS in parallel
                tmp.append(msvis)

        msvis_list = dask.compute(*tmp, scheduler='threads')

    memory = [i.nbytes for i in msvis_list]
    total = numpy.sum(memory) / 1024 ** 2
    print('Loading a single MS(Xarray total volumn:%.2f MB)' % total)

    return msvis_list


@timer
def time_average(seq_vis, avg_time=2):
    """
    Integrate visibility by timesteps
    :param seq_vis:
    :param avg_time:
    :return:
    """

    @delayed
    def time_avg(vis_set):

        assert isinstance(vis_set, Dataset)

        nt = vis_set.sizes['ntime']
        assert avg_time <= nt
        count = nt // avg_time
        last = nt % avg_time

        if last > 0:
            # print("WARNING:It will DROP the last %d timestamps\n" % last)
            nt = nt - last
            vis_set = vis_set.reindex(ntime=numpy.arange(0, nt))

        label = numpy.arange(0, count)

        group_sum = (vis_set.vis * vis_set.weight).groupby_bins('ntime', bins=count, labels=label).sum(dim='ntime')
        group_weight = vis_set.weight.groupby_bins('ntime', bins=count, labels=label).sum(dim='ntime')
        group_vis = (group_sum / group_weight + 1e-16).astype('complex64')  # .fillna(0.0 + 0.0j)
        group_time = vis_set.time.groupby_bins('ntime', bins=count, labels=label).mean(dim='ntime')

        # TODO a complicated operation to correct uvw
        group_uvw = vis_set.uvw.groupby_bins('ntime', bins=count, labels=label).mean(
            dim='ntime')
        # print(group_weight[0, 0].values)

        vis_set = vis_set.drop_dims('ntime')
        vis_set = vis_set.assign(weight=group_weight,
                                 vis=group_vis,
                                 uvw=group_uvw,
                                 time=group_time).rename({'ntime_bins': 'ntime'})
        return vis_set

    assert isinstance(avg_time, int)
    if avg_time == 1:  # no change
        return seq_vis

    vis_list = []
    for vis in seq_vis:
        single = time_avg(vis)
        vis_list.append(single)

    time_avg = dask.compute(*vis_list, scheduler='threads')

    memory = [i.nbytes for i in time_avg]
    total = numpy.sum(memory) / 1024 ** 2
    print('Time(%d) averaging on MS (Xarray volume reduced to:%.2f MB)' % (avg_time, total))

    return time_avg


@timer
def channel_average(seq_vis, avg_channel=4):
    """
    Integrate visibility by channels
    :param seq_vis:
    :param avg_channel: int
    :return:
    """

    @delayed
    def chan_avg(vis_set):

        assert isinstance(vis_set, Dataset)

        nchan = vis_set.sizes['nchan']
        assert avg_channel <= nchan
        count = nchan // avg_channel
        last = nchan % avg_channel

        if last > 0:
            # print("WARNING:It will DROP the last %d channels\n" % last)
            nchan = nchan - last
            vis_set = vis_set.reindex(chan=numpy.arange(0, nchan))

        # https://xray.readthedocs.io/en/stable/generated/xarray.DataArray.groupby_bins.html#xarray.DataArray.groupby_bins
        label = numpy.arange(0, count)

        fre = vis_set.frequency.groupby_bins('nchan', bins=count, labels=label).mean()

        bandwidth = vis_set.channel_bandwidth.groupby_bins('nchan', bins=count, labels=label).sum()

        group_sum = (vis_set.vis * vis_set.weight).groupby_bins('nchan', bins=count, labels=label).sum(dim='nchan')
        group_weight = vis_set.weight * avg_channel
        group_vis = (group_sum / group_weight + 1e-16).astype('complex64')  # .fillna(0.0 + 0.0j)
        # print(group_weight[0, 0].values)

        vis_set = vis_set.drop_dims('nchan')
        vis_set = vis_set.assign(weight=group_weight,
                                 vis=group_vis,
                                 frequency=fre,
                                 channel_bandwidth=bandwidth).rename({'nchan_bins': 'nchan'})
        return vis_set

    assert isinstance(avg_channel, int)
    if avg_channel == 1:  # no change
        return seq_vis

    vis_list = []
    for vis in seq_vis:
        single = chan_avg(vis)
        vis_list.append(single)

    channel_avg = dask.compute(*vis_list, scheduler='threads')

    memory = [i.nbytes for i in channel_avg]
    total = numpy.sum(memory) / 1024 ** 2
    print('Channel(%d) averaging on MS (Xarray volume reduced to:%.2f MB)' % (avg_channel, total))
    del seq_vis
    gc.collect()
    return channel_avg


def combine_msdata(vis1: Dataset, vis2: Dataset):
    assert vis1.tag == vis2.tag
    merge = xr.concat([vis1, vis2],
                      dim='ntime',
                      data_vars=['weight', 'uvw', 'vis'])

    return merge


if __name__ == '__main__':
    # path_fmt = '/mnt/storage-ssd/luokaida/data/rawdata/3C129_pband_target.ms'
    # load_data = load_msdata(path_fmt)

    c = Client('172.31.99.84:8786').restart()
    start1 = t.time()

    print('cpu_count: %d' % cpu_count())

    path_fmt = '/mnt/storage-ssd/luokaida/data/rawdata/3C129_pband_target.ms'
    all_path = [path_fmt for i in range(1)]
    print('the number of ms_files: %d\n' % len(all_path))

    files = db.from_sequence(all_path)

    avg_ms = files.map(load_msdata).map(time_average, avg_time=2).map(channel_average, avg_channel=4)
    combine_ms = avg_ms.flatten().foldby(key=lambda x: x.tag, binop=combine_msdata)
    # groups = avg_ms.flatten().groupby(grouper=lambda x: x.tag)  #
    # example_source = avg_ms.flatten().filter(lambda x: x.tag == '0_2').fold(combine_msdata)

    print('Start computation in the background....... ')  # construct task graphs above
    all_avg_ms = c.persist(combine_ms)  # Persist dask collections on cluster
    res = c.gather(all_avg_ms)
    wait(res)  # persist()--->gather()
    # res = avg_ms.compute()  # Don't use .compute()

    end1 = t.time()
    # combine_ms.visualize(filename='flow_chart_20.pdf')
    print('\nUsing dask-xarray:map-reduce costs %.2f seconds\n' % (end1 - start1))
    print()  # put a breakpoint here
