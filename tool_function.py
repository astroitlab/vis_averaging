import os
import numpy


def arl_path(path):
    """Converts a path that might be relative to ARL root into an absolute path

    :param path:
    :return: absolute path
    """

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    arlhome = os.getenv('ARL', project_root)
    return os.path.join(arlhome, path)


def create_vis_from_blockvis(tag_vis) -> Dataset:
    #  drop_dims('nant'): Only the data needed for gridding is retained
    vis = tag_vis[1].drop_dims('nant').stack(nrow=('ntime', 'nbaseline', 'nchan'))

    c = constants.c.to('m s^-1').value
    vis.update({'uvw': (vis.frequency * vis.uvw / c)})
    vis.update({'vis': vis.vis.transpose('nrow', 'npol')})
    vis.update({'weight': vis.weight.transpose('nrow', 'npol')})

    # test_uvw_true = (tag_vis[1].uvw * tag_vis[1].frequency / c).transpose('ntime', 'nbaseline', 'nchan', 'nfpos') \
    #                     .stack(nrow=('ntime', 'nbaseline', 'nchan',)) == vis.uvw

    # compare results against CASA (avg_time=2(20s),avg_channel=4) TaQL: DATA_DESC_ID==0 AND FIELD_ID==7

    # view data: vis.time.values.reshape([-1,171*16])  # tag='0_7'
    # view data: vis.frequency.values.reshape([-1,16])  # tag='0_7'
    # view data: vis.weight.values.reshape([-1,4])[::16]  # tag='0_7'
    # view data: print(vis.vis.values.reshape([-1,4])[::16][:12])  # tag='0_7'

    """
    <xarray.Dataset>
    Dimensions:            (nant: 19, nbaseline: 171, nchan: 16, nfpos: 3, npol: 4, nspos: 3, ntime: 8)
    Coordinates:
      * nbaseline          (nbaseline) <U5 '0-1' '0-2' '0-3' ... '16-18' '17-18'
      * npol               (npol) <U2 'p0' 'p1' 'p2' 'p3'
      * nfpos              (nfpos) <U1 'u' 'v' 'w'
      * nant               (nant) int64 0 1 2 3 4 5 6 7 ... 11 12 13 14 15 16 17 18
      * nspos              (nspos) <U1 'x' 'y' 'z'
      * ntime              (ntime) int64 0 1 2 3 4 5 6 7
      * nchan              (nchan) int64 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
    Data variables:
        xyz                (nant, nspos) float64 -1.602e+06 -5.042e+06 ... 3.555e+06
        mount              (nant) <U6 'ALT-AZ' 'ALT-AZ' ... 'ALT-AZ' 'ALT-AZ'
        names              (nant) <U4 'ea01' 'ea02' 'ea03' ... 'ea25' 'ea27' 'ea28'
        diameter           (nant) float64 25.0 25.0 25.0 25.0 ... 25.0 25.0 25.0
        weight             (ntime, nbaseline, npol) float64 64.0 64.0 ... 80.0 80.0
        uvw                (ntime, nbaseline, nfpos) float64 -544.7 -288.6 ... 37.74
        time               (ntime) float64 4.779e+09 4.779e+09 ... 4.779e+09
        vis                (ntime, nbaseline, nchan, npol) complex64 (0.0018164966-0.0031041054j) ... (0.0045034965+0.003496466j)
        frequency          (nchan) float64 3.639e+10 3.639e+10 ... 3.639e+10
        channel_bandwidth  (nchan) float64 5e+05 5e+05 5e+05 ... 5e+05 5e+05 5e+05
    Attributes:
        tag:           0_7
        phasecentre:   <SkyCoord (ICRS): (ra, dec) in deg\n    (202.78453327, 30....
        polarisation:  [[5 6 7 8]\n [5 6 7 8]])
        
                                  ||
                                  ||  transform
                                  ||
                                  \/
    <xarray.Dataset>
    Dimensions:            (nfpos: 3, npol: 4, nrow: 21888, nspos: 3)
    Coordinates:
      * npol               (npol) <U2 'p0' 'p1' 'p2' 'p3'
      * nfpos              (nfpos) <U1 'u' 'v' 'w'
      * nspos              (nspos) <U1 'x' 'y' 'z'
      * nrow               (nrow) MultiIndex
      - ntime              (nrow) int64 0 0 0 0 0 0 0 0 0 0 ... 7 7 7 7 7 7 7 7 7 7
      - nbaseline          (nrow) object '0-1' '0-1' '0-1' ... '17-18' '17-18'
      - nchan              (nrow) int64 0 1 2 3 4 5 6 7 8 ... 8 9 10 11 12 13 14 15
    Data variables:
        weight             (nrow, npol) float64 64.0 64.0 64.0 ... 80.0 80.0 80.0
        uvw                (nrow, nfpos) float64 -6.612e+04 -3.502e+04 ... 4.581e+03
        time               (nrow) float64 4.779e+09 4.779e+09 ... 4.779e+09
        vis                (nrow, npol) complex64 (0.0018164966-0.0031041054j) ... (0.0045034965+0.003496466j)
        frequency          (nrow) float64 3.639e+10 3.639e+10 ... 3.639e+10
        channel_bandwidth  (nrow) float64 5e+05 5e+05 5e+05 ... 5e+05 5e+05 5e+05
    Attributes:
        tag:           0_7
        phasecentre:   <SkyCoord (ICRS): (ra, dec) in deg\n    (202.78453327, 30....
        polarisation:  [[5 6 7 8]\n [5 6 7 8]]
    """
    # print(vis)
    return vis
