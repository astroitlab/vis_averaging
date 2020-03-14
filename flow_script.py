import os

calweight = True  # Whether the weights are calibrated along with the data
msfile = 'day2_copy_avgchannel2'  # modify input ms_name

#equal to ['2',         '3',        '5',         '7']
fieldlist=['J0954+1743','IRC+10216','J1229+0203','J1331+3030']


print '\nTask: listobs'
if os.path.exists('mylistobs.out'):
    print 'File: %s already exists ... deleting' % 'mylistobs.out'
    os.remove('mylistobs.out')
listobs(vis=msfile,listfile = 'mylistobs.out')

if os.path.exists('antpos.cal'):
    print 'antpos.cal already exists'
    
else:
    print 'antpos.cal does not exist ... creating antpos.cal. It will query the VLA webpages for the offsets'
    # it will query the VLA webpages for the offsets
    gencal(vis='day2_TDEM0003_10s_norx',caltable='antpos.cal',
           caltype='antpos',
           antenna='')


print '\nTask: flagdata'
flagdata(vis=msfile,
         mode='list', 
         inpfile=["field='2,3' antenna='ea12' timerange='03:41:00~04:10:00'",
                  "field='2,3' antenna='ea08' timerange='03:21:40~04:10:00' spw='1'",
		  "antenna='ea07,ea12,ea28'",
                  "antenna='ea23' timerange='03:21:40~04:10:00'"])

print '\ncreating gaincurve.cal ...'
gencal(vis=msfile,caltable='gaincurve.cal',
       caltype='gceff')

print '\ncreating opacity.cal ...'
myTau = plotweather(vis=msfile, doPlot=False)
gencal(vis=msfile,caltable='opacity.cal',
       caltype='opac',
       spw='0,1',
       parameter=myTau)

print '\nTask setjy: on the field 7,using model 3C286_A.im'
setjy(vis=msfile,field='7',spw='0~1',scalebychan=True,model='3C286_A.im')


print '\ncreating delays.cal ...'
gaincal(vis=msfile, caltable='delays.cal', field='5', 
        refant='ea02', gaintype='K', gaintable=['antpos.cal','gaincurve.cal','opacity.cal'])

print '\ncreating bpphase.gcal(phase-only) ...'
gaincal(vis=msfile,caltable='bpphase.gcal',
        field='5',spw='0~1',
        refant='ea02',calmode='p',solint='int',minsnr=2.0,
        gaintable=['antpos.cal','gaincurve.cal','opacity.cal','delays.cal'])

print '\ncreating bandpass.bcal ...'
bandpass(vis=msfile,caltable='bandpass.bcal',field='5',
        refant='ea02',solint='inf',solnorm=True,
        gaintable=['antpos.cal','gaincurve.cal','opacity.cal','delays.cal','bpphase.gcal'])

print '\ncreating intphase.gcal ...'
gaincal(vis=msfile,caltable='intphase.gcal',field='2,5,7',
        spw='0~1',solint='int',refant='ea02',minsnr=2.0,calmode='p',
        gaintable=['antpos.cal','gaincurve.cal','opacity.cal','delays.cal','bandpass.bcal'])

print '\ncreating scanphase.gcal ...'
gaincal(vis=msfile,caltable='scanphase.gcal',field='2,5,7',
        spw='0~1',solint='inf',refant='ea02',minsnr=2.0,calmode='p',
        gaintable=['antpos.cal','gaincurve.cal','opacity.cal','delays.cal','bandpass.bcal'])

print '\ncreating amp.gcal ...'
gaincal(vis=msfile,caltable='amp.gcal',field='2,5,7',
        spw='0~1',solint='inf',refant='ea02',minsnr=2.0,calmode='ap',
        gaintable=['antpos.cal','gaincurve.cal','opacity.cal','delays.cal','bandpass.bcal','intphase.gcal'])


if os.path.exists('flux.cal'):
    print '\nflux calibration table: %s already exists, deleting ...' % 'flux.cal'
    rmtables('flux.cal')
print '\ncreating flux.cal ...'
fluxscale(vis=msfile,caltable='amp.gcal',
          fluxtable='flux.cal',reference='7',incremental=True)


print '\napplycal on the field 2 ...parameter calwt=%s' % calweight    
applycal(vis=msfile,field='2',
        gaintable=['antpos.cal','gaincurve.cal','opacity.cal','delays.cal','bandpass.bcal','intphase.gcal','amp.gcal','flux.cal'],
        gainfield=['','','','5','5','2','2','2'],
        calwt=calweight)
print '\napplycal on the field 5 ...parameter calwt=%s' % calweight  
applycal(vis=msfile,field='5',
        gaintable=['antpos.cal','gaincurve.cal','opacity.cal','delays.cal','bandpass.bcal','intphase.gcal','amp.gcal','flux.cal'],
        gainfield=['','','','5','5','5','5','5'],
        calwt=calweight)
print '\napplycal on the field 7 ...parameter calwt=%s' % calweight  
applycal(vis=msfile,field='7',
        gaintable=['antpos.cal','gaincurve.cal','opacity.cal','delays.cal','bandpass.bcal','intphase.gcal','amp.gcal','flux.cal'],
        gainfield=['','','','5','5','7','7','7'],
        calwt=calweight)
print '\napplycal on the field 3 ...parameter calwt=%s' % calweight
applycal(vis=msfile,field='3',
        gaintable=['antpos.cal','gaincurve.cal','opacity.cal','delays.cal','bandpass.bcal','scanphase.gcal','amp.gcal','flux.cal'],
        gainfield=['','','','5','5','2','2','2'],
        calwt=calweight)

for fd in fieldlist:
    print '\nTask: split on the field: %s ...'%fd
    splitfile=fd+'.split.ms'
    if os.path.exists(splitfile):
	print 'File: %s already exists, deleting ...' % splitfile
        rmtables(splitfile)
    split(vis=msfile,outputvis=splitfile,datacolumn='corrected',field=fd)
    splitfile = fd + '.split.ms'
    print 'Task: statwt on the field: %s ...'%fd
    statwt(vis=splitfile, datacolumn='data')


#fieldlist=['J0954+1743','IRC+10216','J1229+0203','J1331+3030']
niterlist=[2000,           2000,           2000,            2000]
thresholdlist=['0.5mJy',  '1mJy',         '2mJy',          '2mJy']
imsizelist=[1024,          1024,           1024,            1024]
celllist =['0.08arcsec',  '0.05arcsec',   '0.08arcsec',   '0.1arcsec']
default(tclean)
savemodel           =      'modelcolumn'
specmode            =      'mfs'
gain                =      0.1   
deconvolver         =      'clark_exp'            
cyclefactor         =      3              
interactive         =      False      
mask                =      []
stokes              =      'I'        
weighting           =      'briggs'        
uvtaper             =      []      
pbcor               =      False      
pblimit             =     -0.0001 
selectdata          =      False
for fd, niter, threshold, imsize, cell in zip(fieldlist, niterlist, thresholdlist, imsizelist, celllist):
    print '\nTask tclean on the field : %s ...' % fd
    splitfile = fd + '.split.ms'
    vis = splitfile
    imagename = fd + '_img'
    niter = niter
    threshold = threshold
    imsize = [imsize, imsize]
    cell = [cell, cell]
    if os.path.exists(imagename + '.image'):
        print 'Image File: %s.image already exists, deleting ...' % imagename
        rmtables(imagename + '*') # get ride of earlier versions of the image
    tclean()





