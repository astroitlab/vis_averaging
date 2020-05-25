fieldlist=['1','2','4','8','16']
for fd in fieldlist:
    imagename = fd + '_IRC+10216_img.image'
    psfname = fd + '_IRC+10216_img.psf'
    
    viewer(imagename)
    viewer(psfname)
