from __init__ import *
from util import *


def narr_h5(path, mode):
    pathnorm = os.path.normpath(path)
    if mode.startswith('r') and '+' not in mode:
        action = 'Reading'
    else:
        action = 'Writing'
    print '%s %s...' % (action, pathnorm)
    return h5py.File(pathnorm, mode)

rtagroot = sys.argv[1]
pathdata = os.environ["PCAT_DATA_PATH"] + '/data/outp/'

listrtagdata = fnmatch.filter(os.listdir(pathdata), rtagroot)
numbcnfg = len(listrtagdata)
for k, rtag in enumerate(listrtagdata):
    print 'Processing %s...' % rtag
    pathoutprtag = retr_pathoutprtag(rtag)
    path = pathoutprtag + 'gdatinit'
    gdat = readfile(path) 
    if k == 0:
        cntpdataarry = empty([numbcnfg] + list(gdat.cntpdatareg0.shape))
    cntpdataarry[k, ...] = gdat.cntpdatareg0

path = pathdata + 'truecntpdata_%s.h5' % listrtagdata[0]
print 'Writing to %s...' % path
filearry = narr_h5(path, 'w')
filearry.create_dataset('cntpdataarry', data=cntpdataarry)
filearry.close()


