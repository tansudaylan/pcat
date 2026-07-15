from __init__ import *


def narr_open(path, mode='r'):
    pathnorm = os.path.normpath(path)
    if mode.startswith('r') and '+' not in mode:
        action = 'Reading'
    else:
        action = 'Writing'
    print '%s %s...' % (action, pathnorm)
    return open(pathnorm, mode)

# flag to force-delete all runs
if len(sys.argv) > 1 and sys.argv[1] == 'forcdele':
    boolforcdele = True
else:
    boolforcdele = False

pathdata = os.environ["PCAT_DATA_PATH"] + '/data/outp/'
pathimag = os.environ["PCAT_DATA_PATH"] + '/imag/'

liststrgextn = ['/imag/', '/data/outp/']

for strgextn in liststrgextn:

    path = os.environ["PCAT_DATA_PATH"] + strgextn
    
    for rtag in os.listdir(path):
        
        pathfile = path + rtag
        if os.path.isdir(pathfile) and rtag[:8].isdigit():
            print 'Processing %s...' % rtag

            # check the chain status
            pathchec = pathfile.replace('imag', 'data/outp') + '/stat.txt'
            if os.path.isfile(pathchec):
                filestat = narr_open(pathchec, 'r')
                boolkeep = False
                for line in filestat:
                    if line == 'gdatmodipost written.\n':
                        boolkeep = True
                filestat.close()
        
            strgtemp = pathfile[pathfile.rfind('_')+1:]
            if strgtemp.endswith('tile'):
                strgtemp = strgtemp[:-4]
            if strgtemp.isdigit() and ((not os.path.isfile(pathchec) or not boolkeep or int(strgtemp) <= 1000) and not 'mockonly' in rtag) or boolforcdele:
                print 'Deleting %s...' % pathchec
                cmnd = 'rm -rf ' + pathfile
                os.system(cmnd)
            else:
                pass
                #print 'Saving %s...' % pathchec
            print

listrtagdata = fnmatch.filter(os.listdir(pathdata), '2*')
listrtagimag = fnmatch.filter(os.listdir(pathimag), '2*')

booltemp = False
for rtag in listrtagdata:
    if not rtag in listrtagimag:
        booltemp = True
for rtag in listrtagimag:
    if not rtag in listrtagdata:
        booltemp = True
if booltemp:
    print 'Data and image folders are not synched!'


