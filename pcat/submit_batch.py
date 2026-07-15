from __init__ import *


def narr_open(path, mode='r'):
    pathnorm = os.path.normpath(path)
    if mode.startswith('r') and '+' not in mode:
        action = 'Reading'
    else:
        action = 'Writing'
    print '%s %s...' % (action, pathnorm)
    return open(pathnorm, mode)


path = os.environ["TDGU_PATH"] + '/'
fileoutp = narr_open(path + 'pcatsubm.log', 'w')
cntr = 0
for name in os.listdir(path):
    if name.endswith(".py"):
        print name
        fileobjt = narr_open(path + name, 'r')
        for line in fileobjt:
            if line.startswith('def pcat_'):
                
                #if cntr == 5:
                #    break
                
                cntr += 1
                namefunc = line[4:-1].split('(')[0]
                cmnd = 'python $TDGU_PATH/%s %s' % (name, namefunc)
                print cmnd
                try:
                    os.system(cmnd)
                except Exception as excp:
                    strg = str(excp)
                    fileoutp.write('%s failed.' % namefunc)
                    fileoutp.write(strg)
                print
                print
                print
                print
                print
                print
                print
                print
            fileobjt.close()

fileoutp.close()

