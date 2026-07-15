from __init__ import *


def narr_open(path, mode='r'):
    pathnorm = os.path.normpath(path)
    if mode.startswith('r') and '+' not in mode:
        action = 'Reading'
    else:
        action = 'Writing'
    print '%s %s...' % (action, pathnorm)
    return open(pathnorm, mode)

def comp(nameplot):
    
    if nameplot == 'DS_Store':
        return
    
    nameplot = nameplot[:-4]

    print 'comp() working on ' + nameplot + '...'
    
    cmnd = 'mkdir -p ' + pathimag + 'comprtag/' + nameplot + '/'
    #print cmnd
    os.system(cmnd)
    #print
    
    strgplot = nameplot.split('/')[-1]

    cmndconv = 'convert -density 300'
    for line in listline:
        
        namedest = pathimag + 'comprtag/' + nameplot + '/' + strgplot + '_' + line + '.pdf'
        if not os.path.isfile(namedest):
            strgorig = '%s' % pathimag + line + '/' + nameplot + '.pdf'
            cmnd = 'cp %s %s' % (strgorig, namedest)
            
            #print cmnd
            os.system(cmnd)
            #print
        else:
            pass
            print 'File already exists...'
        #print
    
        cmndconv += ' ' + namedest
    cmndconv += ' ' + pathimag + 'comprtag/' + nameplot + '/merg.pdf'
    os.system(cmndconv)
    #print

print 'comp_rtag() initialized...'

pathimag = os.environ["PCAT_DATA_PATH"] + '/imag/'

cmnd = 'mkdir -p %s' % (os.environ["PCAT_DATA_PATH"] + '/imag/comprtag')
os.system(cmnd)

print 'Listing the available runs...'
cmnd = 'ls %s | xargs -n 1 basename > %s/listrtag.txt' % (pathimag, pathimag)
print cmnd
os.system(cmnd)

pathlist = pathimag + 'listrtag.txt'

with narr_open(pathlist, 'r') as thisfile:
    listline = thisfile.readlines()
    listline = [x.strip() for x in listline] 

if len(sys.argv) > 2:
    listline = sys.argv[1:]
else:
    strgsrch = sys.argv[1]
    listline = fnmatch.filter(listline, strgsrch)

print 'listline'
for line in listline:
    print line

namerefr = listline[0]
pathrefr = os.environ["PCAT_DATA_PATH"] + '/imag/' + namerefr + '/'

print 'Iterating over folders of the reference run...'
for namefrst in os.listdir(pathrefr):
    namefrstprim = os.path.join(pathrefr, namefrst)
    if os.path.isdir(namefrstprim):
        for nameseco in os.listdir(namefrstprim):
            if nameseco == 'fram':
                continue
            namesecoprim = os.path.join(namefrstprim, nameseco)
            if os.path.isdir(namesecoprim):
                for namethrd in os.listdir(namesecoprim):
                    namethrdprim = os.path.join(namesecoprim, namethrd)
                    if os.path.isdir(namethrdprim):
                        for namefrth in os.listdir(namethrdprim):
                            namefrthprim = os.path.join(namethrdprim, namefrth)
                            comp(namefrthprim.split(pathrefr)[-1])
                    else:
                        comp(namethrdprim.split(pathrefr)[-1])
            elif nameseco.endswith('pdf') or nameseco.endswith('gif'):
                comp(namesecoprim.split(pathrefr)[-1])
    elif namefrst.endswith('pdf') or namefrst.endswith('gif'):
        comp(namefrstprim.split(pathrefr)[-1])



