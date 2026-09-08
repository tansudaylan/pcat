"""Legacy PCAT garbage-collection helper.

This file is retained only for historical cleanup scripts and is not part of the
maintained PCAT library API.
"""

import fnmatch
import os
import sys


def narr_open(path, mode='r'):
    pathnorm = os.path.normpath(path)
    if mode.startswith('r') and '+' not in mode:
        action = 'Reading'
    else:
        action = 'Writing'
    print(f'{action} {pathnorm}...')
    return open(pathnorm, mode)


def main():
    boolforcdele = len(sys.argv) > 1 and sys.argv[1] == 'forcdele'
    pathbase = os.environ.get('PCAT_DATA_PATH')
    if not pathbase:
        raise RuntimeError('PCAT_DATA_PATH is not set; cannot clean legacy PCAT output trees.')

    pathdata = os.path.join(pathbase, 'data', 'outp')
    pathimag = os.path.join(pathbase, 'imag')
    liststrgextn = ['/imag/', '/data/outp/']

    for strgextn in liststrgextn:
        path = os.path.join(pathbase, *strgextn.strip('/').split('/')) if strgextn.startswith('/') else os.path.join(pathbase, strgextn)
        for rtag in os.listdir(path):
            pathfile = os.path.join(path, rtag)
            if os.path.isdir(pathfile) and rtag[:8].isdigit():
                print(f'Processing {rtag}...')
                pathchec = os.path.join(pathfile.replace('imag', 'data/outp'), 'stat.txt')
                boolkeep = False
                if os.path.isfile(pathchec):
                    with narr_open(pathchec, 'r') as filestat:
                        for line in filestat:
                            if line == 'gdatmodipost written.\n':
                                boolkeep = True

                strgtemp = pathfile[pathfile.rfind('_') + 1:]
                if strgtemp.endswith('tile'):
                    strgtemp = strgtemp[:-4]
                if strgtemp.isdigit() and ((not os.path.isfile(pathchec) or not boolkeep or int(strgtemp) <= 1000) and 'mockonly' not in rtag) or boolforcdele:
                    print(f'Deleting {pathchec}...')
                    os.system(f'rm -rf "{pathfile}"')

    listrtagdata = fnmatch.filter(os.listdir(pathdata), '2*')
    listrtagimag = fnmatch.filter(os.listdir(pathimag), '2*')

    booltemp = False
    for rtag in listrtagdata:
        if rtag not in listrtagimag:
            booltemp = True
    for rtag in listrtagimag:
        if rtag not in listrtagdata:
            booltemp = True
    if booltemp:
        print('Data and image folders are not synced!')


if __name__ == '__main__':
    main()


