"""Legacy PCAT helper used to collect true-count arrays across run tags.

This script is retained only for historical reference and is not part of the
current supported PCAT API.
"""

import os
import sys


def narr_h5(path, mode):
    pathnorm = os.path.normpath(path)
    if mode.startswith('r') and '+' not in mode:
        action = 'Reading'
    else:
        action = 'Writing'
    print(f'{action} {pathnorm}...')
    return __import__('h5py').File(pathnorm, mode)


def main():
    if len(sys.argv) < 2:
        raise SystemExit('Usage: coll_true.py <run-tag-root>')

    rtagroot = sys.argv[1]
    pathbase = os.environ.get('PCAT_DATA_PATH')
    if not pathbase:
        raise RuntimeError('PCAT_DATA_PATH is not set; cannot run legacy true-count collection.')

    pathdata = os.path.join(pathbase, 'data', 'outp')
    listrtagdata = [name for name in os.listdir(pathdata) if name.startswith(rtagroot)]
    if not listrtagdata:
        raise RuntimeError(f'No run tags matching {rtagroot!r} found under {pathdata}.')

    print(f'Legacy coll_true helper retained for provenance; matched {len(listrtagdata)} run tags.')
    print(f'Output root: {pathdata}')


if __name__ == '__main__':
    main()


