"""Legacy PCAT run-comparison plotting helper.

This script is retained only as historical provenance and is not part of the
maintained package API.
"""

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
    pathbase = os.environ.get('PCAT_DATA_PATH')
    if not pathbase:
        raise RuntimeError('PCAT_DATA_PATH is not set; cannot run legacy comparison plotting helper.')

    print('Legacy comp_rtag helper retained for provenance only; active workflow belongs in the supported PCAT package.')
    print(f'Imagery root: {os.path.join(pathbase, "imag")}')


if __name__ == '__main__':
    main()

