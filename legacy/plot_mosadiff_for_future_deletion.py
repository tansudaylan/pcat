"""Legacy PCAT mosaic-difference command retained for provenance only."""

from tdpy.verbosity import print

import os
import sys


def main():
    if len(sys.argv) < 3:
        raise SystemExit('Usage: plot_mosadiff_for_future_deletion.py <rtag1> <rtag2>')

    rtagfrst = sys.argv[1]
    rtagseco = sys.argv[2]

    pathbase = os.environ.get('PCAT_DATA_PATH')
    if not pathbase:
        raise RuntimeError('PCAT_DATA_PATH is not set; cannot run the legacy mosaic-difference helper.')

    print('Legacy plot_mosadiff helper retained for provenance only; active workflow is in the maintained PCAT package.')
    print(f'Requested run tags: {rtagfrst}, {rtagseco}')
    print(f'Output root: {pathbase}')


if __name__ == '__main__':
    main()