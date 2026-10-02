"""Legacy PCAT mosaic-difference command retained for provenance only."""

from tdpy.verbosity import print

import sys
from pcat.paths import get_repository_path


def main():
    if len(sys.argv) < 3:
        raise SystemExit('Usage: plot_mosadiff_for_future_deletion.py <rtag1> <rtag2>')

    rtagfrst = sys.argv[1]
    rtagseco = sys.argv[2]

    pathbase = get_repository_path()

    print('Legacy plot_mosadiff helper retained for provenance only; active workflow is in the maintained PCAT package.')
    print(f'Requested run tags: {rtagfrst}, {rtagseco}')
    print(f'Output root: {pathbase}')


if __name__ == '__main__':
    main()