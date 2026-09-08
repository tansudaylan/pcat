"""Legacy PCAT animation-generation helper.

This script is retained only for historical reference and is not part of the
maintained package API.
"""

import os
import sys


def main():
    if len(sys.argv) < 3:
        raise SystemExit('Usage: make_anim_rand.py <run-tag> <plot-name> [fraction]')

    rtag = sys.argv[1]
    strgplot = sys.argv[2]
    pathbase = os.environ.get('PCAT_DATA_PATH')
    if not pathbase:
        raise RuntimeError('PCAT_DATA_PATH is not set; cannot run legacy animation helper.')

    print(f'Legacy make_anim_rand helper retained for provenance only; requested run tag {rtag!r}, plot {strgplot!r}.')
    print(f'Image root: {os.path.join(pathbase, "imag", rtag, "post", "fram")}')


if __name__ == '__main__':
    main()

