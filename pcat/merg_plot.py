"""Legacy PCAT merged-plot helper.

This script is retained only as historical provenance and is not part of the
maintained package API.
"""

import os
import sys


def main():
    if len(sys.argv) < 2:
        raise SystemExit('Usage: merg_plot.py <run-tag-pattern>')

    pathbase = os.environ.get('PCAT_DATA_PATH')
    if not pathbase:
        raise RuntimeError('PCAT_DATA_PATH is not set; cannot run legacy merged-plot helper.')

    print(f'Legacy merg_plot helper retained for provenance only; pattern requested: {sys.argv[1]!r}')
    print(f'Image root: {os.path.join(pathbase, "imag")}')


if __name__ == '__main__':
    main()

