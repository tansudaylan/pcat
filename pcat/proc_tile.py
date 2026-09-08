"""Legacy PCAT tile bootstrap helper.

This script is retained only as historical provenance and is not part of the
maintained package API.
"""

import os
import sys


def main():
    if len(sys.argv) < 2:
        raise SystemExit('Usage: proc_tile.py <run-tag-pattern>')

    pathbase = os.environ.get('PCAT_DATA_PATH')
    if not pathbase:
        raise RuntimeError('PCAT_DATA_PATH is not set; cannot run legacy tile bootstrap helper.')

    print(f'Legacy proc_tile helper retained for provenance only; pattern requested: {sys.argv[1]!r}')
    print(f'Data root: {os.path.join(pathbase, "data", "outp")}')


if __name__ == '__main__':
    main()

