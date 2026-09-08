"""Legacy PCAT batch-processing helper.

This script is retained only as historical provenance and is not part of the
maintained package API.
"""

import os
import sys


def main():
    pathbase = os.environ.get('PCAT_DATA_PATH')
    if not pathbase:
        raise RuntimeError('PCAT_DATA_PATH is not set; cannot run legacy batch post-processing helper.')

    print('Legacy proc_btch helper retained for provenance only; active workflow lives in the maintained PCAT core.')
    print(f'Image root: {os.path.join(pathbase, "imag")}')


if __name__ == '__main__':
    main()

