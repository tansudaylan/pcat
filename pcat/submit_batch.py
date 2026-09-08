"""Legacy batch-submission helper.

This script is retained only as a historical convenience and is not part of the
maintained PCAT API surface.
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
    path = os.environ.get('TDGU_PATH')
    if not path:
        raise RuntimeError('TDGU_PATH is not set; cannot run the legacy PCAT batch submission helper.')

    with narr_open(os.path.join(path, 'pcatsubm.log'), 'w') as fileoutp:
        for name in sorted(os.listdir(path)):
            if not name.endswith('.py'):
                continue
            print(name)
            with narr_open(os.path.join(path, name), 'r') as fileobjt:
                for line in fileobjt:
                    if line.startswith('def pcat_'):
                        namefunc = line[4:-1].split('(')[0]
                        cmnd = f'python {os.path.join(path, name)} {namefunc}'
                        print(cmnd)
                        try:
                            os.system(cmnd)
                        except Exception as excp:
                            fileoutp.write(f'{namefunc} failed.\n')
                            fileoutp.write(f'{excp}\n')


if __name__ == '__main__':
    main()

