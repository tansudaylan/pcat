"""Legacy PCAT deletion helper.

This script is retained only as historical provenance and is not part of the
supported PCAT API.
"""

import sys


def main():
    if len(sys.argv) < 2:
        raise SystemExit('Usage: delete_rtag.py <run-tag>')
    print(f'Legacy delete_rtag helper retained for provenance only; run tag requested: {sys.argv[1]}')


if __name__ == '__main__':
    main()


