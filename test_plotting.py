#!/usr/bin/env python
"""Quick test of plotting narration."""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pcat.main as main

# Simple configuration
dictglob = dict()

print("Initializing PCAT for plotting test...")
try:
    dictoutp = main.init(dictglob, numbswep=100, numbsamp=10)
    print("PCAT initialization complete.")
    print("Exit code: 0")
except Exception as e:
    print(f"Error during initialization: {e}")
    print("Exit code: 1")
