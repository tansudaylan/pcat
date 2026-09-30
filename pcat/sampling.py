"""Stable public entry points for the PCAT sampling engine.

The implementation remains in :mod:`pcat.main` while that legacy module is
gradually decomposed. New integrations should import sampling operations here.
"""

from .main import init, init_image, sample, sample_parallel

__all__ = ["init", "init_image", "sample", "sample_parallel"]