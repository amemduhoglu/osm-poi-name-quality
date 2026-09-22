"""Shared library for the POI name audit study.

The package holds what every phase needs: the configuration loader and the
logging setup. Analysis and run scripts live under scripts/ and import from
here.
"""

__all__ = ["config", "logsetup"]
