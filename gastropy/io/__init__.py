"""
GastroPy I/O Module
=====================

BIDS-compliant physiology data I/O and format conversion utilities.

Functions
---------
read_bids_physio
    Read a BIDS ``_physio.tsv.gz`` and companion JSON sidecar.
write_bids_physio
    Write signal data as BIDS ``_physio.tsv.gz`` with JSON sidecar.
parse_bids_filename
    Parse BIDS key-value entities from a filename.
brainvision_to_bids
    Convert BrainVision ``.vhdr`` to BIDS physio format (requires MNE).
"""

from ._bids import parse_bids_filename, read_bids_physio, write_bids_physio
from ._brainvision import brainvision_to_bids

__all__ = [
    "read_bids_physio",
    "write_bids_physio",
    "parse_bids_filename",
    "brainvision_to_bids",
]
