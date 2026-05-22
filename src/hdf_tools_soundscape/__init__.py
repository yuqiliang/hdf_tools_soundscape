"""hdf_tools_soundscape package.

Primary APIs are re-exported here for convenient imports.
"""

from .head_hdf_utils import (
    P_REF,
    HeadHDFInfo,
    compute_leq_pa,
    compute_rms_spl,
    get_channel_calibration_db,
    get_channel_calibration_db_from_text,
    inspect_head_hdf,
    parse_header_info,
    plot_mark_style,
    read_head_file,
)

__all__ = [
    "P_REF",
    "HeadHDFInfo",
    "inspect_head_hdf",
    "get_channel_calibration_db_from_text",
    "get_channel_calibration_db",
    "parse_header_info",
    "read_head_file",
    "compute_leq_pa",
    "compute_rms_spl",
    "plot_mark_style",
]
