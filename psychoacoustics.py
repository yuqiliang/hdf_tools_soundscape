"""
psychoacoustics.py

Psychoacoustic metrics based on MoSQITo:

- Time-varying loudness (Zwicker, ISO 532-1)
- Sharpness (DIN 45692), stationary and time-varying
- Roughness (Daniel & Weber)

All functions expect the input signal in calibrated sound pressure [Pa],
which matches the output of `read_head_file()` in `head_hdf_utils.py`.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    # MoSQITo psychoacoustic metrics
    from mosqito.sq_metrics import (
        loudness_zwtv,
        sharpness_din_st,
        sharpness_din_tv,
        roughness_dw,
    )
except ImportError as e:  # pragma: no cover - gives a clearer error message
    raise ImportError(
        "mosqito is required for psychoacoustic metrics.\n"
        "Install it, for example:\n\n"
        "    uv add mosqito\n\n"
        "See https://mosqito.readthedocs.io/ for details."
    ) from e


# ----------------------------------------------------------------------
# Loudness (Zwicker, ISO 532-1, time-varying)
# ----------------------------------------------------------------------


def compute_loudness_zwicker(
    signal_pa: np.ndarray,
    fs: float,
    field_type: str = "free",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute time-varying loudness according to Zwicker (ISO 532-1:2017).

    This is a thin wrapper around `mosqito.sq_metrics.loudness_zwtv`.

    Parameters
    ----------
    signal_pa : np.ndarray
        1D array of calibrated sound pressure samples [Pa].
    fs : float
        Sampling frequency in Hz.
    field_type : {'free', 'diffuse'}, optional
        Type of sound field, by default 'free'.

    Returns
    -------
    time_axis : np.ndarray
        Time axis [s], shape (N_time,).
    N : np.ndarray
        Overall loudness [sone], shape (N_time,).
    N_specific : np.ndarray
        Specific loudness [sone/Bark], shape (N_bark, N_time).
    bark_axis : np.ndarray
        Bark axis, shape (N_bark,).

    Notes
    -----
    MoSQITo returns (N, N_specific, bark_axis, time_axis).
    This wrapper reorders the outputs to (time_axis, N, N_specific, bark_axis)
    which is often more convenient when plotting against time.
    """
    # MoSQITo signature:
    # N, N_specific, bark_axis, time_axis = loudness_zwtv(signal, fs, field_type='free')
    N, N_specific, bark_axis, time_axis = loudness_zwtv(
        signal_pa, fs, field_type=field_type
    )

    return time_axis, N, N_specific, bark_axis


# ----------------------------------------------------------------------
# Sharpness (DIN 45692)
# ----------------------------------------------------------------------


def compute_sharpness_st(
    signal_pa: np.ndarray,
    fs: float,
    weighting: str = "din",
    field_type: str = "free",
) -> float:
    """
    Compute stationary sharpness according to DIN 45692.

    Wrapper around `mosqito.sq_metrics.sharpness_din_st`.

    Parameters
    ----------
    signal_pa : np.ndarray
        1D array of calibrated sound pressure samples [Pa].
    fs : float
        Sampling frequency in Hz.
    weighting : {'din', 'aures', 'bismarck', 'fastl'}, optional
        Sharpness weighting method, by default 'din'.
    field_type : {'free', 'diffuse'}, optional
        Type of sound field, by default 'free'.

    Returns
    -------
    S : float
        Sharpness value in acum.
    """
    S = sharpness_din_st(
        signal=signal_pa,
        fs=fs,
        weighting=weighting,
        field_type=field_type,
    )
    return float(S)


def compute_sharpness_tv(
    signal_pa: np.ndarray,
    fs: float,
    weighting: str = "din",
    field_type: str = "free",
    skip: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute time-varying sharpness according to DIN 45692.

    Wrapper around `mosqito.sq_metrics.sharpness_din_tv`.

    Parameters
    ----------
    signal_pa : np.ndarray
        1D array of calibrated sound pressure samples [Pa].
    fs : float
        Sampling frequency in Hz.
    weighting : {'din', 'aures', 'bismarck', 'fastl'}, optional
        Sharpness weighting method, by default 'din'.
    field_type : {'free', 'diffuse'}, optional
        Type of sound field, by default 'free'.
    skip : float, optional
        Number of seconds to skip at the beginning (to ignore transients),
        by default 0.0.

    Returns
    -------
    time_axis : np.ndarray
        Time axis [s], shape (N_seg,).
    S : np.ndarray
        Sharpness [acum] over time, shape (N_seg,).
    """
    # MOSQITO signature:
    # S, time_axis = sharpness_din_tv(signal, fs, weighting='din',
    #                                 field_type='free', skip=0)
    S, time_axis = sharpness_din_tv(
        signal=signal_pa,
        fs=fs,
        weighting=weighting,
        field_type=field_type,
        skip=skip,
    )

    return time_axis, S


# ----------------------------------------------------------------------
# Roughness (Daniel & Weber)
# ----------------------------------------------------------------------


def compute_roughness_dw(
    signal_pa: np.ndarray,
    fs: float,
    overlap: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute roughness according to Daniel & Weber.

    Wrapper around `mosqito.sq_metrics.roughness_dw`.

    Parameters
    ----------
    signal_pa : np.ndarray
        1D array of calibrated sound pressure samples [Pa].
    fs : float
        Sampling frequency in Hz.
    overlap : float, optional
        Overlap factor for 200 ms time windows, in [0, 1).
        Default is 0.5 (50% overlap).

    Returns
    -------
    time_axis : np.ndarray
        Time axis [s], shape (N_time,).
    R : np.ndarray
        Overall roughness [asper], shape (N_time,).
    R_spec : np.ndarray
        Specific roughness [asper/Bark], shape (N_bark, N_time).
    bark_axis : np.ndarray
        Bark axis, shape (N_bark,).

    Notes
    -----
    MoSQITo returns (R, R_spec, bark_axis, time).
    This wrapper reorders the outputs to (time_axis, R, R_spec, bark_axis)
    for consistency with `compute_loudness_zwicker`.
    """
    # MOSQITO signature:
    # R, R_spec, bark_axis, time_axis = roughness_dw(signal, fs, overlap=0.5)
    R, R_spec, bark_axis, time_axis = roughness_dw(
        signal=signal_pa,
        fs=fs,
        overlap=overlap,
    )

    return time_axis, R, R_spec, bark_axis
