"""head_hdf_utils
====================

Utilities for working with HEAD acoustics time data *.hdf files.

Features
--------
- Inspect the ASCII / HEX content of the binary header.
- Parse per-channel calibration (dB) from the header.
- Parse basic sampling info (start of data, number of channels, number of scans, delta value).
- Read Left / Right audio channels and optionally apply calibration to obtain sound pressure in Pa.
- Export stereo and mono WAV files.
- Compute Leq and short-time RMS levels in dB SPL.
- Plot a Mark Analyzer-style 2-panel figure: waveform + Level vs. Time.

Notes
-----
This module is tailored to HEAD time data files similar to those produced by SQobold /
SQuadriga, where the header is text-like followed by binary FLOAT32 samples.

You should still verify results against HEAD software for critical analysis.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt


# Standard reference sound pressure for dB SPL
P_REF: float = 2e-5  # 20 µPa


@dataclass
class HeadHDFInfo:
    """Parsed metadata from the HEAD HDF header.

    Attributes
    ----------
    start_of_data : int
        Byte offset where the binary data section begins.
    num_channels : int
        Number of channels in the binary data (e.g. 3 for Left, Right, GPS).
    num_scans : int
        Number of samples per channel ("nbr of scans").
    delta_value : float
        Abscissa delta value in seconds (1 / sampling rate).
    fs : float
        Sampling rate in Hz (computed as 1 / delta_value if available).
    calibration_left_db : Optional[float]
        Calibration in dB for channel definition 1 (usually Left).
    calibration_right_db : Optional[float]
        Calibration in dB for channel definition 2 (usually Right).
    raw_header_text : str
        Header text (decoded ASCII with replacement for undecodable bytes).
    """

    start_of_data: int = 65536
    num_channels: int = 3
    num_scans: Optional[int] = None
    delta_value: Optional[float] = None
    fs: Optional[float] = None
    calibration_left_db: Optional[float] = None
    calibration_right_db: Optional[float] = None
    raw_header_text: str = ""


def _read_header_bytes(filepath: str, header_bytes: int = 65536) -> bytes:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, "rb") as f:
        header = f.read(header_bytes)
    return header


def inspect_head_hdf(filepath: str, n_bytes: int = 4096) -> None:
    """Print an ASCII and HEX preview of the first *n_bytes* of a HEAD *.hdf* file.

    Parameters
    ----------
    filepath : str
        Path to the *.hdf* file.
    n_bytes : int, optional
        Number of bytes to show from the beginning of the file, by default 4096.
    """
    header = _read_header_bytes(filepath, header_bytes=n_bytes)

    print("=== ASCII Preview ===")
    print(header.decode(errors="replace"))

    print("\n=== HEX Preview ===")
    print(header.hex(" "))


def get_channel_calibration_db_from_text(
    header_text: str,
    channel_def: int = 1,
) -> Optional[float]:
    """Parse calibration(dB) from an already-decoded header text.

    Parameters
    ----------
    header_text : str
        ASCII-decoded header.
    channel_def : int
        Channel definition index (1 = Left, 2 = Right, etc.).

    Returns
    -------
    float or None
        Calibration in dB if found, otherwise None.
    """
    pattern = rf"channel definition:\s*{channel_def}.*?calibration:\s*([0-9\.\+\-Ee]+)"
    m = re.search(pattern, header_text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def get_channel_calibration_db(
    filepath: str,
    channel_def: int = 1,
    header_bytes: int = 65536,
) -> Optional[float]:
    """Convenience wrapper: parse calibration(dB) for a given channel from a file.

    Parameters
    ----------
    filepath : str
        Path to the *.hdf* file.
    channel_def : int
        Channel definition index (1 = Left, 2 = Right, etc.).
    header_bytes : int
        Number of bytes to read for the header.

    Returns
    -------
    float or None
        Calibration in dB, or None if it cannot be parsed.
    """
    header = _read_header_bytes(filepath, header_bytes=header_bytes)
    text = header.decode(errors="replace")
    return get_channel_calibration_db_from_text(text, channel_def=channel_def)


def parse_header_info(filepath: str, header_bytes: int = 65536) -> HeadHDFInfo:
    """Parse basic info and calibration values from a HEAD HDF header.

    Parameters
    ----------
    filepath : str
        Path to the *.hdf* file.
    header_bytes : int, optional
        Number of bytes to read as header. For HEAD time data, 65536 is typical.

    Returns
    -------
    HeadHDFInfo
        Dataclass containing parsed information (with sensible fallbacks).
    """
    header = _read_header_bytes(filepath, header_bytes=header_bytes)
    text = header.decode(errors="replace")

    info = HeadHDFInfo(raw_header_text=text)

    # start of data
    m = re.search(r"start of data:\s*([0-9]+)", text)
    if m:
        info.start_of_data = int(m.group(1))

    # number of channels
    m = re.search(r"nbr of channel:\s*([0-9]+)", text)
    if m:
        info.num_channels = int(m.group(1))

    # number of scans (samples per channel)
    m = re.search(r"nbr of scans:\s*([0-9]+)", text)
    if m:
        info.num_scans = int(m.group(1))

    # delta value (seconds)
    m = re.search(r"delta value:\s*([0-9eE\+\-\.]+)", text)
    if m:
        try:
            info.delta_value = float(m.group(1))
            if info.delta_value != 0:
                info.fs = 1.0 / info.delta_value
        except ValueError:
            pass

    # calibration for channel definition 1 and 2
    info.calibration_left_db = get_channel_calibration_db_from_text(text, channel_def=1)
    info.calibration_right_db = get_channel_calibration_db_from_text(text, channel_def=2)

    return info


def _infer_num_scans(
    filepath: str,
    start_of_data: int,
    num_channels: int,
    dtype_size: int = 4,
) -> int:
    """Infer the number of scans from file size when not explicitly available.

    Assumes the rest of the file is contiguous samples with given channel count
    and dtype size (4 bytes for float32).
    """
    file_size = os.path.getsize(filepath)
    data_bytes = file_size - start_of_data
    if data_bytes <= 0:
        raise ValueError("No data bytes found after header.")
    total_values = data_bytes // dtype_size
    if total_values % num_channels != 0:
        raise ValueError(
            f"File size not divisible by num_channels={num_channels} (total float32 values = {total_values})."
        )
    return total_values // num_channels


def read_head_file(
    filepath: str,
    output_stereo: Optional[str] = None,
    output_left: Optional[str] = None,
    output_right: Optional[str] = None,
    apply_calibration: bool = True,
    header_bytes: int = 65536,
) -> Tuple[np.ndarray, float, HeadHDFInfo]:
    """Read a HEAD acoustics *.hdf* data file and optionally save audio as WAV files.

    This function:
    - Parses header information (start of data, number of channels, scans, delta, calibration).
    - Reads Left/Right channels from the binary section as float32.
    - Optionally applies calibration(dB) to convert raw values into Pa.
    - Optionally exports stereo and mono WAV files.

    Parameters
    ----------
    filepath : str
        Path to the *.hdf* binary data file.
    output_stereo : str or None
        If not None, path to save a 2-channel (Left/Right) WAV file.
    output_left : str or None
        If not None, path to save Left channel as a mono WAV file.
    output_right : str or None
        If not None, path to save Right channel as a mono WAV file.
    apply_calibration : bool
        If True, convert raw values to Pa using calibration(dB) from the header.
        If False, return raw float32 values (cast to float64) as stored in the file.
    header_bytes : int
        Number of bytes to treat as header. If the header explicitly specifies
        "start of data", that value overrides this for the data offset.

    Returns
    -------
    audio : np.ndarray
        Array of shape (num_samples, 2) for Left and Right channels.
        If apply_calibration=True, unit is Pa; otherwise, raw (dimensionless).
    fs : float
        Sampling rate in Hz.
    info : HeadHDFInfo
        Parsed metadata from the header.
    """
    info = parse_header_info(filepath, header_bytes=header_bytes)

    start_of_data = info.start_of_data
    num_channels = info.num_channels

    # If num_scans is not available, infer from file size
    if info.num_scans is None:
        info.num_scans = _infer_num_scans(
            filepath, start_of_data=start_of_data, num_channels=num_channels
        )

    # If fs is not available (no delta value), fall back to 48 kHz
    if info.fs is None:
        info.fs = 48000.0

    # --- Read raw float32 samples from data section
    with open(filepath, "rb") as f:
        f.seek(start_of_data)
        raw = np.fromfile(f, dtype=np.float32)

    expected_values = info.num_scans * num_channels
    if raw.size < expected_values:
        raise ValueError(
            f"Data size too small: expected at least {expected_values} float32 values, got {raw.size}."
        )
    # If there are extra values, truncate
    if raw.size > expected_values:
        raw = raw[:expected_values]

    # Reshape to (channels, samples)
    raw = raw.reshape(info.num_scans, num_channels).T

    left_raw = raw[0, :]
    right_raw = raw[1, :]

    # --- Apply calibration (raw -> Pa) or not
    if apply_calibration:
        cal_left = info.calibration_left_db if info.calibration_left_db is not None else 114.0
        cal_right = info.calibration_right_db if info.calibration_right_db is not None else 114.0

        scale_left = P_REF * 10.0 ** (cal_left / 20.0)
        scale_right = P_REF * 10.0 ** (cal_right / 20.0)

        left = left_raw.astype(np.float64) * scale_left
        right = right_raw.astype(np.float64) * scale_right
    else:
        left = left_raw.astype(np.float64)
        right = right_raw.astype(np.float64)

    audio = np.vstack((left, right)).T
    fs = float(info.fs)
    fs_int = int(round(fs))

    # --- Export WAVs if requested
    if output_stereo is not None:
        write(output_stereo, fs_int, audio.astype(np.float32))
        print(f"Saved stereo WAV to {output_stereo}")

    if output_left is not None:
        write(output_left, fs_int, left.astype(np.float32))
        print(f"Saved Left channel WAV to {output_left}")

    if output_right is not None:
        write(output_right, fs_int, right.astype(np.float32))
        print(f"Saved Right channel WAV to {output_right}")

    return audio, fs, info


def compute_leq_pa(signal_pa: np.ndarray, pref: float = P_REF) -> float:
    """Compute broadband Leq (dB SPL) for a signal given in Pascals.

    Parameters
    ----------
    signal_pa : np.ndarray
        Signal in Pascals.
    pref : float
        Reference sound pressure (default = 2e-5 Pa).

    Returns
    -------
    float
        Leq in dB SPL.
    """
    signal_pa = np.asarray(signal_pa, dtype=np.float64)
    # Clip extreme spikes to avoid overflow
    signal_pa = np.clip(signal_pa, -1e6, 1e6)
    rms = np.sqrt(np.mean(signal_pa**2))
    if rms <= 0 or np.isnan(rms):
        return float("-inf")
    return 20.0 * np.log10(rms / pref)


def compute_rms_spl(
    signal_pa: np.ndarray,
    fs: float,
    window_seconds: float = 0.125,
    pref: float = P_REF,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute short-time RMS level vs. time, similar to Mark Analyzer's Level vs. Time.

    Parameters
    ----------
    signal_pa : np.ndarray
        1D signal in Pascals.
    fs : float
        Sampling rate in Hz.
    window_seconds : float
        RMS window length in seconds (e.g. 0.125 s ~ 'Fast').
    pref : float
        Reference pressure in Pa.

    Returns
    -------
    t : np.ndarray
        Time axis (seconds) at the center of each RMS window.
    L : np.ndarray
        Level in dB SPL for each window.
    """
    signal_pa = np.asarray(signal_pa, dtype=np.float64)
    signal_pa = np.clip(signal_pa, -1e6, 1e6)

    win_len = int(round(window_seconds * fs))
    if win_len < 1:
        raise ValueError("window_seconds too small; window length < 1 sample.")

    sq = signal_pa**2
    kernel = np.ones(win_len, dtype=np.float64) / win_len
    rms = np.sqrt(np.convolve(sq, kernel, mode="valid"))

    # Avoid log of zero
    rms = np.where(rms <= 0, np.nan, rms)
    L = 20.0 * np.log10(rms / pref)
    t = (np.arange(len(rms)) + win_len / 2) / fs

    return t, L


def plot_mark_style(
    audio: np.ndarray,
    fs: float,
    window_seconds: float = 0.125,
    channel_labels: Tuple[str, str] = ("Left", "Right"),
    title_prefix: Optional[str] = None,
    show: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a Mark Analyzer-style 2-panel figure: waveform + Level vs. Time.

    Parameters
    ----------
    audio : np.ndarray
        Array of shape (num_samples, 2), typically output of :func:`read_head_file`.
        Values are assumed to be in Pa if calibration was applied.
    fs : float
        Sampling rate in Hz.
    window_seconds : float
        Short-time RMS window length in seconds.
    channel_labels : tuple of str
        Labels for the two channels (used in legend).
    title_prefix : str or None
        Optional prefix added to the figure title(s).
    show : bool
        If True, display the figure via ``plt.show()``.
    save_path : str or None
        If not None, path to save the figure (e.g. PNG).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    """
    if audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError("audio must be a 2D array with shape (num_samples, 2).")

    left = audio[:, 0]
    right = audio[:, 1]
    n = len(left)
    t_wave = np.arange(n) / fs

    # Overall Leq
    leq_left = compute_leq_pa(left)
    leq_right = compute_leq_pa(right)

    # Short-time RMS levels
    t_L, L_left = compute_rms_spl(left, fs, window_seconds=window_seconds)
    _,   L_right = compute_rms_spl(right, fs, window_seconds=window_seconds)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

    # Waveform
    ax1.plot(t_wave, left, label=channel_labels[0])
    ax1.plot(t_wave, right, label=channel_labels[1], alpha=0.7)
    ax1.set_ylabel("p [Pa]")
    wf_title = "Waveform (Left / Right)"
    if title_prefix:
        wf_title = f"{title_prefix} - " + wf_title
    ax1.set_title(wf_title)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    # Level vs. Time
    ax2.plot(t_L, L_left, label=channel_labels[0])
    ax2.plot(t_L, L_right, label=channel_labels[1], alpha=0.7)
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Level [dB SPL]")
    lvl_title = f"Level vs. Time (window = {window_seconds*1000:.0f} ms)"
    if title_prefix:
        lvl_title = f"{title_prefix} - " + lvl_title
    ax2.set_title(lvl_title)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    # Add overall Leq text box
    text = f"{channel_labels[0]} Leq = {leq_left:.2f} dB(SPL)\n{channel_labels[1]} Leq = {leq_right:.2f} dB(SPL)"
    ax2.text(
        0.98,
        0.02,
        text,
        transform=ax2.transAxes,
        ha="right",
        va="bottom",
    )

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        print(f"Saved figure to {save_path}")

    if show:
        plt.show()

    return fig
