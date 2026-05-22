import numpy as np
import pytest

from hdf_tools_soundscape.head_hdf_utils import (
    P_REF,
    compute_leq_pa,
    compute_rms_spl,
    get_channel_calibration_db_from_text,
)


def test_get_channel_calibration_db_from_text_found():
    header_text = """
    channel definition: 1
      calibration: 93.5
    channel definition: 2
      calibration: 94.0
    """
    assert get_channel_calibration_db_from_text(header_text, channel_def=1) == pytest.approx(93.5)
    assert get_channel_calibration_db_from_text(header_text, channel_def=2) == pytest.approx(94.0)


def test_get_channel_calibration_db_from_text_missing():
    header_text = "channel definition: 3 calibration: 88.0"
    assert get_channel_calibration_db_from_text(header_text, channel_def=1) is None


def test_compute_leq_pa_constant_signal():
    signal = np.ones(1000, dtype=np.float64)
    leq = compute_leq_pa(signal)
    expected = 20.0 * np.log10(1.0 / P_REF)
    assert leq == pytest.approx(expected)


def test_compute_rms_spl_constant_signal():
    fs = 10.0
    signal = np.ones(10, dtype=np.float64)
    t, levels = compute_rms_spl(signal, fs=fs, window_seconds=0.2)

    assert len(t) == 9
    expected = 20.0 * np.log10(1.0 / P_REF)
    assert np.allclose(levels, expected)


def test_compute_rms_spl_small_window_raises():
    fs = 48_000.0
    signal = np.ones(100, dtype=np.float64)
    with pytest.raises(ValueError, match="window_seconds too small"):
        compute_rms_spl(signal, fs=fs, window_seconds=0.0)
