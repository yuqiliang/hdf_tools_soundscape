import pytest

from hdf_tools_soundscape.psychoacoustics import _normalize_sampling_rate


def test_normalize_sampling_rate_rounds_header_float_noise():
    assert _normalize_sampling_rate(48000.00000000008) == 48000
    assert _normalize_sampling_rate(44100.000000000095) == 44100


def test_normalize_sampling_rate_keeps_non_integer_rate():
    assert _normalize_sampling_rate(48000.25) == 48000.25


def test_normalize_sampling_rate_rejects_invalid_rate():
    with pytest.raises(ValueError, match="fs must be positive"):
        _normalize_sampling_rate(0.0)
