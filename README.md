# hdf_tools_soundscape

Utilities for working with HEAD acoustics time data (`.hdf`) files in the context of
soundscape research.

This repo provides:

- A Python utility module `head_hdf_utils.py` to:
  - Inspect the text-like header of HEAD `.hdf` files (ASCII / HEX preview)
  - Parse sampling info (start of data, number of channels, number of scans, delta value)
  - Parse per-channel calibration (dB) from the header
  - Read Left / Right audio channels and (optionally) convert to physical sound pressure in Pa
  - Export stereo and mono WAV files
  - Compute broadband Leq and short-time RMS levels in dB SPL
  - Plot Mark Analyzer-style 2-panel figures (waveform + Level vs. Time)
- A demo Jupyter notebook showing typical workflows.

---

## Installation

Create and activate a virtual environment if you like, then install dependencies:

```bash
pip install -r requirements.txt
