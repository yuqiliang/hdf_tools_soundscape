# hdf_tools_soundscape

Utilities for processing HEAD acoustics time data (`.hdf`) in soundscape research workflows.

## What Is Included

- `src/hdf_tools_soundscape/head_hdf_utils.py`
  - Header inspection (ASCII / HEX preview)
  - Header parsing (start offset, channels, scans, delta, calibration)
  - Left/Right audio extraction with optional calibration to Pa
  - WAV export (stereo / mono)
  - Leq, LAeq, and short-time RMS SPL calculations
  - Mark Analyzer-style plotting (waveform + level vs. time)
- `src/hdf_tools_soundscape/psychoacoustics.py`
  - MoSQITo-based loudness, sharpness, and roughness wrappers
- `head_hdf_utils_demo.ipynb`
  - Notebook example workflow

## Project Structure

```text
hdf_tools_soundscape/
├── src/
│   └── hdf_tools_soundscape/
│       ├── __init__.py
│       ├── head_hdf_utils.py
│       └── psychoacoustics.py
├── tests/
├── head_hdf_utils_demo.ipynb
├── pyproject.toml
└── Makefile
```

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
```

For development tools (`pytest`, `ruff`, `pre-commit`):

```bash
uv sync --extra dev
```

## Development Workflow

Run tests:

```bash
make test
```

Run lint:

```bash
make lint
```

Format code:

```bash
make format
```

Run all checks:

```bash
make check
```

Enable pre-commit hooks:

```bash
uv run pre-commit install
```

## Minimal Usage Example

```python
from hdf_tools_soundscape import compute_laeq_pa, parse_header_info, read_head_file, plot_mark_style

info = parse_header_info("path/to/file.hdf")
audio, fs, info = read_head_file("path/to/file.hdf", apply_calibration=True)
laeq_left = compute_laeq_pa(audio[:, 0], fs)
laeq_right = compute_laeq_pa(audio[:, 1], fs)
plot_mark_style(audio, fs, show=True)
```

## Notes

- Calibration-dependent analysis should be validated against HEAD software for critical work.
- Large local sample `.hdf` files are intentionally not tracked in git by default.
