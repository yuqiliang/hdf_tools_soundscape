from reportlab.lib.pagesizes import letter
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit

OUTPUT = "output/pdf/hdf_tools_soundscape_app_summary.pdf"

PAGE_W, PAGE_H = letter
LEFT = 44
RIGHT = PAGE_W - 44
TOP = PAGE_H - 42
BOTTOM = 42
CONTENT_W = RIGHT - LEFT

TITLE_COLOR = HexColor("#133B5C")
HEAD_COLOR = HexColor("#1F2937")
TEXT_COLOR = HexColor("#111827")
MUTED = HexColor("#4B5563")


def draw_heading(c, y, text):
    c.setFillColor(HEAD_COLOR)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(LEFT, y, text)
    return y - 14


def draw_paragraph(c, y, text, font="Helvetica", size=9, leading=12, color=TEXT_COLOR):
    c.setFillColor(color)
    c.setFont(font, size)
    lines = simpleSplit(text, font, size, CONTENT_W)
    for line in lines:
        c.drawString(LEFT, y, line)
        y -= leading
    return y


def draw_bullets(c, y, bullets, size=9, leading=11):
    c.setFillColor(TEXT_COLOR)
    c.setFont("Helvetica", size)
    for b in bullets:
        wrapped = simpleSplit(b, "Helvetica", size, CONTENT_W - 14)
        c.drawString(LEFT, y, "- " + wrapped[0])
        y -= leading
        for cont in wrapped[1:]:
            c.drawString(LEFT + 11, y, cont)
            y -= leading
    return y


def main():
    c = canvas.Canvas(OUTPUT, pagesize=letter)

    y = TOP
    c.setFillColor(TITLE_COLOR)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(LEFT, y, "hdf_tools_soundscape: One-page App Summary")
    y -= 16
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 8)
    c.drawString(LEFT, y, "Evidence basis: README.md, pyproject.toml, head_hdf_utils.py, psychoacoustics.py, head_hdf_utils_demo.ipynb")
    y -= 18

    y = draw_heading(c, y, "What it is")
    y = draw_paragraph(
        c,
        y,
        "A Python utility toolkit for working with HEAD acoustics .hdf time-data files in soundscape research. "
        "It focuses on extracting calibrated stereo audio, computing acoustic metrics, and creating analysis plots.",
    )
    y -= 4

    y = draw_heading(c, y, "Who it is for")
    y = draw_paragraph(
        c,
        y,
        "Primary persona: soundscape or acoustics researchers/analysts using HEAD measurement files and Python workflows.",
    )
    y -= 4

    y = draw_heading(c, y, "What it does")
    y = draw_bullets(
        c,
        y,
        [
            "Inspects binary HDF headers with ASCII and HEX previews.",
            "Parses header metadata: start-of-data, channel count, scan count, delta value, and sampling rate.",
            "Extracts per-channel calibration values (dB) from header text.",
            "Reads Left/Right float32 channels; optionally applies calibration to output sound pressure in Pa.",
            "Exports stereo and mono WAV files via scipy.io.wavfile.",
            "Computes broadband Leq and short-time RMS dB SPL levels.",
            "Plots Mark Analyzer-style waveform plus Level-vs-Time charts; includes psychoacoustic wrappers (Zwicker loudness, DIN sharpness, Daniel-Weber roughness) through MoSQITo.",
        ],
    )
    y -= 4

    y = draw_heading(c, y, "How it works (repo-evidenced architecture)")
    y = draw_bullets(
        c,
        y,
        [
            "Components: head_hdf_utils.py (HDF parsing, calibration, WAV export, SPL metrics, plotting), psychoacoustics.py (MoSQITo wrappers), demo notebook for end-to-end usage.",
            "Libraries: numpy/scipy for signal I/O and processing; matplotlib for plots; mosqito for psychoacoustic metrics.",
            "Data flow: .hdf file -> header parse + binary float32 read -> Left/Right arrays -> optional calibration to Pa -> metrics/plots and optional WAV/PNG outputs.",
            "Services or networked backends: Not found in repo.",
            "Persistent database/storage layer beyond local files: Not found in repo.",
        ],
        leading=10.5,
    )
    y -= 4

    y = draw_heading(c, y, "How to run (minimal getting started)")
    y = draw_bullets(
        c,
        y,
        [
            "Install dependencies: run `uv sync` (README evidence).",
            "Use the demo workflow in `head_hdf_utils_demo.ipynb` and set your HDF path in notebook cells.",
            "Run notebook cells that call `read_head_file(...)`, `plot_mark_style(...)`, and optional psychoacoustic functions.",
            "Dedicated CLI entrypoint command: Not found in repo.",
            "Formal test command documentation: Not found in repo.",
        ],
    )

    if y < BOTTOM:
        raise RuntimeError(f"Content overflow: y={y} < bottom={BOTTOM}")

    c.showPage()
    c.save()
    print(OUTPUT)


if __name__ == "__main__":
    main()
