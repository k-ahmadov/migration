"""Project-anchored filesystem locations.

Paths are resolved relative to the repository root (the parent of the
``fracinj`` package), so scripts work regardless of the current working
directory.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
OVERLEAF = ROOT / "overleaf"


def results_dir(*parts: str) -> Path:
    return RESULTS.joinpath(*parts)


def figure_path(name: str, *, ext: str = "eps", overleaf: bool = False) -> Path:
    """Location for a figure file, creating the parent directory."""
    base = OVERLEAF / "figures_main" if overleaf else FIGURES / "paper"
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{name}.{ext}"
