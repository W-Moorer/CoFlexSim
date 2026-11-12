"""Regenerate publication figures; timing and memory are reported as tables."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name("plot_publication.py")), run_name="__main__")
