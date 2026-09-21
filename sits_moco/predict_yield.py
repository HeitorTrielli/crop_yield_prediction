"""Backward-compatible shim — prefer ``python eval/predict_yield.py``."""
from path_setup import ensure_repo_on_path
import runpy
from pathlib import Path

ensure_repo_on_path()
runpy.run_path(str(Path(__file__).resolve().parent / "eval" / "predict_yield.py"), run_name="__main__")
