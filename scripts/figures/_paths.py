"""Portable paths and checked staging for the approved plotting sources."""
from pathlib import Path
import hashlib
import json
import os
import shutil

REPOSITORY = Path(__file__).resolve().parents[2]
INPUT = REPOSITORY / "source_data/figure_inputs"
OUTPUT = Path(os.environ.get("TS_GRAPHRAG_FIGURE_OUTPUT", REPOSITORY / "results/figures")).resolve()
FIGDIR = OUTPUT / "figures"
SD = OUTPUT / "source_data"
QA = OUTPUT / "qa"
for directory in [FIGDIR, SD, QA, OUTPUT / "notes"]:
    directory.mkdir(parents=True, exist_ok=True)

# Check the public frozen inputs before each replay. Generated data go to OUTPUT.
manifest = json.loads((REPOSITORY / "source_data/figure_input_manifest.json").read_text(encoding="utf-8"))
for entry in manifest["files"]:
    path = REPOSITORY / entry["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError(f"Changed frozen figure input: {entry['file']}")
for path in (INPUT / "plot_tables").glob("*"):
    if path.is_file():
        shutil.copy2(path, SD / path.name)


def prepare_fig2() -> Path:
    target = OUTPUT / "fig2_build"
    for folder in ["assets", "source_data"]:
        shutil.copytree(INPUT / "fig2" / folder, target / folder, dirs_exist_ok=True)
    return target
