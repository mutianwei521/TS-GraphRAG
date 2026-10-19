"""Replay the approved figures from public frozen observations.

This command draws figures, rather than rerunning simulations, model training,
or hosted LLM requests. Figure 3 uses the approved vector artwork by default;
its optional regeneration needs separately obtained municipal models.
"""
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
SCRIPTS = {
    "1": "make_fig1_infographic.py", "2": "compose_fig2.py",
    "4": "make_fig4_holdout.py", "5": "make_fig5_allocation.py",
    "6": "make_fig6_components.py", "7": "make_fig7_robustness.py",
    "S1": "make_figS1_drawlevel.py", "S2": "make_figS2_confusion.py",
    "S3": "make_figS3.py", "S4": "make_figS4_S5.py", "S5": "make_figS4_S5.py",
    "S6": "make_figS6_current.py", "S6-standalone": "make_figS6_standalone.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figures", nargs="*", default=["all"],
                        help="1-7, S1-S6, S6-standalone, or all (current manuscript/SI)")
    parser.add_argument("--output", type=Path, default=REPOSITORY / "results/figures")
    parser.add_argument("--municipal-data-dir", type=Path,
                        help="Regenerate Fig. 3 using separately obtained municipal inputs")
    args = parser.parse_args()
    figures = (["1", "2", "3", "4", "5", "6", "7", "S1", "S2", "S3", "S4", "S5", "S6"]
               if "all" in args.figures else args.figures)
    unknown = set(figures) - (set(SCRIPTS) | {"3"})
    if unknown:
        parser.error("Unknown figures: " + ", ".join(sorted(unknown)))
    environment = os.environ.copy()
    environment["TS_GRAPHRAG_FIGURE_OUTPUT"] = str(args.output.resolve())
    environment["PYTHONIOENCODING"] = "utf-8"
    completed = set()
    for figure in figures:
        if figure == "3" and not args.municipal_data_dir:
            target = args.output / "figures"
            target.mkdir(parents=True, exist_ok=True)
            for extension in ["svg", "pdf"]:
                name = f"Fig3_partition_maps.{extension}"
                shutil.copy2(REPOSITORY / "source_data/artwork" / name, target / name)
            print("Fig. 3: approved vector artwork copied; municipal-model regeneration is optional.")
            continue
        script = "make_fig3_maps.py" if figure == "3" else SCRIPTS[figure]
        if script in completed:
            continue
        command = [sys.executable, str(HERE / script)]
        if figure == "3":
            command += ["--municipal-data-dir", str(args.municipal_data_dir.resolve())]
        subprocess.run(command, check=True, env=environment, cwd=REPOSITORY)
        completed.add(script)
    print(f"Figure replay complete: {args.output.resolve()}")


if __name__ == "__main__":
    main()
