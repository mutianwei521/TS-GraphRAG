# README visuals

`teaser.png` is a compact project overview. Its builder is `scripts/readme/build_teaser.py`; `teaser_provenance.json` identifies the source network, partition, fingerprints and example inference. The projection geometry is schematic.

The remaining PNGs reuse approved paper exports without changing their pixels or numerical content:

| Asset | Paper figure | Rebuild command |
|---|---|---|
| `methodology.png` | Fig. 2 | `python scripts/figures/replay.py 2 --output outputs/readme_figures` |
| `baselines.png` | Fig. 4 | `python scripts/figures/replay.py 4 --output outputs/readme_figures` |
| `allocation.png` | Fig. 5 | `python scripts/figures/replay.py 5 --output outputs/readme_figures` |

`paper_figure_provenance.json` records their SHA-256 values. The README distinguishes the repeated reference benchmark, fixed development/test evaluation and allocation study. Their query sets and repeat units remain separate.
