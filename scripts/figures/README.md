# Approved figure replay

```console
python scripts/figures/replay.py all
python scripts/figures/replay.py 4 6 S3
python scripts/figures/replay.py S6-standalone
python scripts/figures/replay.py all --output results/figure_replay
```

The scripts reuse the approved Python layouts and all plotted observations.
Changes provide repository-relative checked inputs and a writable output folder.
The Figure 1 H8 annotation reads the strict five-seed summary. Scientific
transformations, mark positions and uncertainty definitions follow the source
versions; this release does not redesign the figures.

Inputs live in `source_data/figure_inputs/` and their hashes/provenance are in
`source_data/figure_input_manifest.json`. `_paths.py` verifies these inputs and
stages writable copies of the small plotting tables under the output folder.
Generated files never overwrite the public frozen data. Outputs normally go to
`results/figures/figures/`; Figure 2 uses
`results/figures/fig2_build/figures/`, and the standalone Fig. S6 uses
`results/figures/figS6_standalone_20261017/`. Scripts also export plot data and
layout checks. Matching current and standalone captions are in
`source_data/captions/`.

## Figure 3 and municipal inputs

The default Figure 3 command copies the reviewed SVG/PDF artwork. Full municipal
network models are separately obtained inputs. Regenerate the three maps with:

```console
python scripts/figures/replay.py 3 --municipal-data-dir PATH
```

That directory must contain `dagang.inp`, `haining.inp`,
`dagang_partitions.pkl`, and `haining_partitions.pkl`. The partition files follow
the original trusted Python pickle format: key `15`, then `node_to_community`.
The fixed EXA7 inputs and the published municipal sensor assignments are bundled.
The optional map script requires WNTR and shapely.

## Versions and evidence

- Main Figures 1–2 use the approved October dense/graphic versions.
- Figure 4 and Fig. S1 use the approved 13 September observation-point versions;
  all 690 and 300 observations respectively are retained.
- Figure 6b uses the strict retained-pool rerun. Panels a/c/d preserve the
  historical single realization and archived language-model experiment.
- Figs. S3–S5 use the corrected audited hydraulic/design outputs.
- `S6` is the integrated SI heatmap; `S6-standalone` is the separate 17 October
  seven-panel revision. Both use all 34 settings, four protocols and five seeds.

The plotting contract is exact reuse: retain numerical transformations,
categories, means, sample SDs, pairing and output dimensions. Figure-level QA
records exported text bounds, observation counts or independent source checks
as implemented by each approved script. Rendering a figure verifies the plotting
replay; experiment reproduction is documented separately in the figure/table map.

The verified environment uses Python with numpy, pandas, matplotlib, scipy,
Pillow, openpyxl and pypdf 6.12.2. Figure 1 also reads the EXA7 INP with WNTR.
PDF/SVG preserve editable text; PNG exports use the approved 600 dpi settings.
