# Source Data

The public Source Data contain all **36 worksheets** of the reviewed workbook,
all 28 current Supplementary Information tables, the inputs used by the figure
scripts, and the matching figure captions. The numerical workbook source is
`0829/reviewer_closure_20261010/Source_Data.xlsx`; its actual imported SHA256 and
sheet order are recorded in `workbook_manifest.json`.

```console
python scripts/build_source_data.py --output results/source_data
python scripts/figures/replay.py all
python scripts/figures/replay.py S6-standalone
```

These commands rebuild the workbook or redraw figures from frozen observations.
The experiment commands in
[`figure_table_map.csv`](../reproducibility/figure_table_map.csv) identify the
separate simulation, fitting and scoring workflows. An archived hosted-language-
model summary supports reconstruction of its reported estimates; plotting it
does not repeat the original hosted requests.

## Worksheet representation

Each `sheets/NAME.json` contains the worksheet's complete rectangular cell array,
native cell types, dimensions, header rule, number formats, merged ranges,
column widths and row heights. Each corresponding CSV contains exactly the same
rows, including row 1, blank rows, and notes. Only row 1 is designated as a header;
later rows must not automatically be treated as observations. JSON distinguishes
empty strings from null cells and preserves text identifiers such as sensor node
names. JSON numbers are numeric measurements rather than preformatted strings.
Publication `mean ± SD` strings are also preserved verbatim; the associated
replicate/summary worksheets provide their numeric values. Worksheet JSON is
authoritative for types; CSV is an accessible view.

`build_source_data.py` checks CSV/JSON consistency and frozen input hashes, writes
the workbook using openpyxl, then reopens it to compare every native cell. It
recalculates the nine Table S17 publication rows from
`data/frozen/node_selection/test_results.json` and `test_masked_results.json`.
Unrounded percentages and paired McNemar P values are also written to
`results/source_data/TableS17_numeric.csv`. Workbook presentation styles may
differ from the reviewed original; cell values, sheet order, blank/note rows and
merged ranges follow the recorded source.

## Table and figure versions

`manuscript_tables/` supplies text-cell CSV/JSON snapshots of all 28 tables from
the current SI, including the configuration tables, Table S1 weights, corpus
compositions and implemented baseline settings. The manifest records source
document hash and native table index. Tables are named by scientific content:
S16 is inference sensitivity; S17 is the node-held-out estimator test. Native
preceding paragraph text remains in the metadata so caption linkage can be
inspected independently of numerical content.

`scripts/figures/` preserves the approved figure layouts and transformations.
Figure 1 uses the dense October design with the strict H8 summary annotation;
Figure 2 uses the October graphic and its original RGBA assets. Figure 3's
reviewed vector artwork is supplied; optional regeneration accepts separately
obtained municipal inputs. The integrated Fig. S6 is the 34-setting heatmap.
The 17 October standalone seven-panel Fig. S6 has a separate command and matching
caption and is explicitly marked as a standalone revision in the map.

## Statistical and physical definitions

- E4 reference comparisons use five complete runs on fixed 540- or 90-query
  identities, varying measurement noise and stochastic fitting. E5 allocation
  uses 15 corpus/query-location draws and one shared perturbation vector per
  draw. These repeat units are distinct.
- Strict fusion uses five measurement-noise realizations on the same 540
  queries, retained-artifact rebuilding, and full eligible-ranking reciprocal
  rank. Historical allocation reports MRR@5. H9 is a candidate-support integrity
  condition. The attribution data belong to Note 1.9, whereas S13 contains the
  hydraulic study.
- Hydraulic mismatch reports all-planned-query correct-and-admissible success,
  conditional accuracy, effective seed counts and separate service-junction
  diagnostics. Its projection coefficient is unrestricted.
- Partition/sensor controls use five paired split/design/noise runs, 156
  held-out-node queries per configuration, a nonnegative fitted coefficient and
  explicit abstentions. Regional labels can change while node predictions stay
  fixed. Candidate-pipe length is a set-extent measure.
- Historical Euclidean map-coordinate distances have no established conversion
  to metres. New partition/sensor distances follow physical pipe lengths in
  metres. A pressure perturbation, a leak rate and a localization distance are
  different quantities and retain their original units.

No private spreadsheet or document runtime is required. Maintainers can refresh
the worksheet representation from a newly reviewed 36-sheet workbook with
`python scripts/build_source_data.py --import-workbook PATH`.
