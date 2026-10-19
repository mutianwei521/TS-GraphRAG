# Historical comparison studies

```text
python -m experiments.historical.run --profile replay --output-dir outputs/historical/replay
python -m experiments.historical.run --profile full --output-dir outputs/historical/full
```

Replay independently recalculates all 80 physics-baseline accuracy cells (Table S5) from archived query correctness and all 11 exact McNemar comparisons with the original Holm family (Table S6). It also recalculates population-weighted LLM recovery/breakage statistics from saved counts and verifies emitter-retrieval accuracy from saved query outcomes. Full freshly executes the eight physical-dictionary methods across all 540 reference queries and ten deletion protocols (43,200 predictions). It compares query outcomes against the archived physics study. Other historical components remain explicitly identified as archived statistics.

The physical methods are the original projection residual, cosine 1-NN, WLS/NNLS, shrinkage Mahalanobis prototype, scale-marginalized Bayes dictionary, normalized-fingerprint KDE, rank-weighted votes with adjacency smoothing, and clamped label propagation. They read the reference fingerprint and partition adjacency matrices directly, with the original `RandomState(41 + original_query_index)` observation noise. Historical cosine includes zero fingerprints as the original normalized dictionary did; repeated E4 cosine excludes them. This distinction is retained.

For individual physics methods or protocols:

```text
python -m experiments.historical.physics --methods projection "cosine 1-NN" --protocols H0 H2 H8 --output-dir outputs/historical/projection
```

The EXA7 studies behind S8–S10 and the corrected resolution analysis have separate executable entry points:

```text
python -m experiments.historical.exa7 --study pressure_dependent --output-dir outputs/historical/pressure_dependent
python -m experiments.historical.exa7 --study sensor_degradation --output-dir outputs/historical/sensor_degradation
python -m experiments.historical.exa7 --study resolution --output-dir outputs/historical/resolution
python -m experiments.historical.exa7 --study multileak_exhaustive --output-dir outputs/historical/multileak_exhaustive
python -m experiments.historical.exa7 --study multileak_sparse --output-dir outputs/historical/multileak_sparse
```

These commands preserve the original full query generation, seeds, estimators and settings. Inputs can be selected explicitly with `--inp`, `--reference-dir`, `--frozen-dir`, `--sensitivity` and `--dense-corpus`. The default startpoint is the distributed sensor/partition design and `data/reference/sensitivity_matrix.npz`. Pressure-dependent retrieval also uses the original 20 L/s prescribed-demand dense corpus, `data/frozen/historical/exp2a_dense_corpus.npz`, and original held-node design from `exp1b_results.json`. It freshly simulates emitter dictionaries, emitter queries and the prescribed-demand control. Sensor degradation freshly simulates its event fingerprints while retaining the original complete-link dictionary and corruptions. Both simultaneous-leak studies freshly simulate their events. The resolution command executes `expA1a_fix.py`'s complete-linkage and effective-rank calculations on the fixed sensitivity matrix; the earlier connected-components definition is retained only as a source reference.

Full commands compare every original result field to its archived counterpart before applying public coordinate-unit aliases. A mismatch or computation error saves a failed execution report and raises an exception. Successful reports state the hydraulic simulation count and exact numeric comparison results. Before averaging, each hydraulic result must have WNTR success status `None` and a complete, finite node-by-report-time pressure grid. `simulation_trace.jsonl.gz` records the actual solver status, complete node/time identifiers, grid shape, mean pressures and the unaveraged grid's SHA-256 for every new hydraulic simulation. Each command uses a per-run working directory and a process-lifetime native temporary path synchronized with Windows CRTs, restores the caller's temporary environment on exit, binds source/input/archive hashes and refuses to overwrite an existing result or report. `--smoke` explicitly reduces workloads and marks the result `completed_full_study=false`; it never asserts archive parity.

Saved traces can be independently audited under the final solver policy with `python -m experiments.historical.audit_exa7_traces --input-dir OUTPUT_PREFIX --output OUTPUT_PREFIX/final_policy_trace_audit.json`. This preserves original execution reports and source bindings, verifies actual success statuses and complete grid metadata, and binds the audit to the final guard/helper sources. Full-grid hashes are checked for presence and format; raw unaveraged matrices are not serialized for later hash recomputation.

Coordinate distances and class spans remain in native network-coordinate units. Public fields map explicitly to legacy `med_m`, `class_span_m` and `dist_median` archive names. The original sparse-pursuit coefficient diagnostic is retained as reported; no new discharge calibration is introduced.

`dependencies.json` identifies original scripts, their SHA-256 hashes, direct input dependencies and the supported portable status for Tables S7–S12, the resolution analysis and the unseen-node study. `exa7_sources/*.py` contains the executable EXA7 computations with direct frozen inputs and concise descriptions. `original_sources/*.py.txt` preserves the broader computational references; its introductory process narratives are replaced by concise descriptions. Those reference texts retain original file locations and imports and are not portable entry points.

LLM comparison statistics are reconstructed from archived query outcomes and model-call counts. Municipal-network source tables use the distributed numerical records; dependency records identify their hydraulic, partitioning and sensor-placement inputs, with raw-model access through the paper's Data availability contact. The EXA7 allocation tensor and its hydraulic screen are rebuilt through `experiments.allocation`.
