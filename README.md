# TS-GraphRAG

**Hydraulic scenario retrieval for leak localization in water distribution networks.**

TS-GraphRAG connects hydraulic structure with sensor-space retrieval. Offline simulations form a structured library of pressure responses; online observations are matched directly to that library to rank candidate leak regions. Its projection-residual inference profiles scenario amplitude without gradient-based model training or a new hydraulic simulation for each observation.

The release connects the complete numerical workflow: **EXA7 model → frozen physical design → hydraulic fingerprints → retained-corpus evaluation → query-level predictions → paper tables and figures**.

- **Physics-grounded representation:** 15 hydraulic partitions, 30 ordered sensors and 540 reference scenarios covering six leak rates and three demand states.
- **Direct numerical inference:** an explicit projection-residual rule, with cosine retrieval, physical-dictionary estimators and full fusion available in the comparison suite.
- **Traceable experiments:** fixed query identities, node splits, noise streams, development-selected configurations, retained-pool provenance and expected outputs.
- **Executable paper assets:** a 36-sheet Source Data workbook, figure sources and a panel-by-panel experiment map.

## Quick start

The validated environment uses **Python 3.11.15**, WNTR 1.4.0 / EPANET 2.2 and NumPy 2.3.5. The core installation runs numerical inference, frozen-embedding evaluation, hydraulic simulations and figure generation on the CPU.

```bash
git clone https://github.com/mutianwei521/TS-GraphRAG.git
cd TS-GraphRAG
python -m venv .venv
```

Activate `.venv` using `.venv\Scripts\Activate.ps1` on Windows or `source .venv/bin/activate` on Linux/macOS, then install the fixed environment:

```bash
python -m pip install -r requirements-core.lock.txt
python main.py --mode inference --observations data/example_observation.csv
python scripts/reproduce.py --profile smoke --output-dir outputs/smoke
python scripts/verify_outputs.py --run-dir outputs/smoke
```

The example contains pressure changes in metres, in the released sensor order. Inference returns the complete partition ranking and scenario matches. Each reproduction run records its effective inputs, execution commands, outputs and elapsed time.

For the exact Python 3.11.15 interpreter and pinned packages, use `conda env create -f environment.yml`, then `conda activate ts-graphrag-repro`. Automated Windows checks use the official CPython 3.11.9 Windows build with the same package lock; the execution reports identify the interpreter used for each validation.

## Reproduce the study

| Workflow | Starting inputs | Command |
|---|---|---|
| Input and execution checks | Included EXA7 model, fingerprint library and frozen feature assets | `python scripts/reproduce.py --profile smoke --output-dir outputs/smoke` |
| Numerical/semantic holdout | Fingerprints, fixed document vectors, retained-community and query embeddings | `python scripts/reproduce.py --profile replay --study strict_holdout --output-dir outputs/holdout` |
| Frozen estimator test | Saved strict-evaluation features and development-selected configurations | `python scripts/reproduce.py --profile replay --study node_selection --output-dir outputs/selection` |
| Raw hydraulic reconstruction | EXA7 model, frozen scenario IDs, sensor order and partition map | `python scripts/reproduce.py --profile full --study exa7 --output-dir outputs/exa7` |
| All source-data and figure workflows | Included experiment records and approved design assets | `python scripts/reproduce.py --profile replay --study all --output-dir outputs/paper_replay` |

`smoke` exercises a small, fixed suite. `replay` executes each study from its documented frozen inputs and recomputes its predictions or statistics. `full` rebuilds the selected study from its declared construction inputs: hydraulic models, training dictionaries, development-search inputs or the fixed semantic checkpoint. [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) lists the individual commands and output contracts.

Rebuild the source workbook and draw individual figures:

```bash
python scripts/build_source_data.py --output outputs/source_data
python scripts/figures/replay.py all --output outputs/figures
python scripts/figures/replay.py S6-standalone --output outputs/figures_s6
```

The [figure/table map](reproducibility/figure_table_map.csv) connects every panel, supplementary table and component comparison to its source data and execution path. Current manuscript/SI artwork and the standalone 17 October Fig. S6 have separate entries and matching captions.

## Evaluation design

Each protocol specifies the admissible reference scenarios. Numerical groups, centroids, semantic retrieval matrices, community summaries and candidate graphs are built from those retained records. Fixed physical topology supplies network structure.

| Protocol | Reference-library operation |
|---|---|
| H0 | Retain the complete reference library: noisy closed-world retrieval |
| H1 | Remove the query's exact node–rate–period scenario |
| H2 | Remove every scenario at the query leak node |
| H3 | Remove every scenario at the query leak rate |
| H4 | Remove every scenario in the query demand period |
| H5 / H6 / H7 | Remove the unions of node+rate / node+period / rate+period matches |
| H8 | Remove the union of node, rate and period matches |
| H9 | Remove every scenario in the true partition: candidate-support integrity control |

H2, H5, H6 and H8 evaluate node-excluded retrieval. H9 checks the represented-label support contract. The full protocol definitions, zero-response rules and repeat units are in [EVALUATION.md](docs/EVALUATION.md).

## Reference results

The strict comparison uses **five measurement-noise streams, each applied to the same 540 scenarios**; sensor noise has standard deviation 0.02 m. The projection-residual results are:

| Protocol | Top-1 (%) | Top-3 (%) | MRR |
|---|---:|---:|---:|
| H0: full reference library | 88.00 | 94.41 | 0.9183 |
| H2: leak node excluded | 76.56 | 87.52 | 0.8297 |
| H4: demand period excluded | 73.15 | 93.48 | 0.8354 |
| H8: node, rate and period exclusion | 59.67 | 85.11 | 0.7307 |

The separate estimator-selection study uses **15 development nodes and 15 disjoint test nodes**, with one node per partition in each split. Each split contains 270 scenarios evaluated under five noise streams, yielding 1,350 observations. Development selection freezes projection-residual inference with weight 1 and propagation strength 0. Table S17 supplies all H0–H8 test results and paired comparisons with cosine retrieval and full fusion. Its H2 test Top-1 accuracy is **81.11%**.

The earlier five-run classifier benchmark, strict fusion study, allocation draws and estimator-selection study retain their own query sets and random-number streams. Their configurations and source values are released together.

## Study entry points

| Study | Module | Paper assets |
|---|---|---|
| Repeated classical/GNN comparison | `experiments.repeated_benchmarks` | Fig. 4, Fig. S2, Table S2 |
| Fixed-budget scenario allocation | `experiments.allocation` | Fig. 5, Fig. S1, Table S3 |
| Retained-corpus fusion and attribution | `experiments.strict_engine` | Fig. 6b, Table S4, Note 1.9 |
| Physical-dictionary comparisons | `experiments.historical` | Fig. 6a/c, Tables S5–S6 |
| Emitter, degradation and simultaneous leaks | `experiments.historical.exa7` | Tables S8–S10 and resolution analysis |
| Hydraulic/demand perturbations | `experiments.hydraulic_robustness` | Fig. S3, Table S13 |
| Partition and sensor controls | `experiments.partition_sensor` | Figs. S4–S5, Tables S14–S15 |
| Frozen 34-setting sensitivity grid | `experiments.strict_engine` | Fig. S6, Table S16 |
| Development selection and frozen test | `experiments.node_selection` | Note 1.13, Table S17 |

Additional historical query outcomes, LLM statistics and network comparisons have indexed sources in [LEGACY_RESULTS.md](docs/LEGACY_RESULTS.md). Model downloads and GNN training use the separately pinned semantic/GNN environments. [DATA_ACCESS.md](docs/DATA_ACCESS.md) identifies the distributed inputs and municipal-model access paths.

## Release and citation

Use the fixed [resubmission release](https://github.com/mutianwei521/TS-GraphRAG/releases/tag/v1.0.1-resubmission) when reproducing the revised paper. The release includes checksums, source data and execution records. Machine-readable software citation is provided in [CITATION.cff](CITATION.cff).

Code is distributed under the [MIT License](LICENSE). Model and data provenance are recorded with their corresponding assets.
