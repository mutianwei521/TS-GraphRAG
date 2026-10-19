# Reproduction guide

## Fixed version and environment

Use tag `v1.0.0-resubmission` and Python 3.11.15. `requirements-core.lock.txt` is the complete clean-installed Windows x64 numerical/hydraulic/plotting environment. `environment.yml` defines its conda equivalent. The validation records identify the interpreter, package versions, inputs, command and output hashes for each executed study.

Each optional lock file is a complete profile. Install it in a separate Python 3.11.15 environment:

```bash
python -m pip install -r requirements-semantic.lock.txt
# Or, for original GNN training:
python -m pip install -r requirements-gnn.lock.txt
```

The semantic/GNN profiles preserve the original executed package versions, including PyTorch 2.9.1+cu128. Text encoding explicitly uses the CPU; the GNN runner records its selected device. The legacy profile includes the original vector-store and optional LLM integration dependencies. Numerical inference and frozen-embedding replay use the core profile.

## Input verification and a first inference

```bash
python scripts/verify_outputs.py --assets-only
python main.py --mode inference --observations data/example_observation.csv
python main.py --mode evaluate --protocol H2 --output-dir outputs/numerical_H2
```

`config.yaml` controls the numerical API. Paths resolve relative to that configuration file; explicit CLI paths override their corresponding inputs. Each experiment module instead reads its named frozen design, listed below. This separation makes exploratory numerical settings and paper-study configurations explicit.

Observations use an ordered `sensor_id,value` CSV or a single wide row with exactly the released sensor columns. Values are leak-minus-baseline pressure changes in metres. The returned ranking contains represented partition labels; the complete ordered scenario matches are also included.

## Three execution profiles

```bash
python scripts/reproduce.py --profile smoke --output-dir outputs/smoke
python scripts/reproduce.py --profile replay --study all --output-dir outputs/paper_replay
python scripts/verify_outputs.py --run-dir outputs/paper_replay
```

Every recorded run uses a fresh output directory. `smoke` runs a small real hydraulic reconstruction, offline inference, strict retention checks, frozen estimator evaluation and source-workbook reconstruction. `replay` starts each study at its released numerical/semantic features or query-outcome records. `full` executes the selected study's construction stages. Its precise starting inputs appear in the table below and each experiment README.

| Study | Replay starting point | Full construction |
|---|---|---|
| `exa7` | Frozen fingerprint library and explicit query manifest | Simulate all 540 scenarios from the EXA7 INP using the fixed sensor/partition/scenario design, then evaluate regenerated fingerprints |
| `strict_holdout`, `sensitivity` | Fingerprints and released scenario/query/retained-community embeddings | Rebuild reference hydraulics and encode fixed-checkpoint text; construct retained pools and evaluate |
| `node_selection` | Fixed feature/masked-score arrays and frozen configurations | Recompute channel features, repeat the two 400-draw development searches, freeze selections and evaluate test rows |
| `repeated_benchmarks` | Archived per-query classical/GNN predictions; fresh projection inference | Refit the classical methods; `--include-gnn` executes the original GNN architectures/training for main90 |
| `allocation` | Per-draw archived outcomes and tensor-derived design identities | Rebuild the 350×12×3×30 hydraulic tensor and execute 270,000 retrieval queries |
| `hydraulic_robustness` | 140,400 archived query outcomes and seed-level statistics | Nominal corpus, independent corpus replay, 106 physical perturbation batches and all scoring/endpoint audits |
| `partition_sensor` | Cached primary pressure-response matrices; design/localization recomputed | Full design feasibility screen and primary response simulations, then paired train-only designs and evaluation |
| `historical` | Saved physical/LLM/emitter query outcomes and counts | Eight physical-dictionary estimators; fresh EXA7 emitter, sensor-degradation and two simultaneous-leak studies; resolution calculations from the fixed sensitivity matrix |
| `source_data`, `figures` | Audited numerical cells and design assets | Source-data/figure reconstruction from the same explicitly mapped study records |

## Raw EXA7 and physical design

```bash
python wds_partition_leiden_main.py --output-dir outputs/frozen_partition
python build_knowledge_base.py --output-dir outputs/rebuilt_reference
python scripts/reproduce.py --profile full --study exa7 --output-dir outputs/exa7_raw
```

The first command exports the released physical partition. `--design leiden --seed 41` generates a separate deterministic design. The sensor optimizer accepts `--num-partitions`/`--partition-k`, an explicit partition file and an output directory; the paper's reproduced predictions use the released ordered 30-sensor configuration. Sensor/partition design comparisons have their own paired training/test protocol.

## Strict fusion and sensitivity

```bash
python -m experiments.strict_engine.run --study all --legacy-attribution --output-dir outputs/strict
python -m experiments.strict_engine.verify --output-dir outputs/strict --compare-frozen
```

This generates primary predictions, all 34 sensitivity settings, retained-source provenance, independent metric reconstruction and paired attribution. The complete primary/sensitivity/attribution grids contain 83,700 / 367,200 / 54,000 records respectively. For text-feature reconstruction, install the semantic profile and use the fixed checkpoint:

```bash
python scripts/reproduce.py --profile full --study strict_holdout --allow-download --output-dir outputs/strict_raw
```

Alternatively pass `--model PATH_TO_FIXED_CHECKPOINT`. The builder verifies revision `c9745ed1d9f207416be6d2e6f8de32d1f16199bf` through the archived checkpoint-file hashes before encoding. Each query's retained summaries have their own reconstructed source IDs and text.

## Frozen estimator selection

```bash
python -m experiments.node_selection.run --stage replay --output-dir outputs/selection_test
python -m experiments.node_selection.run --stage all --output-dir outputs/selection_rebuild
```

The first command evaluates already-frozen configurations. The second recomputes features and the original development searches before the test evaluation. The node groups and all feature axes remain fixed. `--limit` selects a small evaluation subset for checks; the complete input grid is validated before subsetting.

## Repeated baselines and allocation

```bash
python -m experiments.repeated_benchmarks.run --profile full --scope main90 --workers 4 --output-dir outputs/e4_main90
python -m experiments.repeated_benchmarks.run --profile full --scope all540 --workers 4 --output-dir outputs/e4_all540
# Optional original GNN training, in the GNN environment:
python -m experiments.repeated_benchmarks.run --profile full --scope main90 --include-gnn --workers 4 --output-dir outputs/e4_gnn
python scripts/reproduce.py --profile full --study allocation --output-dir outputs/e5_raw
```

Complete classical refitting uses 650 main90 or 7,115 all540 seed/pool tasks. Cached fit results are bound to source inputs and parameters. `--max-pools 1` is an explicitly labelled fit smoke test. The allocation builder performs 381 screen solves, three baselines and 12,600 leak solves. Its retrieval preserves the original shared-noise-per-draw rule and reports network-coordinate distances in native coordinate units.

## Perturbations and infrastructure controls

```bash
python -m experiments.hydraulic_robustness.run --output-dir outputs/hydraulic_raw
python -m experiments.partition_sensor.run --profile full --output-dir outputs/design_raw
```

EPANET workers use separate working and temporary directories. The hydraulic study independently reproduces all 1,053 nominal templates, with 1,041 rank-eligible atoms across the three states (351/350/340). Every query batch reproduces its baseline before/after the batch. All planned queries remain in the primary denominator; hydraulic service diagnostics are separately tabulated.

The design study rebuilds 1,143 primary hydraulic scenarios and a 1,144-solve design-feasibility screen. Training/test IDs and sensor lists are exported with the 44,460 query/configuration outcomes. Candidate extent uses physical pipe lengths in metres. E5's geometric distance uses the network's native coordinates, as recorded in its output schema.

## Emitter, degradation and simultaneous-leak studies

```bash
python scripts/reproduce.py --profile full --study historical --output-dir outputs/historical_full
```

This recomputes the eight physical-dictionary comparisons and runs five EXA7 components: pressure-dependent emitters, sensor degradation, resolution, exhaustive simultaneous-leak estimation and sparse simultaneous-leak estimation. Each component also has a separate `python -m experiments.historical.exa7 --study NAME --output-dir PATH` entry point. The resolution study starts at the distributed sensitivity matrix; the other four generate fresh hydraulic event responses with the original query rules and seeds. Every component compares its complete original result structure with the frozen archive. [The historical study guide](../experiments/historical/README.md) defines inputs, outputs and coordinate units. Saved LLM outcomes and municipal-network statistics retain their indexed reconstruction paths.

## Tables, figures and execution evidence

```bash
python scripts/build_source_data.py --output outputs/source_data
python scripts/figures/replay.py all --output outputs/figures
```

`reproducibility/figure_table_map.csv` indexes every figure version, panel and supplementary table. Worksheet JSON retains native cell values and blank rows; CSV exports support independent inspection. Table S17 is recalculated from both test-result JSON files before workbook construction. The workbook has 36 sheets.

`run_report.json` records execution status, steps, elapsed times and hashes of generated outputs. The published validation summary records the actual verification level for each study. Hardware-sensitive training records retain their original environment/device metadata; deterministic query/ranking comparisons use exact identities, and aggregate floating comparisons use a fixed absolute tolerance of 1e-8.
