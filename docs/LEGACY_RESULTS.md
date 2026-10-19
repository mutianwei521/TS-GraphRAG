# Indexed historical experiment records

Historical experiments retain their original query design, noise stream, candidate support and statistic definitions. Their records accompany the new strict comparison and frozen estimator-selection study.

| Evidence | Reproduction path |
|---|---|
| Physical-dictionary estimators, Tables S5–S6 | `python -m experiments.historical.run --profile full --output-dir outputs/physics`; all eight estimators recomputed from the reference fingerprint library |
| Historical LLM re-ranking, Table S7 and Fig. 6d | Population-weighted recovery/breakage statistics recomputed from archived counts by the historical replay command; original cloud requests are represented by their saved study outputs |
| EXA7 pressure-dependent leakage, sensor degradation and simultaneous leakage, Tables S8–S10 | `experiments.historical.exa7`: fresh hydraulic reconstruction with the original algorithms, seeds and complete result comparisons |
| EXA7 resolution analysis | `experiments.historical.exa7 --study resolution`: corrected complete-linkage and effective-rank calculations from the released sensitivity matrix |
| Further networks, Tables S11–S12 / Fig. 7 | Indexed original sources and dependencies in `experiments/historical/dependencies.json`; workbook and figure programs read the frozen numerical records |
| Old/full-corpus attribution diagnostic | `python -m experiments.strict_engine.run --study holdout --legacy-attribution --output-dir outputs/strict`; paired information-path attribution uses the original reference realization |

The historical source files in `experiments/historical/original_sources/` provide readable computational records. Their dependency index identifies model files, response caches and original experiment outputs. Each table's map entry distinguishes archived-statistic reconstruction from a portable full experiment.

The current paper's main numerical estimator is the frozen projection-residual mode. The full-fusion implementation supplies the retained-corpus component comparison and 34-setting sensitivity analysis. Development-selected projection and saved cosine predictions agree query by query on the released test set; paired masking results and exact McNemar statistics are distributed alongside the selected configuration.
