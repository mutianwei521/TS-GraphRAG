# Evaluation contracts

The reference input `data/reference/sensor_fingerprints.json` fixes the order of 30 sensor nodes and 540 scenarios. EXA7 contains 381 junctions, 469 pipes, three pumps, one reservoir and one tank. The partition includes all 383 network nodes. Reference demand multipliers are 0.7, 1.0 and 1.2; leak magnitudes are 2, 5, 10, 20, 35 and 50 L/s.

## Retained-corpus protocols

`src/evaluation/protocols.py` defines H0–H9 for numerical API evaluation. `experiments/strict_engine/engine.py` implements the original frozen strict-comparison computations and checks the same deletion rules. H0 keeps all records. H1 removes the exact node/rate/period tuple; H2/H3/H4 remove the matching node/rate/period; H5/H6/H7/H8 remove the respective unions. H9 removes the true partition's records and checks the represented-label candidate contract.

The strict evaluator constructs each pool from retained records before computing centroids, period groups, semantic matrices, deterministic community text and the induced candidate graph. The provenance identifies every source scenario for each artifact. Sensor-prior features use all observed sensors and are restricted to represented labels before final per-channel normalization. A separate open-topology diagnostic exposes all physical labels while retaining the filtered scenario artifacts.

The query text constructor receives pressure observations and the demand period. The fixed MiniLM checkpoint encodes scenario documents independently; retained summaries are reconstructed from retained records. `--embedding-mode replay` uses released document/query/retained-summary features; `--embedding-mode rebuild` re-encodes their text with the hash-validated checkpoint.

## Numerical ranking and metrics

Projection residual profiles an unrestricted scalar amplitude for each candidate fingerprint. Zero-norm fingerprints receive infinite residual and are removed before ranking. The strict cosine baseline removes templates with norm at most 1e-12. Eligible scenario rankings use the frozen NumPy ordering; partition ranks retain each label's first occurrence. The historical physics study preserves its original zero-template conventions, as recorded in its module README.

Top-1 and Top-3 use the first one and three distinct labels. MRR uses the complete eligible partition ranking in the strict comparison. The allocation study preserves MRR@5. All reference queries, including 18 zero-response cases, remain in the strict denominator. The sensor-design estimator uses a nonnegative amplitude and records abstention; the hydraulic-mismatch study uses the original unrestricted amplitude.

## Repetition units

| Study | Fixed design and repetition |
|---|---|
| Repeated E4 benchmark | Seeds 41, 581, 1121, 1661, 2201; observation noise `RandomState(seed + original_query_index)`; fixed main90/all540 query manifests |
| Strict fusion/sensitivity | Seeds 41, 141, 241, 341, 441; `default_rng(seed)` over the 540×30 query field; five measurement-noise realizations |
| E5 allocation | 15 corpus/query-location draws, actual seeds 41–55; one shared 30-sensor noise vector per draw |
| Hydraulic perturbations | Five perturbation/noise realizations, seeds 27101–27105; 60 queries at 30 node-excluded locations per condition |
| Partition/sensor controls | Five paired train/test, design and noise runs, seeds 101, 202, 303, 404, 505; 303 training nodes and 78 test nodes; 156 queries per configuration |
| Estimator selection | Lexicographic one-node-per-partition development/test grouping; 15 nodes and 1,350 observations per split; test evaluation uses the frozen development choice |

The strict sensitivity grid contains 34 settings across H0/H2/H4/H8, producing 680 seed-level cells. Sample standard deviation uses `ddof=1`. Each study's repeat unit is preserved when rebuilding uncertainty and paired statistics.

The development/test grouping controls estimator selection. Protocol-specific deletion determines knowledge-library eligibility: H0 retains the full library; H2/H5/H6/H8 remove the query node's templates. The frozen selection is projection-only, beta=0. Full fusion and cosine remain explicit comparison methods.

## Integrity checks

The independent strict audit validates the complete seed/query/protocol/method grid, binds each prediction to its recomputed retained pool, recomputes metrics and summaries, checks excluded-record poisoning and compares expected archived records. Empty, missing, duplicate or misbound outputs fail validation. Run it with standard Python execution so validation remains active:

```bash
python -m experiments.strict_engine.verify --output-dir outputs/strict_engine --compare-frozen
```

Input hashes bind the reference model, query/split manifests, model checkpoint, ordered sensor schema and intermediate arrays to the frozen version. `scripts/verify_outputs.py` checks the raw EXA7 network as well as distributed asset bytes and recorded run outputs.
