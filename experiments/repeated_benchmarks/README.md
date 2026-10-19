# Repeated benchmarks (E4 / Table S3)

The reference corpus has 540 hydraulic fingerprints and 30 ordered pressure channels. The matched query panel uses the original checkpoint's 90 fixed indices (six per partition); the complete panel uses all 540 indices. Repetition seeds are 41, 581, 1121, 1661 and 2201. Query noise is exactly `RandomState(run_seed + original_query_index).normal(0, 0.02, 30)`. Sample SD uses `ddof=1` over complete repetitions.

Run from the repository root:

```text
python -m experiments.repeated_benchmarks.run --profile replay --scope main90 --output-dir outputs/e4/main90
python -m experiments.repeated_benchmarks.run --profile replay --scope all540 --output-dir outputs/e4/all540
python -m experiments.repeated_benchmarks.run --profile full --scope main90 --workers 4 --output-dir outputs/e4/full_main90
python -m experiments.repeated_benchmarks.run --profile full --scope all540 --workers 4 --output-dir outputs/e4/full_all540
```

Replay recalculates correctness from archived predicted and true labels, then recomputes Top-1 means and sample SD. It independently regenerates every projection observation and prediction, comparing labels against the archive. The matched panel also recalculates all three archived GNN summaries. Replay needs no PyTorch, embeddings, API key or network service.

Full refits the six supervised classical models, projection and cosine retrieval. Identical retained pools are fitted once per seed, retaining their original index order. Output includes per-query labels, run statistics, exact model configurations, source hashes, fitting warnings/errors, and MLP internal training/validation splits. A fitting error leaves its affected rates null and raises an exception. Reusable pool fits are bound to input hashes, implementation, configurations and numerical package versions.

The complete matched panel requires 650 seed/pool fits; all540 requires 7,115. To check fitting and the environment quickly, use `--profile full --scope main90 --max-pools 1 --workers 1` with a fresh output directory. Limited runs are explicitly marked `profile=smoke`, `completed=false` and use separate `_smoke` output filenames.

`classical.py` ports `results/E4_mean_std_20260909/classical_repeats.py`; `primitives.py` retains the `wds_verify.py` projection residual, including exclusion of zero-norm atoms. `architecture.py` carries the original `expH_gnn2.py` build function. `gnn.py` carries the original repeated-GNN fitting schedule: 200 epochs, three graph layers, sensor embeddings, raw fingerprint plus mean/max readout, Adam, StepLR, noise augmentation and retained-pool validation.

Optional exact GNN training requires the original CUDA environment: Python 3.11, torch 2.9.1+cu128 and torch-geometric 2.7.0 (plus compatible CUDA hardware). Each new fit records its actual numerical environment. Invoke the matched classical/GNN run with `--include-gnn`, or run `python -m experiments.repeated_benchmarks.gnn run --output-dir outputs/e4/full_gnn`. GNN fitting can involve 1,950 fits; hardware and CUDA kernels may affect numerical results. `python -m experiments.repeated_benchmarks.gnn verify --output-dir outputs/e4/gnn_inputs` verifies all retained pools, 198 directed graph edges and the seed-41 projection anchor without loading PyTorch.
