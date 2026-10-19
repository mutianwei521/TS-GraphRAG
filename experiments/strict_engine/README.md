# Strict retained-corpus evaluation

Run from the repository root:

```text
python -m experiments.strict_engine.run --study all --legacy-attribution --output-dir outputs/strict_engine
python -m experiments.strict_engine.verify --output-dir outputs/strict_engine --compare-frozen
```

The default replay uses NumPy, frozen per-document scenario vectors, exact-SHA retained-community embeddings, and frozen observation-only query embeddings. It opens no vector database and makes no model or LLM request. Explicit `--data-dir`, `--frozen-dir`, and `--output-dir` paths override the bundled defaults. `--limit 2` runs two queries for each of the five noise seeds, while retaining the complete reference corpus before each protocol filters it. `--study holdout` and `--study sensitivity` run either study independently.

The ten protocols preserve the original definitions: H0 retains every scenario; H1 excludes the exact node/rate/period combination; H2, H3 and H4 exclude the query node, rate and period, respectively; H5–H8 combine those exclusions; H9 excludes the complete truth partition. Every retained pool independently supplies templates, centroids, scenario vectors and deterministic community summaries. The scoring API receives observations and observed period. Static network topology and sensor assignments supply deployment context.

Projection sorts eligible retained templates by the residual after fitting an unrestricted scalar multiplier. Cosine sorts by signed cosine similarity. Each converts its template ordering to a partition ranking by the first occurrence of each partition. Zero-norm eligibility follows the original thresholds (`>0` for projection, `>1e-12` for cosine). Six-channel fusion combines sensor prior, centroids, rank matches, fingerprint matches, semantic retrieval and community similarity, then propagates scores over the induced retained-label graph. H9 zeros under this domain contract are a structural integrity control. The separate open-topology and optional unfiltered-semantic attribution outputs preserve their diagnostic definitions.

The five seeds are 41, 141, 241, 341 and 441. All 34 prespecified sensitivity configurations are emitted; evaluation preserves the default configuration. The complete ranked labels, metrics, source hashes, retained-ID provenance and integrity checks accompany aggregate CSVs. The independent verifier requires the prediction files and both aggregate tables for the requested study, rejects empty or incomplete grids, and binds every prediction to the retained pool reconstructed from its actual query and protocol. It recalculates metrics and aggregate values, checks retained-label support and numeric eligibility, and can compare every archived output field. Limited runs compare their selected queries against the corresponding archive records. The auditor rejects optimized Python (`-O` or `PYTHONOPTIMIZE`) because its validation assertions must remain enabled.

To rebuild scenario, community and query embeddings with the audited MiniLM checkpoint:

```text
python -m experiments.strict_engine.run --embedding-mode rebuild --model /path/to/local/checkpoint --study holdout --output-dir outputs/rebuilt
```

Omit `--model` to resolve the canonical checkpoint from the local Hugging Face cache; add `--allow-download` to download it explicitly. The model revision is `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`, and all checkpoint files must match the archived SHA-256 manifest. Rebuilt caches and their model manifest are saved beside outputs. Query text uses the original observation-only formatting without importing the legacy application.
