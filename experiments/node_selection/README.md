# Node-disjoint development selection

Replay the archived frozen configurations on their test rows:

```text
python -m experiments.node_selection.run --stage replay --output-dir outputs/node_selection
```

Rebuild the complete experiment chain, including both development searches:

```text
python -m experiments.node_selection.run --stage all --output-dir outputs/node_selection_full
```

Stages are also selectable as `features`, `select`, `masked`, and `masked-select`. They rebuild the eight channels, repeat the original 400-draw development search, save its selected configuration, and evaluate the frozen choice on test rows. The masked stages apply the same observation-derived sensor mask to each query and every candidate template; they repeat the original four noise-floor thresholds and 400 masked-fusion draws. Candidate scoring uses development arrays only. Test arrays enter evaluation after the configuration is saved.

The eight-channel tensor contains the strict engine's six channels, a partition-wise projection-residual channel, and maximum squared cosine per partition. The squared-cosine feature is distinct from the strict engine's signed-cosine baseline. Masked cosine candidates clamp negative cosine to zero before squaring, and projection candidates normalize residuals by the original observation norm; these definitions preserve the executed experiment.

Each partition contributes its lexicographically smaller leak node to development and its other node to test: 15 development and 15 test nodes, with 1,350 observation rows in each set. Protocols H0–H8 are evaluated; H0 retains the entire corpus and therefore measures the complete-corpus baseline within each query split. H9 remains the separate strict engine's structural check.

`--stage replay --limit 2` provides a balanced smoke evaluation with two scenario cases from each split per noise seed. The complete 2,700-row feature and masked-score schemas are validated before selecting that subset, and replay verifies the full cached test results against both archived result tables. `--stage all --limit 2 --draws 4` limits development selection and test evaluation to that subset and four draws; feature and mask rebuilding still produces complete canonical caches. The complete search uses 400 draws. Explicit `--data-dir`, `--frozen-dir`, `--selection-frozen-dir`, and `--output-dir` override the bundled paths. Frozen replay saves its configuration copies and freshly calculated test results, and labels its provenance as archived test replay. Full execution saves rebuilt features and masked scores, both development searches, selected configurations, test results, source hashes and the stage-1 query-ranking comparison.
