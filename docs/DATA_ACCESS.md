# Data and model inputs

| Asset | Location and provenance |
|---|---|
| EXA7 hydraulic model | `data/Exa7.inp`, the reference model already distributed by this project; SHA-256 `d2ed48e6722b65120dca6f391a655477d55ccadb277a40fc65b76c22e5cc72c1` |
| Reference physical design | `data/reference/partition_summary.json`, `sensor_summary.json` and the ordered sensor schema in `sensor_fingerprints.json` |
| Reference scenario library | `data/reference/sensor_fingerprints.json`: 540 scenarios, 30 sensors, 15 partitions; scenario manifest in `reproducibility/queries/reference540.csv` |
| Development/test and design splits | `reproducibility/splits/` and the study-specific frozen inputs |
| Query features and outcomes | `data/frozen/`, grouped by experiment; original query IDs and seeds are preserved |
| Source workbook and figures | `source_data/`: typed worksheet data, rectangular CSV exports, manuscript-table snapshots, captions, approved artwork and plotting inputs |
| MiniLM text encoder | `sentence-transformers/all-MiniLM-L6-v2`, revision `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`; the strict evaluator verifies the eleven checkpoint-file hashes stored in `source_manifest.json` |
| City D/Haining hydraulic models | Author-supplied municipal inputs, obtained through the paper's Data availability contact; archived numerical outputs and approved figure artwork are included in this release |

The municipal-map renderer accepts `--municipal-data-dir` containing the documented model/partition inputs; its default copies the approved vector artwork. The distributed EXA7 workflow executes from the included raw model. Model downloads retain the model provider's license and fixed revision; the repository MIT license covers its software.

`reproducibility/assets.json` records distributed file hashes. `source_data/workbook_manifest.json` and `source_data/figure_input_manifest.json` retain source-specific provenance. Historical metadata may contain the original execution paths; portable commands resolve inputs from the release and explicit arguments.
