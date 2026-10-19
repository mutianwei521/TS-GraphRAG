"""Numerical evaluation with explicit ordered queries and retained-pool provenance."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from src.configuration import AppConfig
from src.phase3_query.numerical_retrieval import NumericalRetriever, load_fingerprints, validate_sensor_order
from .metrics import aggregate_metrics, rank_metrics
from .protocols import PROTOCOLS, retain, allowed_candidate_domain


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _csv_queries(path: Path, lookup: dict) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "scenario_id" not in reader.fieldnames:
            raise ValueError("CSV query manifest requires a scenario_id column")
        records = list(reader)
    queries = []
    indices = []
    for record in records:
        sid = record["scenario_id"]
        if sid not in lookup:
            raise ValueError(f"Query ID absent from corpus: {sid}")
        query = lookup[sid]
        if "query_index" in record:
            try:
                index = int(record["query_index"])
            except (ValueError, TypeError) as exc:
                raise ValueError("query_index must be a non-negative integer") from exc
            if index < 0:
                raise ValueError("query_index must be a non-negative integer")
            indices.append(index)
        for field in ("leak_node", "leak_partition", "leak_rate_Ls", "demand_period", "demand_multiplier"):
            if field not in record:
                continue
            if field not in query:
                raise ValueError(f"Query metadata field {field} absent from corpus for {sid}")
            if field in {"leak_partition", "leak_rate_Ls", "demand_multiplier"}:
                try:
                    equal = float(record[field]) == float(query[field])
                except (ValueError, TypeError):
                    equal = False
            else:
                equal = record[field] == str(query[field])
            if not equal:
                raise ValueError(f"Query metadata mismatch for {sid}: {field}")
        queries.append(query)
    if indices and (len(indices) != len(set(indices)) or indices != sorted(indices)):
        raise ValueError("query_index values must be unique and ascending in saved CSV row order")
    return queries


def load_queries(path: Path | None, fingerprints: list[dict], sensor_nodes: list[str]) -> list[dict]:
    """Read saved CSV/JSON IDs or observed query rows; never sample implicitly."""
    lookup = {row["scenario_id"]: row for row in fingerprints}
    if path is None:
        queries = list(fingerprints)
    elif Path(path).suffix.lower() == ".csv":
        queries = _csv_queries(Path(path), lookup)
    else:
        with Path(path).open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if isinstance(manifest, dict) and "query_ids" in manifest:
            ids = manifest["query_ids"]
            if not isinstance(ids, list) or not all(isinstance(sid, str) for sid in ids):
                raise ValueError("query_ids must be an ordered list of scenario IDs")
            missing = [sid for sid in ids if sid not in lookup]
            if missing:
                raise ValueError(f"Query IDs absent from corpus: {missing}")
            queries = [lookup[sid] for sid in ids]
        elif isinstance(manifest, dict) and "fingerprints" in manifest:
            schema, queries = load_fingerprints(path)
            validate_sensor_order(schema, sensor_nodes)
        elif isinstance(manifest, dict) and "queries" in manifest:
            if "sensor_nodes" not in manifest:
                raise ValueError("A query-row manifest must declare sensor_nodes")
            validate_sensor_order(manifest["sensor_nodes"], sensor_nodes)
            queries = manifest["queries"]
            NumericalRetriever(sensor_nodes, queries)
        else:
            raise ValueError("Query manifest requires query_ids, fingerprints, or queries with sensor_nodes")
    if not queries:
        raise ValueError("Query manifest must not be empty")
    if len({row["scenario_id"] for row in queries}) != len(queries):
        raise ValueError("Query manifest contains duplicate scenario IDs")
    return queries


def run_evaluation(config: AppConfig, protocol: str | None = None, seed: int | None = None,
                   method: str | None = None, output_dir: str | Path | None = None,
                   fingerprints_path: str | Path | None = None,
                   query_manifest_path: str | Path | None = None) -> dict:
    """Write all fixed queries, including zero observations and empty support.

    One default_rng(seed).normal(0, sigma, (N, S)) draw is consumed in saved
    manifest order. For the same ordered 540 queries this matches the strict
    engine's measurement-noise generation exactly. H9 is a negative control.
    """
    protocol = protocol if protocol is not None else config.raw["evaluation"]["protocol"]
    method = method if method is not None else config.method
    seed = seed if seed is not None else config.seed
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown protocol {protocol!r}")
    if method not in {"projection", "cosine"}:
        raise ValueError("Retained-corpus evaluation supports projection and cosine; legacy fusion is not substituted")
    if config.beta != 0:
        raise ValueError("Frozen numerical evaluation requires inference.beta = 0")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    corpus_path = Path(fingerprints_path).resolve() if fingerprints_path is not None else config.fingerprints_path
    query_path = Path(query_manifest_path).resolve() if query_manifest_path is not None else config.query_manifest_path
    sensor_nodes, fingerprints = load_fingerprints(corpus_path)
    queries = load_queries(query_path, fingerprints, sensor_nodes)
    noise = np.random.default_rng(seed).normal(0, config.noise_sigma, (len(queries), len(sensor_nodes)))
    output = Path(output_dir).resolve() if output_dir is not None else config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    records, pools = [], {}
    for qi, query in enumerate(queries):
        retained = retain(fingerprints, query, protocol)
        retained_ids = tuple(row["scenario_id"] for row in retained)
        if retained_ids not in pools:
            pools[retained_ids] = NumericalRetriever(sensor_nodes, retained)
        pool = pools[retained_ids]
        y = np.array(query["sensor_fingerprint"], dtype=float) + noise[qi]
        result = pool.rank(y, method)
        allowed = allowed_candidate_domain(retained)
        if not set(result["ranking"]) <= set(allowed):
            raise AssertionError("Prediction escaped retained candidate domain")
        metrics = rank_metrics(result["ranking"], int(query["leak_partition"]))
        if protocol == "H9" and any(metrics.values()):
            raise AssertionError("H9 returned a deleted true partition")
        records.append({
            "seed": seed, "query_index": qi, "query_id": query["scenario_id"],
            "truth": int(query["leak_partition"]), "protocol": protocol,
            **result, **metrics, "pool_id": _digest(list(retained_ids)),
            "retained_scenario_ids": list(retained_ids), "observations": y.tolist(),
        })
    metrics = aggregate_metrics(records)
    manifest = {
        "query_ids": [row["scenario_id"] for row in queries],
        "query_source": str(query_path) if query_path is not None else str(corpus_path),
        "source_kind": "explicit_manifest" if query_path is not None else "all_corpus_rows_in_saved_order",
        "sensor_nodes": sensor_nodes, "seed": seed, "noise_sigma": config.noise_sigma,
        "noise_generation": "numpy.random.default_rng(seed).normal(0, sigma, (n_queries, n_sensors))",
        "query_rows_sha256": _digest(queries),
    }
    effective = config.as_dict()
    effective["evaluation"].update(seed=seed, protocol=protocol)
    effective["inference"].update(method=method, beta=0)
    effective["paths"].update(fingerprints=str(corpus_path),
                              query_manifest=str(query_path) if query_path else None,
                              output_dir=str(output))
    summary = {
        "method": method, "protocol": protocol, "seed": seed,
        "noise_sigma": config.noise_sigma, "beta": 0,
        "metrics": metrics, "n_corpus": len(fingerprints), "n_retained_pools": len(pools),
        "corpus_path": str(corpus_path),
        "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        "query_rows_sha256": manifest["query_rows_sha256"], "sensor_nodes": sensor_nodes,
        "candidate_domain": "Labels represented by retained scenarios; zero support excluded before ranking",
        "rr_definition": "Reciprocal rank over the complete returned distinct-partition ranking; missing truth is zero",
        "denominator": "All queries, including zero observations and zero/empty eligible support",
        "H9_interpretation": "Structural integrity negative control; true label absent from retained corpus",
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "effective_config": effective, "output_dir": str(output),
    }
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    _write_json(output / "query_manifest.json", manifest)
    _write_json(output / "summary.json", summary)
    return summary
