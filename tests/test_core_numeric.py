"""Scientific contracts for the dependency-light numerical reference path."""
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def fp(sid, vector, label=0, node="n", rate=2, period="day_normal"):
    return dict(scenario_id=sid, sensor_fingerprint=vector, leak_partition=label,
                leak_node=node, leak_rate_Ls=rate, demand_period=period)


def corpus_file(path, rows, sensors=("s1", "s2")):
    path.write_text(json.dumps(dict(sensor_nodes=list(sensors), fingerprints=rows)),
                    encoding="utf-8")
    return path


def test_numerical_prepare_matches_frozen_projection_cosine_and_numpy_ties():
    from src.phase3_query.numerical_retrieval import NumericalRetriever
    rows = [fp(str(i), v, i % 4) for i, v in enumerate(
        [[1., 0.], [2., 0.], [0., 1.], [1., 1.], [3., 0.], [0., 0.],
         [1e-13, 0.], [-1., 0.], [0., 2.], [2., 2.], [4., 0.]])]
    pool = NumericalRetriever(["s1", "s2"], rows)
    y = np.array([.7, .2])
    matrix = np.array([r["sensor_fingerprint"] for r in rows])
    norms = np.linalg.norm(matrix, axis=1)
    residual = np.linalg.norm(y[None, :] -
        (matrix @ y / np.maximum(norms ** 2, 1e-18))[:, None] * matrix, axis=1)
    residual[norms == 0] = np.inf
    cosine = matrix @ y / np.maximum(norms * np.linalg.norm(y), 1e-18)
    cosine[norms <= 1e-12] = -np.inf
    p = np.flatnonzero(norms > 0)
    c = np.flatnonzero(norms > 1e-12)
    prepared = pool.prepare(y)
    np.testing.assert_array_equal(prepared["projection_order"], p[np.argsort(residual[p])])
    np.testing.assert_array_equal(prepared["cosine_order"], c[np.argsort(-cosine[c])])
    np.testing.assert_array_equal(prepared["projection_residuals"], residual)
    np.testing.assert_array_equal(prepared["cosine_scores"], cosine)


def test_grouped_label_ranking_is_complete_and_zeros_have_no_rank():
    from src.phase3_query.numerical_retrieval import NumericalRetriever
    rows = [fp("zero", [0, 0], 9), fp("best", [1, 0], 4),
            fp("same", [2, 0], 4), fp("other", [1, 1], 2),
            fp("last", [0, 1], 8)]
    result = NumericalRetriever(["s1", "s2"], rows).rank([1, .1])
    assert result["ranking"] == [4, 2, 8]
    assert result["allowed_candidate_domain"] == [2, 4, 8, 9]
    assert "zero" not in result["scenario_ranking"]
    assert len(result["scenario_ranking"]) == 4


def test_zero_query_is_retained_and_zero_only_or_empty_support_returns_no_ranks():
    from src.phase3_query.numerical_retrieval import NumericalRetriever
    rows = [fp("zero", [0, 0], 5)]
    for rows in (rows, []):
        pool = NumericalRetriever(["s1", "s2"], rows)
        for method in ("projection", "cosine"):
            assert pool.rank([0, 0], method)["ranking"] == []


def test_sensor_order_dimensions_and_finite_values_are_validated(tmp_path):
    from src.phase3_query.numerical_retrieval import NumericalRetriever, read_observations
    pool = NumericalRetriever(["s1", "s2"], [fp("a", [1, 0])])
    with pytest.raises(ValueError, match="sensor order"):
        pool.rank([1, 0], sensor_nodes=["s2", "s1"])
    for vector in ([1], [1, float("nan")], [1, float("inf")]):
        with pytest.raises(ValueError):
            pool.prepare(vector)
    for text in ("sensor_id,value\ns2,2\ns1,1\n", "s2,s1\n2,1\n",
                 "sensor_id,value\ns1,1\ns1,2\n", "s1,s2,s3\n1,2,3\n"):
        path = tmp_path / "bad.csv"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError):
            read_observations(path, ["s1", "s2"])
    path = tmp_path / "good.csv"
    path.write_text("sensor_id,value\ns1,1\ns2,2\n", encoding="utf-8")
    np.testing.assert_array_equal(read_observations(path, ["s1", "s2"]), [1, 2])
    path.write_text("s1,s2\n3,4\n", encoding="utf-8")
    np.testing.assert_array_equal(read_observations(path, ["s1", "s2"]), [3, 4])


def test_protocols_are_exact_unions_and_h9_removes_true_label():
    from src.evaluation.protocols import PROTOCOLS, keep, retain, allowed_candidate_domain
    q = fp("q", [1, 0], 0, "A", 2, "day_normal")
    rows = [fp(f"{n}{r}{t}", [1, 0], int(n),
               "A" if n == 0 else "B", 2 if r == 0 else 5,
               "day_normal" if t == 0 else "night_low")
            for n in (0, 1) for r in (0, 1) for t in (0, 1)]
    expected = {"H0": 8, "H1": 7, "H2": 4, "H3": 4, "H4": 4,
                "H5": 2, "H6": 2, "H7": 2, "H8": 1, "H9": 4}
    assert tuple(expected) == tuple(PROTOCOLS)
    assert {p: len(retain(rows, q, p)) for p in PROTOCOLS} == expected
    assert allowed_candidate_domain(retain(rows, q, "H9")) == [1]
    assert not keep(q, q, "H1")
    with pytest.raises(ValueError, match="protocol"):
        keep(q, q, "H10")


def test_metrics_use_full_label_rank_and_all_queries_denominator():
    from src.evaluation.metrics import aggregate_metrics, rank_metrics
    long_rank = list(range(15))
    assert rank_metrics(long_rank, 14) == dict(top1=0, top3=0, rr=1 / 15)
    metrics = aggregate_metrics([rank_metrics([1], 1), rank_metrics([], 0)])
    assert metrics == dict(n_queries=2, top1_accuracy=.5, top3_accuracy=.5, mrr=.5)


def test_config_values_and_config_relative_paths_are_effective(tmp_path):
    from src.configuration import load_config
    path = tmp_path / "custom.yaml"
    path.write_text("""network:
  inp_file: network.inp
paths:
  fingerprints: corpus.json
  query_manifest: queries.json
  output_dir: out
inference:
  method: cosine
  beta: 0
evaluation:
  seed: 17
  noise_sigma: 0.17
""", encoding="utf-8")
    cfg = load_config(path)
    assert cfg.fingerprints_path == tmp_path / "corpus.json"
    assert cfg.query_manifest_path == tmp_path / "queries.json"
    assert cfg.network_inp == tmp_path / "network.inp"
    assert cfg.output_dir == tmp_path / "out"
    assert cfg.method == "cosine" and cfg.seed == 17 and cfg.noise_sigma == .17
    assert cfg.beta == 0
    assert cfg.demand_multipliers == {"night_low": .7, "day_normal": 1., "evening_peak": 1.2}
    assert cfg.as_dict()["evaluation"]["noise_sigma"] == .17


def test_cli_numerical_inference_from_external_directory_without_optional_imports(tmp_path):
    corpus_file(tmp_path / "corpus.json", [fp("right", [1, 0], 3), fp("up", [0, 1], 6)])
    (tmp_path / "observations.csv").write_text("sensor_id,value\ns1,1\ns2,0.1\n", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("paths:\n  fingerprints: corpus.json\ninference:\n  method: cosine\n", encoding="utf-8")
    # sitecustomize makes accidental embedding/LLM imports a hard failure.
    (tmp_path / "sitecustomize.py").write_text(
        "import sys\nclass Block:\n def find_spec(self, name, *args):\n"
        "  if name.split('.')[0] in {'torch', 'sentence_transformers', 'lancedb', 'langchain_ollama'}: raise RuntimeError(name)\n"
        "sys.meta_path.insert(0, Block())\n", encoding="utf-8")
    import os
    env = dict(os.environ, PYTHONPATH=str(tmp_path))
    result = subprocess.run([sys.executable, str(ROOT / "main.py"), "--mode", "inference",
        "--config", str(tmp_path / "config.yaml"), "--observations", str(tmp_path / "observations.csv")],
        cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["method"] == "cosine"
    assert output["predicted_partition"] == 3
    assert output["ranking"] == [3, 6]


def test_evaluation_records_explicit_queries_retained_domain_noise_and_empty_support(tmp_path):
    from src.configuration import load_config
    from src.evaluation.runner import run_evaluation
    rows = [fp("a", [1, 0], 0, "a"), fp("b", [0, 1], 1, "b"), fp("zero", [0, 0], 2, "z")]
    corpus_file(tmp_path / "corpus.json", rows)
    manifest = dict(query_ids=["zero", "b", "a"])
    (tmp_path / "queries.json").write_text(json.dumps(manifest), encoding="utf-8")
    config = tmp_path / "cfg.yaml"
    config.write_text("paths:\n  fingerprints: corpus.json\n  query_manifest: queries.json\n  output_dir: out\nevaluation:\n  noise_sigma: 0\n", encoding="utf-8")
    summary = run_evaluation(load_config(config), protocol="H9", seed=23)
    assert summary["metrics"]["n_queries"] == 3
    assert summary["metrics"]["top1_accuracy"] == 0
    assert summary["metrics"]["mrr"] == 0
    records = [json.loads(line) for line in (tmp_path / "out" / "predictions.jsonl").read_text().splitlines()]
    assert [r["query_id"] for r in records] == manifest["query_ids"]
    assert all(r["truth"] not in r["allowed_candidate_domain"] for r in records)
    assert all(set(r["ranking"]) <= set(r["allowed_candidate_domain"]) for r in records)
    saved_manifest = json.loads((tmp_path / "out" / "query_manifest.json").read_text())
    assert saved_manifest["query_ids"] == manifest["query_ids"]
    assert saved_manifest["seed"] == 23 and saved_manifest["noise_sigma"] == 0
    assert summary["effective_config"]["evaluation"]["seed"] == 23
    assert summary["effective_config"]["evaluation"]["protocol"] == "H9"


def test_csv_query_manifest_preserves_saved_order_and_checks_metadata(tmp_path):
    from src.evaluation.runner import load_queries
    rows = [fp("a", [1, 0], 0, "a"), fp("b", [0, 1], 1, "b")]
    path = tmp_path / "queries.csv"
    path.write_text("query_index,scenario_id,leak_node,leak_partition,leak_rate_Ls,demand_period\n"
                    "0,b,b,1,2,day_normal\n1,a,a,0,2,day_normal\n", encoding="utf-8")
    assert [r["scenario_id"] for r in load_queries(path, rows, ["s1", "s2"])] == ["b", "a"]
    path.write_text(path.read_text().replace("0,b,b,1", "0,b,b,5"), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata"):
        load_queries(path, rows, ["s1", "s2"])


def test_config_rejects_invalid_sections_and_ineffective_nonzero_graph_setting(tmp_path):
    from src.configuration import load_config
    path = tmp_path / "bad.yaml"
    for text in ("inference: null\n", "paths: []\n", "inference:\n  beta: 0.15\n"):
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError):
            load_config(path)


def test_frozen_rebuild_rejects_unsupported_configuration(tmp_path):
    from src.configuration import load_config, validate_reference_configuration
    rows = [dict(fp("a", [1, 0], 0, "a"), demand_multiplier=1)]
    path = tmp_path / "cfg.yaml"
    path.write_text("knowledge_base:\n  leak_rates_Ls: [99]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen reference"):
        validate_reference_configuration(load_config(path), ["s1", "s2"], rows)


@pytest.mark.parametrize("field,value", [
    ("leak_node", None), ("leak_node", ""), ("leak_node", "  "), ("leak_node", 2),
    ("demand_period", None), ("demand_period", ""), ("demand_period", "\t"), ("demand_period", 2),
    ("leak_rate_Ls", "2"), ("leak_rate_Ls", None), ("leak_rate_Ls", True),
    ("leak_rate_Ls", float("nan")), ("leak_rate_Ls", float("inf")),
    ("leak_rate_Ls", 0), ("leak_rate_Ls", -2),
])
def test_illegal_scenario_metadata_is_rejected_for_corpus_and_explicit_queries(tmp_path, field, value):
    from src.phase3_query.numerical_retrieval import NumericalRetriever, load_fingerprints
    from src.evaluation.runner import load_queries
    good = fp("good", [1, 0])
    bad = dict(fp("bad", [1, 0]), **{field: value})
    with pytest.raises(ValueError, match=field):
        NumericalRetriever(["s1", "s2"], [bad])
    path = corpus_file(tmp_path / "bad.json", [bad])
    with pytest.raises(ValueError, match=field):
        load_fingerprints(path)
    path.write_text(json.dumps(dict(sensor_nodes=["s1", "s2"], queries=[bad])), encoding="utf-8")
    with pytest.raises(ValueError, match=field):
        load_queries(path, [good], ["s1", "s2"])


def test_h3_compares_numeric_magnitude_without_coercing_illegal_string_or_bool():
    from src.evaluation.protocols import keep, retain
    query = fp("query", [1, 0], rate=2)
    assert not keep(fp("float", [0, 1], rate=2.0), query, "H3")
    for bad_rate in ("2", True, float("nan")):
        malformed = fp("bad", [0, 1], rate=bad_rate)
        with pytest.raises(ValueError, match="leak_rate_Ls"):
            keep(malformed, query, "H3")
        with pytest.raises(ValueError, match="leak_rate_Ls"):
            retain([query], malformed, "H3")


def test_build_kb_forwards_exact_selected_fingerprint_filename(tmp_path, monkeypatch):
    import main as entrypoint
    import types
    # Distinct neighboring files make a directory-only handoff observably wrong.
    selected = dict(fp("selected", [1, 0], node="selected-node"), demand_multiplier=1)
    sibling = dict(fp("sibling", [0, 1], node="wrong-node"), demand_multiplier=1)
    path = corpus_file(tmp_path / "alternate.json", [selected])
    corpus_file(tmp_path / "sensor_fingerprints.json", [sibling])
    config = tmp_path / "cfg.yaml"
    config.write_text("""partitioning:
  num_partitions: 1
sensor_placement:
  total_sensors: 2
knowledge_base:
  nodes_per_partition: 1
  leak_rates_Ls: [2]
  demand_periods:
    - name: day_normal
      multiplier: 1
""", encoding="utf-8")
    captured = {}
    # Hydraulic simulation belongs to the builder integration tests; this seam
    # verifies which physical scenario design the entry point actually hands it.
    def builder(inp, reference_dir, output_dir, fingerprint_file=None):
        actual = fingerprint_file if fingerprint_file is not None else reference_dir / "sensor_fingerprints.json"
        captured["path"] = actual
        captured["scenario"] = json.loads(actual.read_text())["fingerprints"][0]
        return captured
    monkeypatch.setitem(sys.modules, "experiments.reference_corpus", types.SimpleNamespace(rebuild_reference=builder))
    entrypoint.main(["--mode", "build-kb", "--config", str(config),
                     "--fingerprints", str(path), "--output-dir", str(tmp_path / "rebuilt")])
    assert captured["path"] == path
    assert captured["scenario"]["scenario_id"] == "selected"
    assert captured["scenario"]["leak_node"] == "selected-node"
