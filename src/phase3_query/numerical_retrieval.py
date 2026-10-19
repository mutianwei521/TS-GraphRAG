"""Frozen projection-residual and cosine retrieval using NumPy only.

Expressions, eligibility thresholds and default np.argsort tie ordering match
strict_engine.Pool.prepare (2026-10-10). Both rules use every retained period.
No template rescaling, clipping or graph propagation is added.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from src.evaluation.protocols import validate_scenario_metadata


METHODS = ("projection", "cosine")


def validate_sensor_order(sensor_nodes: Sequence[str], expected: Sequence[str]) -> None:
    actual, canonical = list(sensor_nodes), list(expected)
    if actual != canonical:
        raise ValueError(f"Observation sensor order must exactly match corpus sensor order: {canonical}; got {actual}")


def _vector(values, count: int, name: str) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if vector.shape != (count,):
        raise ValueError(f"{name} must have exactly {count} ordered sensor values; got shape {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def load_fingerprints(path: str | Path) -> tuple[list[str], list[dict]]:
    """Load the ordered sensor schema and validate every corpus row."""
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or "sensor_nodes" not in payload or "fingerprints" not in payload:
        raise ValueError("Fingerprint JSON requires sensor_nodes and fingerprints")
    pool = NumericalRetriever(payload["sensor_nodes"], payload["fingerprints"])
    if "num_sensors" in payload and payload["num_sensors"] != len(pool.sensor_nodes):
        raise ValueError("num_sensors does not match sensor_nodes")
    if "num_scenarios" in payload and payload["num_scenarios"] != len(pool.fingerprints):
        raise ValueError("num_scenarios does not match fingerprints")
    return pool.sensor_nodes, pool.fingerprints


def read_observations(path: str | Path, sensor_nodes: Sequence[str]) -> np.ndarray:
    """Read ordered sensor_id,value rows or exactly one wide observation row.

    Duplicate, missing, extra or reordered sensors fail before scoring.
    """
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2:
        raise ValueError("Observations CSV requires a header and sensor values")
    header, data = rows[0], rows[1:]
    if header == ["sensor_id", "value"]:
        if any(len(row) != 2 for row in data):
            raise ValueError("Each sensor_id,value row must have exactly two fields")
        validate_sensor_order([row[0] for row in data], sensor_nodes)
        values = [row[1] for row in data]
    else:
        validate_sensor_order(header, sensor_nodes)
        if len(data) != 1:
            raise ValueError("Wide observations CSV must contain exactly one observation row")
        values = data[0]
    return _vector(values, len(sensor_nodes), "Observation")


class NumericalRetriever:
    """An ordered retained corpus with scenario and distinct-partition rankings."""

    def __init__(self, sensor_nodes: Sequence[str], fingerprints: Sequence[dict]):
        if not isinstance(sensor_nodes, (list, tuple)) or not sensor_nodes:
            raise ValueError("sensor_nodes must be a non-empty ordered list")
        if not all(isinstance(node, str) and node for node in sensor_nodes):
            raise ValueError("sensor_nodes must contain non-empty string identifiers")
        if len(set(sensor_nodes)) != len(sensor_nodes):
            raise ValueError("sensor_nodes must not contain duplicates")
        self.sensor_nodes = list(sensor_nodes)
        if not isinstance(fingerprints, (list, tuple)):
            raise ValueError("fingerprints must be an ordered list")
        self.fingerprints = list(fingerprints)
        ids, matrix, labels = [], [], []
        for row in self.fingerprints:
            if not isinstance(row, dict):
                raise ValueError("Each fingerprint must be a mapping")
            required = {"scenario_id", "sensor_fingerprint", "leak_partition",
                        "leak_node", "leak_rate_Ls", "demand_period"}
            if not required <= row.keys():
                raise ValueError(f"Fingerprint is missing fields: {sorted(required - row.keys())}")
            validate_scenario_metadata(row)
            sid, label = row["scenario_id"], row["leak_partition"]
            if not isinstance(sid, str) or not sid:
                raise ValueError("scenario_id must be a non-empty string")
            if isinstance(label, bool) or not isinstance(label, (int, np.integer)) or label < 0:
                raise ValueError("leak_partition must be a non-negative integer")
            ids.append(sid)
            labels.append(int(label))
            matrix.append(_vector(row["sensor_fingerprint"], len(self.sensor_nodes), f"Fingerprint {sid}"))
        if len(set(ids)) != len(ids):
            raise ValueError("scenario_id values must be unique")
        self.ids = ids
        self.D = np.asarray(matrix, dtype=float).reshape((-1, len(self.sensor_nodes)))
        self.norms = np.linalg.norm(self.D, axis=1)
        self.partitions = np.asarray(labels, dtype=int)
        self.labels = np.unique(self.partitions)

    @classmethod
    def from_file(cls, path: str | Path) -> "NumericalRetriever":
        sensor_nodes, fingerprints = load_fingerprints(path)
        return cls(sensor_nodes, fingerprints)

    def prepare(self, y) -> dict[str, np.ndarray]:
        """Score all retained rows, excluding zero support before assigning ranks."""
        y = _vector(y, len(self.sensor_nodes), "Observation")
        norm = np.linalg.norm(y)
        cos = self.D @ y / np.maximum(self.norms * norm, 1e-18)
        projection = np.linalg.norm(
            y[None, :] - (self.D @ y / np.maximum(self.norms ** 2, 1e-18))[:, None] * self.D,
            axis=1,
        )
        projection[self.norms == 0] = np.inf
        cos[self.norms <= 1e-12] = -np.inf
        projection_eligible = np.flatnonzero(self.norms > 0)
        cosine_eligible = np.flatnonzero(self.norms > 1e-12)
        return {
            "projection_residuals": projection, "cosine_scores": cos,
            # Preserve default NumPy tie behavior; stable/ID sorting changes it.
            "projection_order": projection_eligible[np.argsort(projection[projection_eligible])],
            "cosine_order": cosine_eligible[np.argsort(-cos[cosine_eligible])],
        }

    def baseline_rank(self, order: Sequence[int]) -> list[int]:
        """Group scenario ranks, keeping each label's first occurrence."""
        return list(dict.fromkeys(int(self.partitions[index]) for index in order))

    def rank(self, y, method: str = "projection", sensor_nodes: Sequence[str] | None = None) -> dict:
        if method not in METHODS:
            raise ValueError(f"Numerical method must be one of {METHODS}")
        if sensor_nodes is not None:
            validate_sensor_order(sensor_nodes, self.sensor_nodes)
        if isinstance(y, Mapping):
            validate_sensor_order(list(y), self.sensor_nodes)
            y = list(y.values())
        prepared = self.prepare(y)
        order = prepared[f"{method}_order"]
        ranking = self.baseline_rank(order)
        scores = prepared["projection_residuals" if method == "projection" else "cosine_scores"]
        return {
            "method": method,
            "predicted_partition": ranking[0] if ranking else None,
            "top3_partitions": ranking[:3], "ranking": ranking,
            "scenario_ranking": [self.ids[index] for index in order],
            "scores": [float(scores[index]) for index in order],
            "score_direction": "ascending" if method == "projection" else "descending",
            "allowed_candidate_domain": self.labels.tolist(),
            "n_retained": len(self.ids), "n_eligible": len(order),
        }
