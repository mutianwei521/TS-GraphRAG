"""Partition Top-1, Top-3 and full returned-label reciprocal rank."""
from __future__ import annotations


def rank_metrics(ranking: list[int], truth: int) -> dict:
    if len(ranking) != len(set(ranking)):
        raise ValueError("Metrics require distinct partition labels in ranked order")
    return {
        "top1": int(bool(ranking) and ranking[0] == truth),
        "top3": int(truth in ranking[:3]),
        "rr": 1 / (ranking.index(truth) + 1) if truth in ranking else 0.0,
    }


def aggregate_metrics(records: list[dict]) -> dict:
    """Every query contributes, including empty/zero-support rankings."""
    count = len(records)
    if count == 0:
        raise ValueError("Cannot evaluate an empty query manifest")
    return {
        "n_queries": count,
        "top1_accuracy": sum(row["top1"] for row in records) / count,
        "top3_accuracy": sum(row["top3"] for row in records) / count,
        "mrr": sum(row["rr"] for row in records) / count,
    }
