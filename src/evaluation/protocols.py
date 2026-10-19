"""One implementation of the frozen H0--H9 retained-corpus rules."""
from __future__ import annotations

import math
from numbers import Real


PROTOCOLS = tuple(f"H{i}" for i in range(10))


def validate_scenario_metadata(row: dict) -> None:
    """Validate factor identity before deletion without coercing its types."""
    if not isinstance(row, dict):
        raise ValueError("Scenario metadata must be a mapping")
    for field in ("leak_node", "demand_period"):
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string")
    rate = row.get("leak_rate_Ls")
    if isinstance(rate, bool) or not isinstance(rate, Real) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("leak_rate_Ls must be a positive finite numeric value, excluding bool")


def keep(candidate: dict, query: dict, protocol: str) -> bool:
    """Whether a scenario survives deletion for this query, preserving row order."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown protocol {protocol!r}; expected one of {PROTOCOLS}")
    validate_scenario_metadata(candidate)
    validate_scenario_metadata(query)
    sn, sq, st = [candidate[key] == query[key]
                  for key in ("leak_node", "leak_rate_Ls", "demand_period")]
    return {
        "H0": True, "H1": not (sn and sq and st), "H2": not sn,
        "H3": not sq, "H4": not st, "H5": not (sn or sq),
        "H6": not (sn or st), "H7": not (sq or st),
        "H8": not (sn or sq or st),
        "H9": candidate["leak_partition"] != query["leak_partition"],
    }[protocol]


def retain(fingerprints: list[dict], query: dict, protocol: str) -> list[dict]:
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown protocol {protocol!r}")
    validate_scenario_metadata(query)
    return [row for row in fingerprints if keep(row, query, protocol)]


def allowed_candidate_domain(fingerprints: list[dict]) -> list[int]:
    """Retained labels, including zero-only labels that receive no numerical rank."""
    return sorted({int(row["leak_partition"]) for row in fingerprints})
