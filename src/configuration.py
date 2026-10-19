"""Effective YAML settings, with paths anchored to the configuration file."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {
    "network": {"inp_file": "data/Exa7.inp", "name": "EXA7"},
    "partitioning": {"algorithm": "leiden", "num_partitions": 15},
    "sensor_placement": {"sensors_per_partition": 2, "total_sensors": 30},
    "knowledge_base": {
        "output_dir": "knowledge_base", "nodes_per_partition": 2,
        "leak_rates_Ls": [2, 5, 10, 20, 35, 50],
        "demand_periods": [
            {"name": "night_low", "multiplier": .7},
            {"name": "day_normal", "multiplier": 1.0},
            {"name": "evening_peak", "multiplier": 1.2},
        ],
    },
    "paths": {
        "fingerprints": "data/reference/sensor_fingerprints.json",
        "query_manifest": None, "output_dir": "results/evaluation",
    },
    "inference": {"method": "projection", "beta": 0.0},
    "evaluation": {"noise_sigma": .02, "seed": 41, "protocol": "H0"},
}


def _merge(base: dict, overrides: dict) -> dict:
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


@dataclass(frozen=True)
class AppConfig:
    """Effective settings and resolved inputs; ``as_dict`` preserves the YAML schema."""

    config_path: Path
    raw: dict[str, Any]

    def _path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.config_path.parent / path).resolve()

    @property
    def network_inp(self) -> Path:
        return self._path(self.raw["network"]["inp_file"])

    @property
    def fingerprints_path(self) -> Path:
        return self._path(self.raw["paths"]["fingerprints"])

    @property
    def query_manifest_path(self) -> Path | None:
        value = self.raw["paths"]["query_manifest"]
        return self._path(value) if value is not None else None

    @property
    def output_dir(self) -> Path:
        return self._path(self.raw["paths"]["output_dir"])

    @property
    def kb_dir(self) -> Path:
        return self._path(self.raw["knowledge_base"]["output_dir"])

    @property
    def method(self) -> str:
        return self.raw["inference"]["method"]

    @property
    def beta(self) -> float:
        return float(self.raw["inference"]["beta"])

    @property
    def seed(self) -> int:
        return self.raw["evaluation"]["seed"]

    @property
    def noise_sigma(self) -> float:
        return float(self.raw["evaluation"]["noise_sigma"])

    @property
    def demand_multipliers(self) -> dict[str, float]:
        return {item["name"]: float(item["multiplier"])
                for item in self.raw["knowledge_base"]["demand_periods"]}

    def as_dict(self) -> dict:
        return deepcopy(self.raw)


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load YAML and reject numerical settings that would be silently ignored."""
    config_path = Path(path).resolve() if path is not None else PROJECT_ROOT / "config.yaml"
    with config_path.open(encoding="utf-8") as handle:
        overrides = yaml.safe_load(handle) or {}
    if not isinstance(overrides, dict):
        raise ValueError("Configuration must be a YAML mapping")
    raw = _merge(DEFAULTS, overrides)
    for section in DEFAULTS:
        if not isinstance(raw[section], dict):
            raise ValueError(f"Configuration section {section} must be a mapping")
    if raw['partitioning']['algorithm']!='leiden':
        raise ValueError('partitioning.algorithm must be leiden for this implementation')
    for section,name in [('partitioning','num_partitions'),('sensor_placement','total_sensors'),
                         ('sensor_placement','sensors_per_partition'),('knowledge_base','nodes_per_partition')]:
        value=raw[section][name]
        if isinstance(value,bool) or not isinstance(value,int) or value<1:
            raise ValueError(f'{section}.{name} must be a positive integer')
    if raw['sensor_placement']['sensors_per_partition']*raw['partitioning']['num_partitions']!=raw['sensor_placement']['total_sensors']:
        raise ValueError('total_sensors must equal sensors_per_partition × num_partitions for the fixed physical design')
    if raw["inference"]["method"] not in {"projection", "cosine", "fusion"}:
        raise ValueError("inference.method must be projection, cosine, or fusion")
    beta = raw["inference"]["beta"]
    if isinstance(beta, bool) or not isinstance(beta, (int, float)) or not math.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("inference.beta must be finite and between 0 and 1")
    if raw["inference"]["method"] != "fusion" and beta != 0:
        raise ValueError("Numerical projection/cosine retrieval requires inference.beta = 0")
    sigma = raw["evaluation"]["noise_sigma"]
    if isinstance(sigma, bool) or not isinstance(sigma, (int, float)) or not math.isfinite(sigma) or sigma < 0:
        raise ValueError("evaluation.noise_sigma must be a finite non-negative number")
    seed = raw["evaluation"]["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("evaluation.seed must be a non-negative integer")
    periods = raw["knowledge_base"]["demand_periods"]
    if not isinstance(periods, list) or not periods:
        raise ValueError("knowledge_base.demand_periods must be a non-empty list")
    names = []
    for period in periods:
        if not isinstance(period, dict) or not isinstance(period.get("name"), str):
            raise ValueError("Each demand period requires a name and positive multiplier")
        mult = period.get("multiplier")
        if isinstance(mult, bool) or not isinstance(mult, (int, float)) or not math.isfinite(mult) or mult <= 0:
            raise ValueError("Each demand period requires a positive finite multiplier")
        names.append(period["name"])
    if len(names) != len(set(names)):
        raise ValueError("Demand period names must be unique")
    for section, name in (("network", "inp_file"), ("paths", "fingerprints"),
                          ("paths", "output_dir"), ("knowledge_base", "output_dir")):
        if not isinstance(raw[section][name], str) or not raw[section][name].strip():
            raise ValueError(f"{section}.{name} must be a non-empty path")
    manifest = raw["paths"]["query_manifest"]
    if manifest is not None and (not isinstance(manifest, str) or not manifest.strip()):
        raise ValueError("paths.query_manifest must be null or a non-empty path")
    return AppConfig(config_path=config_path, raw=raw)


def validate_reference_configuration(config: AppConfig, sensor_nodes: list[str], fingerprints: list[dict]) -> None:
    """Refuse custom rebuild settings that the frozen physical design cannot use."""
    expected_rates = sorted({float(row["leak_rate_Ls"]) for row in fingerprints})
    configured_rates = sorted(float(rate) for rate in config.raw["knowledge_base"]["leak_rates_Ls"])
    periods = {}
    for row in fingerprints:
        name, multiplier = row["demand_period"], float(row["demand_multiplier"])
        if name in periods and periods[name] != multiplier:
            raise ValueError("Reference demand-period multipliers are inconsistent")
        periods[name] = multiplier
    nodes = {}
    for row in fingerprints:
        nodes.setdefault(row["leak_partition"], set()).add(row["leak_node"])
    matching = (
        configured_rates == expected_rates and config.demand_multipliers == periods
        and config.raw["sensor_placement"]["total_sensors"] == len(sensor_nodes)
        and config.raw["partitioning"]["num_partitions"] == len(nodes)
        and all(len(group) == config.raw["knowledge_base"]["nodes_per_partition"] for group in nodes.values())
    )
    if not matching:
        raise ValueError("Configuration differs from the frozen reference design. Use its saved rates, demand periods, sensor count, partitions and nodes per partition; custom corpus design requires a separate explicit experiment.")
