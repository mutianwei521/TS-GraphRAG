"""TS-GraphRAG entry point; numerical inference/evaluation run fully offline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.configuration import load_config, validate_reference_configuration


def main(argv=None):
    parser = argparse.ArgumentParser(description="TS-GraphRAG reproducibility tools")
    parser.add_argument("--mode", required=True, choices=["partition", "build-kb", "inference", "evaluate"])
    parser.add_argument("--config", type=Path, help="YAML config; paths resolve relative to the config")
    parser.add_argument("--method", choices=["projection", "cosine", "fusion"], help="Override inference.method")
    parser.add_argument("--fingerprints", type=Path, help="Override saved fingerprint corpus")
    parser.add_argument("--query-manifest", type=Path, help="Explicit ordered evaluation query manifest")
    parser.add_argument("--protocol", choices=[f"H{i}" for i in range(10)], help="Retained-corpus deletion rule")
    parser.add_argument("--seed", type=int, help="Evaluation noise seed")
    parser.add_argument("--output-dir", type=Path, help="Evaluation artifact directory")
    parser.add_argument("--observations", type=Path, help="Ordered sensor_id,value or single-row wide CSV")
    parser.add_argument("--inp", type=Path, help="Override EPANET network file")
    parser.add_argument("--k", type=int, help="Override requested number of partitions")
    parser.add_argument("--kb-dir", type=Path, help="Legacy knowledge base directory")
    parser.add_argument("--sensor-results-dir", type=Path, help="Legacy sensor summary directory")
    parser.add_argument("--demand-period", help="Legacy fusion demand period")
    parser.add_argument("--use-llm", action="store_true", help="Enable optional legacy fusion LLM arbitration")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        method = args.method or config.method
        if args.use_llm and method != "fusion":
            raise ValueError("--use-llm requires the explicit --method fusion path")
        corpus = args.fingerprints.resolve() if args.fingerprints else config.fingerprints_path
        if args.mode == "inference":
            if args.observations is None:
                raise ValueError("Inference requires --observations <CSV path>")
            from src.phase3_query.numerical_retrieval import NumericalRetriever, read_observations
            numerical = NumericalRetriever.from_file(corpus)
            y = read_observations(args.observations, numerical.sensor_nodes)
            if method == "fusion":
                # Optional legacy behavior is explicit. Dependency/model errors
                # are exposed; projection is never substituted for fusion.
                from src.phase3_query.leak_locator import LeakLocator
                kb = args.kb_dir.resolve() if args.kb_dir else config.kb_dir
                sensors = args.sensor_results_dir.resolve() if args.sensor_results_dir else config.config_path.parent / "sensor_results"
                locator = LeakLocator(knowledge_base_dir=str(kb), sensor_results_dir=str(sensors))
                result = locator.localize_leak(dict(zip(numerical.sensor_nodes, y)),
                    demand_period=args.demand_period, use_llm=args.use_llm, verbose=args.verbose,
                    propagation_beta=config.beta)
            else:
                if config.beta != 0:
                    raise ValueError("Numerical inference requires inference.beta = 0")
                result = numerical.rank(y, method)
                result["sensor_nodes"] = numerical.sensor_nodes
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
            return result
        if args.mode == "evaluate":
            from src.evaluation.runner import run_evaluation
            result = run_evaluation(config, protocol=args.protocol, seed=args.seed,
                method=method, output_dir=args.output_dir,
                fingerprints_path=args.fingerprints, query_manifest_path=args.query_manifest)
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
            return result
        inp = args.inp.resolve() if args.inp else config.network_inp
        k = args.k if args.k is not None else config.raw["partitioning"]["num_partitions"]
        if args.mode == "build-kb":
            from src.phase3_query.numerical_retrieval import load_fingerprints
            schema, rows = load_fingerprints(corpus)
            validate_reference_configuration(config, schema, rows)
            if k != config.raw["partitioning"]["num_partitions"]:
                raise ValueError("Frozen reference rebuild does not support a different --k")
            # This stage rebuilds the frozen numerical corpus from EPANET;
            # optional semantic/LLM assets have their own explicit legacy path.
            from experiments.reference_corpus import rebuild_reference
            output = args.output_dir or args.kb_dir or config.kb_dir
            return rebuild_reference(inp, corpus.parent, Path(output).resolve(), fingerprint_file=corpus)
        from wds_partition_leiden_main import main as partition_main
        output = args.output_dir or config.config_path.parent / "partition_results_leiden"
        return partition_main(inp_file=str(inp), output_dir=str(Path(output).resolve()), num_partitions=k)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
