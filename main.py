"""
TS-GraphRAG: Topology-Stratified Graph RAG for Leak Localization
================================================================

Main executable for the full TS-GraphRAG pipeline.

Usage:
  # Phase 1: Hydraulic partitioning & sensor placement
  python main.py --mode partition --inp data/Exa7.inp --k 15

  # Phase 2: Build knowledge base (offline)
  python main.py --mode build-kb --inp data/Exa7.inp --k 15

  # Phase 3: Real-time leak localization (inference)
  python main.py --mode inference --kb-dir knowledge_base --observations data/test_obs.csv

  # Full pipeline evaluation
  python main.py --mode evaluate --kb-dir knowledge_base
"""

import argparse
import sys
import os


def main():
    parser = argparse.ArgumentParser(
        description='TS-GraphRAG: Topology-Stratified Graph RAG for Leak Localization'
    )
    parser.add_argument(
        '--mode', type=str, required=True,
        choices=['partition', 'build-kb', 'inference', 'evaluate'],
        help='Pipeline stage to execute'
    )
    parser.add_argument('--inp', type=str, default='data/Exa7.inp',
                        help='Path to EPANET INP file')
    parser.add_argument('--k', type=int, default=15,
                        help='Number of Leiden partitions')
    parser.add_argument('--kb-dir', type=str, default='knowledge_base',
                        help='Knowledge base directory')
    parser.add_argument('--observations', type=str, default=None,
                        help='Path to sensor observations CSV (inference mode)')
    parser.add_argument('--use-llm', action='store_true',
                        help='Enable confidence-gated LLM arbitration')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Enable verbose output')

    args = parser.parse_args()

    if args.mode == 'partition':
        print("=" * 60)
        print("Phase 1: Hydraulic Partitioning & Sensor Placement")
        print("=" * 60)
        # Run Leiden partitioning
        from wds_partition_leiden_main import main as partition_main
        partition_main()

    elif args.mode == 'build-kb':
        print("=" * 60)
        print("Phase 2: Build Knowledge Base")
        print("=" * 60)
        from build_knowledge_base import main as kb_main
        sys.argv = [
            'build_knowledge_base.py',
            '--inp', args.inp,
            '--partition-k', str(args.k),
            '--output-dir', args.kb_dir,
        ]
        kb_main()

    elif args.mode == 'inference':
        print("=" * 60)
        print("Phase 3: Real-Time Leak Localization")
        print("=" * 60)
        from src.phase3_query.leak_locator import LeakLocator
        locator = LeakLocator(knowledge_base_dir=args.kb_dir)

        if args.observations:
            import csv
            obs = {}
            with open(args.observations, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    obs[row['sensor_node']] = float(row['pressure_drop'])
            result = locator.localize_leak(obs, use_llm=args.use_llm,
                                           verbose=args.verbose)
            print(f"\n{'=' * 40}")
            print(f"Predicted Partition: {result['predicted_partition']}")
            print(f"Confidence: {result['confidence']}")
            print(f"Top-3: {result['top3_partitions']}")
        else:
            print("Please provide --observations <CSV path> for inference.")

    elif args.mode == 'evaluate':
        print("=" * 60)
        print("Comprehensive Evaluation")
        print("=" * 60)
        from src.phase3_query.eval_comprehensive import main as eval_main
        eval_main()

    print("\nDone.")


if __name__ == '__main__':
    main()
