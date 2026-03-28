"""
TS-GraphRAG Phase 2: Build Knowledge Base
==========================================
Main entry script that orchestrates all Phase 2 steps:
  Step 2.1: Static semantic mapping
  Step 2.2: Batch leak simulation
  Step 2.3: LLM scenario summary generation
  Step 2.4: Vector embedding & storage

Usage:
  python build_knowledge_base.py --partition-k 15 --inp dataset/Exa7.inp
  python build_knowledge_base.py --partition-k 15 --dry-run --max-scenarios 3
  python build_knowledge_base.py --partition-k 15 --step 2.3  # run a single step
"""
import argparse
import time

from phase2_knowledge_base.semantic_mapping import run_semantic_mapping
from phase2_knowledge_base.batch_simulation import run_batch_simulation
from phase2_knowledge_base.scenario_summary_generator import run_scenario_summary_generation
from phase2_knowledge_base.vector_store import run_vector_store


def main():
    parser = argparse.ArgumentParser(
        description='TS-GraphRAG Phase 2: Build Dynamic Knowledge Base'
    )
    parser.add_argument(
        '--inp', type=str, default='dataset/Exa7.inp',
        help='Path to EPANET INP file'
    )
    parser.add_argument(
        '--partition-k', type=int, default=15,
        help='Number of Leiden partitions to use'
    )
    parser.add_argument(
        '--partition-file', type=str,
        default='partition_results_leiden/partitions.pkl',
        help='Path to partition results pickle file'
    )
    parser.add_argument(
        '--output-dir', type=str, default='knowledge_base',
        help='Output directory for knowledge base'
    )
    parser.add_argument(
        '--max-scenarios', type=int, default=None,
        help='Limit number of simulation scenarios (for testing)'
    )
    parser.add_argument(
        '--nodes-per-partition', type=int, default=2,
        help='Number of representative leak nodes per partition'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Only run steps 2.1 and 2.2 (skip LLM and vector store)'
    )
    parser.add_argument(
        '--step', type=str, default=None,
        help='Run only a specific step: 2.1, 2.2, 2.3, or 2.4'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("TS-GraphRAG Phase 2: Build Dynamic Knowledge Base")
    print("=" * 60)
    print(f"  INP file:    {args.inp}")
    print(f"  Partitions:  k={args.partition_k}")
    print(f"  Output dir:  {args.output_dir}")
    if args.max_scenarios:
        print(f"  Max scenarios: {args.max_scenarios}")
    if args.dry_run:
        print("  Mode: DRY RUN (skip LLM & vector store)")
    if args.step:
        print(f"  Running only step: {args.step}")
    print()

    t_start = time.time()

    # Step 2.1: Static Semantic Mapping
    if args.step is None or args.step == '2.1':
        t1 = time.time()
        run_semantic_mapping(
            inp_file=args.inp,
            partition_file=args.partition_file,
            partition_k=args.partition_k,
            output_dir=args.output_dir,
        )
        print(f"  [Step 2.1 time: {time.time() - t1:.1f}s]\n")

    # Step 2.2: Batch Leak Simulation
    if args.step is None or args.step == '2.2':
        t2 = time.time()
        run_batch_simulation(
            inp_file=args.inp,
            partition_file=args.partition_file,
            partition_k=args.partition_k,
            output_dir=args.output_dir,
            max_scenarios=args.max_scenarios,
            nodes_per_partition=args.nodes_per_partition,
        )
        print(f"  [Step 2.2 time: {time.time() - t2:.1f}s]\n")

    # Step 2.3: Scenario Summary Generation (LLM)
    if not args.dry_run and (args.step is None or args.step == '2.3'):
        t3 = time.time()
        run_scenario_summary_generation(
            knowledge_base_dir=args.output_dir,
        )
        print(f"  [Step 2.3 time: {time.time() - t3:.1f}s]\n")

    # Step 2.4: Vector Embedding & Storage
    if not args.dry_run and (args.step is None or args.step == '2.4'):
        t4 = time.time()
        run_vector_store(
            knowledge_base_dir=args.output_dir,
            partition_file=args.partition_file,
            partition_k=args.partition_k,
        )
        print(f"  [Step 2.4 time: {time.time() - t4:.1f}s]\n")

    total_time = time.time() - t_start
    print("=" * 60)
    print(f"Phase 2 complete! Total time: {total_time:.1f}s")
    print(f"Knowledge base saved in: {args.output_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
