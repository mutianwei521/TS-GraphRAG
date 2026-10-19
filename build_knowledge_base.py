"""Build the TS-GraphRAG numerical scenario library from the released EXA7 model."""
from pathlib import Path
import argparse
from experiments.reference_corpus import rebuild_reference

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inp',type=Path,default=ROOT/'data/Exa7.inp')
    parser.add_argument('--reference-dir',type=Path,default=ROOT/'data/reference')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/reference_corpus')
    parser.add_argument('--partition-k',type=int,default=15)
    parser.add_argument('--max-scenarios','--limit',type=int,default=0)
    args=parser.parse_args()
    if args.partition_k!=15:parser.error('Reference-corpus rebuild uses the frozen K=15 physical design')
    if args.max_scenarios<0:parser.error('--max-scenarios must be nonnegative')
    rebuild_reference(args.inp,args.reference_dir,args.output_dir,args.max_scenarios)


if __name__=='__main__':main()
