from source.profiler import Profiler

import sys
import argparse

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Profile arguments")
  parser.add_argument("-k", help="kmer size.", type=int, default=3)
  parser.add_argument("-reads", help="number of reads in matrix.", type=int, default=100000)
  parser.add_argument("-runs", help="number of runs performed (discounting warmup runs).", type=int, default=20)
  parser.add_argument("-nowarmup", action="store_true")
  args = parser.parse_args()

  #profiler = Profiler(file_name="reads.fa", k=args.k, num_reads=args.reads, num_runs=args.runs)
  profiler = Profiler(file_name="reads.fa", k=args.k, num_reads=args.reads, num_runs=args.runs, nowarmup=args.nowarmup)
  profiler.benchmark()
  profiler.print_results()
