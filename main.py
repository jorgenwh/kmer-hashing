from source.profiler import Profiler

if __name__ == "__main__":
  profiler = Profiler()
  profiler.benchmark(file_name="reads.fa", k=3, num_reads=10000, num_runs=1)
  profiler.print_results()
