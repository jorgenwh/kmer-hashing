from source.profiler import Profiler

if __name__ == "__main__":
  profiler = Profiler(file_name="reads.fa", k=3, num_reads=20000, num_runs=20)
  profiler.benchmark()
  profiler.print_results()
