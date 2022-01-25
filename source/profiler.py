from source.timer import Timer
from source.reader import fasta_to_matrix

from source.kmer_hash_cpu import remap_matrix as remap_matrix_cpu, hash_matrix as hash_matrix_cpu
from source.kmer_hash_gpu import remap_matrix as remap_matrix_gpu, hash_matrix as hash_matrix_gpu

from collections import defaultdict
import numpy as np
import cupy as cp
import time

class Profiler():
  def __init__(self, file_name, k, num_reads, num_runs, nowarmup):
    self.k = k
    self.num_reads = num_reads
    self.num_runs = num_runs
    self.warmup_runs = 0 if nowarmup else 3 
    print(self.warmup_runs)

    self.reads_matrix = fasta_to_matrix(file_name, self.num_reads)
    if self.reads_matrix.shape[0] != self.num_reads:
      self.num_reads = self.reads_matrix.shape[0]

    # Each key represents something that is being timed, and each value is a list
    # containing its total time elapsed for all runs, its slowest run and its fastest run
    self.cpu_time_data = {} 
    self.gpu_time_data = {} 

  def benchmark(self):
    # Run CPU benchmarks
    print("Running CPU benchmarks ...")
    for i in range(self.num_runs + self.warmup_runs): 
      np_hashes = self.benchmark_cpu(np.copy(self.reads_matrix), warmup=i<self.warmup_runs)

    # Run GPU benchmarks
    print("Running GPU benchmarks ...")
    for i in range(self.num_runs + self.warmup_runs): 
      cp_hashes = self.benchmark_gpu(np.copy(self.reads_matrix), warmup=i<self.warmup_runs)

    assert np.array_equal(np_hashes, cp_hashes), "Resulting hashes from CPU and GPU differ"

  def _add_time_data(self, time_data, device):
    d = self.cpu_time_data if device == "cpu" else self.gpu_time_data
    for key in time_data:
      if key not in d:
        d[key] = [0, -float("inf"), float("inf")]
      d[key][0] += time_data[key]
      d[key][1] = max(time_data[key], d[key][1])
      d[key][2] = min(time_data[key], d[key][2])

  def benchmark_cpu(self, reads_matrix, warmup):
    time_data = {}
    start = time.time_ns()

    # Initialize necessary utility arrays
    from_values = np.array([65, 67, 71, 84, 97, 99, 103, 116], dtype=np.uint64)
    to_values = np.array([0, 1, 3, 2, 0, 1, 3, 2], dtype=np.uint64)
    power_arr = np.power(4, np.arange(self.k, dtype=np.uint64), dtype=np.uint64)
    time_data["utility-array-init"] = time.time_ns() - start

    # Remap reads matrix values
    ts = time.time_ns()
    reads_matrix = remap_matrix_cpu(reads_matrix, from_values, to_values)
    time_data["matrix-remapping"] = time.time_ns() - ts 

    # Hash the kmers
    ts = time.time_ns()
    hashes = hash_matrix_cpu(reads_matrix, self.k, power_arr)
    time_data["matrix-hashing"] = time.time_ns() - ts 

    time_data["total-time"] = time.time_ns() - start
    if not warmup:
      self._add_time_data(time_data, "cpu")
    return hashes

  def benchmark_gpu(self, reads_matrix, warmup):
    time_data = {}
    start = time.time_ns()

    # Initialize necessary utility arrays
    from_values = cp.array([65, 67, 71, 84, 97, 99, 103, 116], dtype=cp.uint64)
    to_values = cp.array([0, 1, 3, 2, 0, 1, 3, 2], dtype=cp.uint64)
    power_arr = cp.power(4, cp.arange(self.k, dtype=cp.uint64), dtype=cp.uint64)
    time_data["utility-array-init"] = time.time_ns() - start

    # Transfer reads matrix to GPU memory
    ts = time.time_ns()
    reads_matrix = cp.asarray(reads_matrix)
    time_data["numpy2gpu"] = time.time_ns() - ts 

    # Remap reads matrix values
    ts = time.time_ns()
    reads_matrix = remap_matrix_gpu(reads_matrix, from_values, to_values)
    time_data["matrix-remapping"] = time.time_ns() - ts 

    # Hash the kmers
    ts = time.time_ns()
    hashes = hash_matrix_gpu(reads_matrix, self.k, power_arr)
    time_data["matrix-hashing"] = time.time_ns() - ts 

    # Fetch the kmer hashes back from the GPU memory
    ts = time.time_ns()
    hashes = cp.asnumpy(hashes)
    time_data["gpu2numpy"] = time.time_ns() - ts 

    time_data["total-time"] = time.time_ns() - start
    if not warmup:
      self._add_time_data(time_data, "gpu")
    return hashes

  def print_results(self):
    import math
    import os
    _, width = os.popen("stty size", "r").read().split()
    width = int(width)
    endc = "\33[0m"
    bold = "\33[1m"
    gray = "\33[2m"
    green = "\33[92m"
    blue = "\33[94m"

    print(f"{gray} {''.join(['-' for _ in range(int(math.floor(width - 19) / 2))])} {endc}{bold}Time Statistics{endc}{gray} {''.join(['-' for _ in range(int(math.ceil((width - 19) / 2)))])} {endc}")
    print(f"  {blue}k{endc}                               : {bold}{self.k}{endc}")
    print(f"  {blue}Number of reads (of length 150){endc} : {bold}{self.num_reads}{endc}")
    print(f"  {blue}Runs performed{endc}                  : {bold}{self.num_runs}{endc}")

    # CPU
    print(f"{gray} {''.join(['-' for _ in range(int(width - 2))])} {endc}")
    print(f"{gray} {''.join(['-' for _ in range(int(math.floor(width - 7) / 2))])} {endc}{bold}{green}CPU{endc}{gray} {''.join(['-' for _ in range(int(math.ceil((width - 7) / 2)))])} {endc}")
    print(f"{gray} {''.join(['-' for _ in range(int(width - 2))])} {endc}\n")

    for key in self.cpu_time_data:
      mean = round((self.cpu_time_data[key][0] / self.num_runs) / 1e6, 4)
      p = round((self.cpu_time_data[key][1] - mean) / 1e6, 4)
      m = round((mean - self.cpu_time_data[key][2]) / 1e6, 4)

      print(f" {gray}-----------{endc} {bold}{key}{endc}:")
      print(f" mean      : {bold}{mean}{endc} ms")
      print(f" +/-       : {bold}{p}{endc} ms / {m} ms\n")

    # GPU
    print(f"{gray} {''.join(['-' for _ in range(int(width - 2))])} {endc}")
    print(f"{gray} {''.join(['-' for _ in range(int(math.floor(width - 7) / 2))])} {endc}{bold}{green}GPU{endc}{gray} {''.join(['-' for _ in range(int(math.ceil((width - 7) / 2)))])} {endc}")
    print(f"{gray} {''.join(['-' for _ in range(int(width - 2))])} {endc}")

    for key in self.gpu_time_data:
      mean = round((self.gpu_time_data[key][0] / self.num_runs) / 1e6, 4)
      p = round((self.gpu_time_data[key][1] - mean) / 1e6, 4)
      m = round((mean - self.gpu_time_data[key][2]) / 1e6, 4)

      print(f" {gray}-----------{endc} {bold}{key}{endc}:")
      print(f" mean      : {bold}{mean}{endc} ms")
      print(f" +/-       : {bold}{p}{endc} ms / {m} ms\n")

