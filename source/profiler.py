from source.timer import Timer
from source.reader import fasta_to_matrix

from source.kmer_hash_cpu import remap_matrix as remap_matrix_cpu, hash_matrix as hash_matrix_cpu
from source.kmer_hash_gpu import remap_matrix as remap_matrix_gpu, hash_matrix as hash_matrix_gpu

import numpy as np
import cupy as cp
import time

class Profiler():
  """
  """
  def __init__(self, file_name, k, num_reads, num_runs):
    self.file_name = file_name  
    self.k = k
    self.num_reads = num_reads
    self.num_runs = num_runs

    # Total runtime to perform matrix remapping + kmer hashing (and mem transfer for GPU)
    self.full_cpu_runtime_timer = Timer()
    self.full_gpu_runtime_timer = Timer()

    # Time elapsed moving reads matrix to memory (only needs to be done for GPU)
    self.move_to_gpu_mem_timer = Timer() 

    # Time elapsed to remap np.ndarray/cp.array
    self.remap_matrix_cpu_timer = Timer() 
    self.remap_matrix_gpu_timer = Timer() 

    # Time elapsed to compute the kmer hashes
    self.hash_cpu_timer = Timer() 
    self.hash_gpu_timer = Timer() 

    # Time to initialize utility arrays
    self.util_arrays_cpu_timer = Timer()
    self.util_arrays_gpu_timer = Timer()

  def reset(self):
    self.move_to_gpu_mem_timer.reset()
    self.remap_matrix_cpu_timer.reset()
    self.remap_matrix_gpu_timer.reset()
    self.hash_cpu_timer.reset()
    self.hash_gpu_timer.reset()

  def benchmark(self):
    """...

    Args:
      file_name (str): the FASTA file name containing the reads.
      k (int): the kmer size.
      num_reads (int): number of reads to fetch from the FASTA file into one matrix.
      num_runs (int): number of benchmark runs to measure.
    """
    # Read the FASTA file to memory. This process seemingly will not change whether CPU or GPU is used as the reads matrix must be read into a numpy array before it can be transferred to the GPU memory
    reads_matrix = fasta_to_matrix(self.file_name, self.num_reads)
    
    for i in range(self.num_runs + 3): # 3 'warmup' runs
      np_hashes = self.benchmark_cpu(np.copy(reads_matrix), warmup=i>=3)
      cp_hashes = self.benchmark_gpu(np.copy(reads_matrix), warmup=i>=3)

      assert np.array_equal(cp.asnumpy(cp_hashes), np_hashes), "Resulting hashes from CPU and GPU differ"

  def benchmark_cpu(self, reads_matrix, warmup):
    # Prepare utility arrays. The timing for this is not counted in the total runtime
    t1 = time.time_ns()
    from_values = np.array([65, 67, 71, 84, 97, 99, 103, 116], dtype=np.uint64)
    to_values = np.array([0, 1, 3, 2, 0, 1, 3, 2], dtype=np.uint64)
    power_arr = np.power(4, np.arange(self.k, dtype=np.uint64), dtype=np.uint64)
    t_utils = time.time_ns() - t1

    total_runtime = 0

    # Measure time to remap matrix
    t1 = time.time_ns()
    reads_matrix = remap_matrix_cpu(reads_matrix, from_values, to_values)
    t_remap = time.time_ns() - t1
    total_runtime += t_remap

    # Measure time to hash kmers
    t1 = time.time_ns()
    hashes = hash_matrix_cpu(reads_matrix, self.k, power_arr)
    t_hash = time.time_ns() - t1
    total_runtime += t_hash

    if not warmup:
      self.util_arrays_cpu_timer.add_observation(t_utils / 1e6)
      self.remap_matrix_cpu_timer.add_observation(t_remap / 1e6)
      self.hash_cpu_timer.add_observation(t_hash / 1e6)
      self.full_cpu_runtime_timer.add_observation(total_runtime / 1e6)

    return hashes

  def benchmark_gpu(self, reads_matrix, warmup):
    # Prepare utility arrays. The timing for this is not counted in the total runtime
    t1 = time.time_ns()
    from_values = cp.array([65, 67, 71, 84, 97, 99, 103, 116], dtype=cp.uint64)
    to_values = cp.array([0, 1, 3, 2, 0, 1, 3, 2], dtype=cp.uint64)
    power_arr = cp.power(4, cp.arange(self.k, dtype=cp.uint64), dtype=cp.uint64)
    t_utils = time.time_ns() - t1

    total_runtime = 0

    # Measure time to transfer reads_matrix to GPU memory
    t1 = time.time_ns()
    reads_matrix = cp.asarray(reads_matrix)
    t_transfer = time.time_ns() - t1
    total_runtime += t_transfer

    # Measure time to remap matrix
    t1 = time.time_ns()
    reads_matrix = remap_matrix_gpu(reads_matrix, from_values, to_values)
    t_remap = time.time_ns() - t1
    total_runtime += t_remap

    # Measure time to hash kmers
    t1 = time.time_ns()
    hashes = hash_matrix_gpu(reads_matrix, self.k, power_arr)
    t_hash = time.time_ns() - t1
    total_runtime += t_hash

    if not warmup:
      self.util_arrays_gpu_timer.add_observation(t_utils / 1e6)
      self.move_to_gpu_mem_timer.add_observation(t_transfer / 1e6)
      self.remap_matrix_gpu_timer.add_observation(t_remap / 1e6)
      self.hash_gpu_timer.add_observation(t_hash / 1e6)
      self.full_gpu_runtime_timer.add_observation(total_runtime / 1e6)

    return hashes

  def print_results(self):
    # CPU variables (mean/+/-)
    init_util_arrays_cpu = [
        self.util_arrays_cpu_timer.get_mean_time(),
        (self.util_arrays_cpu_timer.get_max_time() - self.util_arrays_cpu_timer.get_mean_time()),
        (self.util_arrays_cpu_timer.get_mean_time() - self.util_arrays_cpu_timer.get_min_time())
    ]
    matrix_remap_cpu = [
        self.remap_matrix_cpu_timer.get_mean_time(),
        (self.remap_matrix_cpu_timer.get_max_time() - self.remap_matrix_cpu_timer.get_mean_time()),
        (self.remap_matrix_cpu_timer.get_mean_time() - self.remap_matrix_cpu_timer.get_min_time())
    ]
    matrix_hash_cpu = [
        self.hash_cpu_timer.get_mean_time(),
        (self.hash_cpu_timer.get_max_time() - self.hash_cpu_timer.get_mean_time()),
        (self.hash_cpu_timer.get_mean_time() - self.hash_cpu_timer.get_min_time())
    ]
    total_runtime_cpu = [
        self.full_cpu_runtime_timer.get_mean_time(),
        (self.full_cpu_runtime_timer.get_max_time() - self.full_cpu_runtime_timer.get_mean_time()),
        (self.full_cpu_runtime_timer.get_mean_time() - self.full_cpu_runtime_timer.get_min_time())
    ]

    # GPU variables
    init_util_arrays_gpu = [
        self.util_arrays_gpu_timer.get_mean_time(),
        (self.util_arrays_gpu_timer.get_max_time() - self.util_arrays_gpu_timer.get_mean_time()),
        (self.util_arrays_gpu_timer.get_mean_time() - self.util_arrays_gpu_timer.get_min_time())
    ]
    transfer2gpu_mem  = [
        self.move_to_gpu_mem_timer.get_mean_time(),
        (self.move_to_gpu_mem_timer.get_max_time() - self.move_to_gpu_mem_timer.get_mean_time()),
        (self.move_to_gpu_mem_timer.get_mean_time() - self.move_to_gpu_mem_timer.get_min_time())
    ]
    matrix_remap_gpu = [
        self.remap_matrix_gpu_timer.get_mean_time(),
        (self.remap_matrix_gpu_timer.get_max_time() - self.remap_matrix_gpu_timer.get_mean_time()),
        (self.remap_matrix_gpu_timer.get_mean_time() - self.remap_matrix_gpu_timer.get_min_time())
    ]
    matrix_hash_gpu = [
        self.hash_gpu_timer.get_mean_time(),
        (self.hash_gpu_timer.get_max_time() - self.hash_gpu_timer.get_mean_time()),
        (self.hash_gpu_timer.get_mean_time() - self.hash_gpu_timer.get_min_time())
    ]
    total_runtime_gpu = [
        self.full_gpu_runtime_timer.get_mean_time(),
        (self.full_gpu_runtime_timer.get_max_time() - self.full_gpu_runtime_timer.get_mean_time()),
        (self.full_gpu_runtime_timer.get_mean_time() - self.full_gpu_runtime_timer.get_min_time())
    ]

    endc = "\33[0m"
    bold = "\33[1m"
    gray = "\33[2m"
    green = "\33[92m"
    blue = "\33[94m"

    print(f"{gray}----------{endc} {bold}Time Statistics{endc} {gray}----------{endc}")
    print(f"{blue}k                               : {endc}{bold}{self.k}{endc}")
    print(f"{blue}Number of reads (of length 150) : {endc}{bold}{self.num_reads}{endc}")
    print(f"{blue}Runs performed                  : {endc}{bold}{self.num_runs}{endc}\n")

    print(f"{gray}-------------------------------------{endc}")
    print(f"{gray}----------------{endc}{green} CPU {endc}{gray}----------------{endc}")
    print(f"{gray}-------------------------------------{endc}")

    print("Initializing utility arrays (not counted in total runtime):")
    print(f"Mean time   : {bold}{round(init_util_arrays_cpu[0], 5)}{endc} ms")
    print(f"+ / -       : {bold}{round(init_util_arrays_cpu[1], 5)}{endc} ms / {bold}{round(init_util_arrays_cpu[2], 5)}{endc} ms\n")

    print("Matrix remapping:")
    print(f"Mean time   : {bold}{round(matrix_remap_cpu[0], 5)}{endc} ms")
    print(f"+ / -       : {bold}{round(matrix_remap_cpu[1], 5)}{endc} ms / {bold}{round(matrix_remap_cpu[2], 5)}{endc} ms\n")

    print("Kmer hashing:")
    print(f"Mean time   : {bold}{round(matrix_hash_cpu[0], 5)}{endc} ms")
    print(f"+ / -       : {bold}{round(matrix_hash_cpu[1], 5)}{endc} ms / {bold}{round(matrix_hash_cpu[2], 5)}{endc} ms\n")

    print("Total runtime:")
    print(f"Mean time   : {bold}{round(total_runtime_cpu[0], 5)}{endc} ms")
    print(f"+ / -       : {bold}{round(total_runtime_cpu[1], 5)}{endc} ms / {bold}{round(total_runtime_cpu[2], 5)}{endc} ms\n")

    print(f"{gray}-------------------------------------{endc}")
    print(f"{gray}----------------{endc}{green} GPU {endc}{gray}----------------{endc}")
    print(f"{gray}-------------------------------------{endc}")

    print("Initializing utility arrays (not counted in total runtime):")
    print(f"Mean time   : {bold}{round(init_util_arrays_gpu[0], 5)}{endc} ms")
    print(f"+ / -       : {bold}{round(init_util_arrays_gpu[1], 5)}{endc} ms / {bold}{round(init_util_arrays_gpu[2], 5)}{endc} ms\n")

    print("Data transfer:")
    print(f"Mean time   : {bold}{round(transfer2gpu_mem[0], 5)}{endc} ms")
    print(f"+ / -       : {bold}{round(transfer2gpu_mem[1], 5)}{endc} ms / {bold}{round(transfer2gpu_mem[2], 5)}{endc} ms\n")

    print("Matrix remapping:")
    print(f"Mean time   : {bold}{round(matrix_remap_gpu[0], 5)}{endc} ms")
    print(f"+ / -       : {bold}{round(matrix_remap_gpu[1], 5)}{endc} ms / {bold}{round(matrix_remap_gpu[2], 5)}{endc} ms\n")

    print("Kmer hashing:")
    print(f"Mean time   : {bold}{round(matrix_hash_gpu[0], 5)}{endc} ms")
    print(f"+ / -       : {bold}{round(matrix_hash_gpu[1], 5)}{endc} ms / {bold}{round(matrix_hash_gpu[2], 5)}{endc} ms\n")

    print("Total runtime:")
    print(f"Mean time   : {bold}{round(total_runtime_gpu[0], 5)}{endc} ms")
    print(f"+ / -       : {bold}{round(total_runtime_gpu[1], 5)}{endc} ms / {bold}{round(total_runtime_gpu[2], 5)}{endc} ms")
