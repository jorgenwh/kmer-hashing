from source.kmer_hash_cpu import fasta_to_matrix

import numpy as np
import cupy as cp
import time

"""
def read_fasta_to_matrix(file_name, num_reads, read_length):
  # Allocate memory
  matrix = np.empty((num_reads, read_length), dtype=np.uint8)

  f = open(file_name, "rb") # read as bytes
  i = 0
  for line in f:
    if line[0] != 62:
      matrix[i] = np.frombuffer(line, dtype=np.uint8)[:-1][:read_length]
      i += 1
      if i == num_reads:
        break
  f.close()
  
  return matrix

def remap_array(arr, from_values, to_values, using_cp=False):
  if not using_cp:  
    index = np.digitize(arr.ravel(), from_values, right=True)
  else:
    index = cp.digitize(arr.ravel(), from_values, right=True)
  return to_values[index].reshape(arr.shape)

def get_hashes(kmer_arr, power_arr, using_cp=False):
  if not using_cp:
    hashes = np.convolve(kmer_arr.ravel(), power_arr, mode="full")[k - 1:]
  else:
    hashes = cp.convolve(kmer_arr.ravel(), power_arr, mode="full")[k - 1:]
  return hashes.reshape(kmer_arr.shape[0], -1)[:,:-(k - 1)]
"""

if __name__ == "__main__":
  file_name = "reads.fa"
  num_reads = 10000

  x = fasta_to_matrix(file_name, num_reads)
