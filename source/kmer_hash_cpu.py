
import numpy as np

def fasta_to_matrix(file_name: str, num_reads: int) -> np.ndarray:
  # Allocate memory for the reads matrix
  read_matrix = np.empty((num_reads, 150), dtype=np.uint8)

  i = 0
  f = open(file_name, "rb")
  for line in f:
    if line[0] != 62:
      reads_matrix[i] = np.frombuffer(line, dtype=np.uint8)[:-1]

      i += 1
      if i >= num_reads:
        break
  f.close()

  # If <num_reads> is greater than the number of reads in the fasta file
  if i < num_reads:
    print("\33[91mWarning\33[0m: Fasta file contained less reads than <num_reads>. Resulting matrix has shape ({i}, 150).")
    return reads_matrix[:i, :]

  return reads_matrix

def remap_matrix(matrix: np.ndarray, from_values: np.ndarray, to_values: np.ndarray) -> np.ndarray:
  indices = np.digitize(matrix.ravel(), from_values, right=True)
  return to_values[indices].reshape(matrix.shape)

