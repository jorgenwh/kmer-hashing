
import cupy as cp


def remap_matrix(matrix: cp.array, from_values: cp.array, to_values: cp.array) -> cp.array:
  indices = cp.digitize(matrix.ravel(), from_values, right=True)
  return to_values[indices].reshape(matrix.shape)

def hash_matrix(matrix: cp.array, k: int, power_arr: cp.array) -> cp.array:
  hashes = cp.convolve(matrix.ravel(), power_arr, mode="full")[k - 1:]
  return hashes.reshape(matrix.shape[0], -1)[:, :-(k - 1)]
