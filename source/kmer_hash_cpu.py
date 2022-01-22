
import numpy as np


def remap_matrix(matrix: np.ndarray, from_values: np.ndarray, to_values: np.ndarray) -> np.ndarray:
  indices = np.digitize(matrix.ravel(), from_values, right=True)
  return to_values[indices].reshape(matrix.shape)

def hash_matrix(matrix: np.ndarray, k: int, power_arr: np.ndarray) -> np.ndarray:
  hashes = np.convolve(matrix.ravel(), power_arr, mode="full")[k - 1:]
  return hashes.reshape(matrix.shape[0], -1)[:, :-(k - 1)]
