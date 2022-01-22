import numpy as np

class Timer():
  def __init__(self):
    self.runs = 0
    self.max_time = -np.inf 
    self.min_time = np.inf 
    self.total_time_elapsed = 0

  def add_observation(self, t):
    self.max_time = max(self.max_time, t)
    self.min_time = min(self.min_time, t)
    self.total_time_elapsed += t
    self.runs += 1

  def reset(self):
    self.runs = 0
    self.total_time_elapsed = 0

  def get_runs(self):
    return self.runs

  def get_total_time(self):
    return self.total_time_elapsed

  def get_max_time(self):
    return self.max_time

  def get_min_time(self):
    return self.min_time

  def get_mean_time(self):
    return self.total_time_elapsed / self.runs

