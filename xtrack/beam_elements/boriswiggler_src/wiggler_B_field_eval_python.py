import numpy as np


def evaluate_wiggler_B(x, y, s, k_u, b_tilde):
  """Evaluate the analytic wiggler B-field (Bx, By, Bs) in Tesla."""
  x_arr, y_arr, s_arr = np.broadcast_arrays(
      np.asarray(x, dtype=float),
      np.asarray(y, dtype=float),
      np.asarray(s, dtype=float),
  )
  Bx = np.zeros_like(x_arr, dtype=float)
  By = b_tilde * np.cosh(k_u * y_arr) * np.cos(k_u * s_arr)
  Bs = b_tilde * np.sinh(k_u * y_arr) * np.sin(k_u * s_arr)
  return Bx, By, Bs
