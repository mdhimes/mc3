import numpy as np


def quad(p, x):
  """
  Quadratic polynomial function.

  Parameters:
  -----------
  p: ndarray
     Polynomial constant, linear, and quadratic coefficients.
  x: ndarray
     Array of dependent variables where to evaluate the polynomial.

  Returns:
  --------
  y: ndarray
     The polinomial evaluated at x:  y = p0 + p1*x + p2*x^2

  Modification History:
  ---------------------
  2014-04-17  patricio  Initial implementation.  pcubillos@fulbrightmail.org
  """
  if len(p.shape) == 1:
    p = np.expand_dims(p, 0)
  y = np.expand_dims(p[:,0], -1) + np.outer(p[:,1], x) + np.outer(p[:,2], x**2.0)
  return y 

