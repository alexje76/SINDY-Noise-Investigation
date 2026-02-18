"""
Module that contains all the functions we need for our project.
"""
#IMPORTS
import numpy as np

#FUNCTIONS
def placeholder_function():
  pass

def library_function(X, n, **kwargs):
  """
  Library function that takes matrix X, and n (the maximum degree of the polynomial features)
  Returns Theta, the library which is  a matrix of shape rows: time steps, cols: number of features in the library
  **kwargs includes print: if True, it will print the shape of Theta and the Theta matrix itself.
  """
  import itertools

  rows_X, cols_X = X.shape
  # listing the polynomial features in library
  variables = np.arange(1, cols_X+1).astype(str)

  polynomials_list = [combo for r in range(1, n + 1) for combo in itertools.combinations_with_replacement(variables, r)]

  Theta_no_ones = np.zeros((rows_X, len(polynomials_list)))

  for i, poly in enumerate(list(polynomials_list)):
      #Convert strings to 0-based integer indices
      indices = [int(s) - 1 for s in poly]
      #select the columns of X corresponding to the current polynomial feature and compute their product
      Theta_no_ones[:, i] = np.prod(X[:, indices], axis=1) 

  Theta = np.concatenate((np.ones((rows_X, 1)), Theta_no_ones), axis=1)

  if 'print' in kwargs and kwargs['print']==True:
    print("Theta shape: ", Theta.shape)
    print("Theta: ", Theta)

  return Theta

def test_x():
  X = np.array([[1, 2,8], [3, 4,9], [5, 6,10]])
  return X
#CLasses
class PlaceholderClass:
  pass

#Runtime info
if __name__ == '__main__': 
  pass

