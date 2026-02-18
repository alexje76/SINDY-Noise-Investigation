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
  """
  import itertools

  rows_X, cols_X = X.shape

  # listing the polynomial features in library
  variables = np.arange(1, cols_X+1)
  variables = np.concatenate((1,variables)) 

  polynomials_list = itertools.combinations_with_replacement(variables, n)
  print("Polynomials list: ", list(polynomials_list)) #Debugging

  Theta_no_ones = np.zeros((rows_X, len(list(polynomials_list))))

  #for i, poly in enumerate(list(polynomials_list)):
    

  Theta = np.concatenate((np.ones((rows_X, 1)), Theta_no_ones), axis=1)

  if 'print' in kwargs and kwargs['print']==True:
    print("Theta shape: ", Theta.shape)
    print("Theta: ", Theta)

  return Theta

def test_x():
  X = np.array([[1, 2], [3, 4], [5, 6]])
  return X
#CLasses
class PlaceholderClass:
  pass

#Runtime info
if __name__ == '__main__': 
    testXx = test_x()
    library_function(testXx, 2, print=True)

