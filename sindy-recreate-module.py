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
  variables = np.arange(1, cols_X+1).astype(str)
  print("Variables: ", variables) #Debugging

  polynomials_list = [combo for r in range(1, n + 1) for combo in itertools.combinations_with_replacement(variables, r)]
  print("Polynomials list: ", list(polynomials_list)) #Debugging

  Theta_no_ones = np.zeros((rows_X, len(polynomials_list)))
  #print("Theta_no_ones shape: ", Theta_no_ones.shape) #Debugging

  for i, poly in enumerate(list(polynomials_list)):
      #Convert strings to 0-based integer indices
      indices = [int(s) - 1 for s in poly]
      ##print(f"Processing polynomial: {poly}, indices: {indices}") #Debugging
      #select the columns of X corresponding to the current polynomial feature and compute their product
      Theta_no_ones[:, i] = np.prod(X[:, indices], axis=1) 
  
  ##print("Theta_no_ones shape: ", Theta_no_ones.shape) #Debugging

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
    testXx = test_x()
    print("Test X: ", testXx)
    library_function(testXx, 3, print=True)

