"""
Module that contains all the functions we need for our project.
"""

# IMPORTS
import numpy as np
import numpy.typing as npt
from scipy import integrate as intg
import matplotlib.pyplot as plt
import pynumdiff as nd # some submodules requrie cvxpy or tqdm
from pynumdiff import smooth_finite_difference as smoothfd
import itertools


# FUNCTIONS
def generate_gaussian_noise(
    std: float, shape: tuple[int, ...]
) -> npt.NDArray[np.float64]:
    """Generate numpy array of Gaussian noise with specified std and array size.

    Parameters
    ----------
    std : float
        Standard deviation of noise.
    shape : tuple[int, ...]
        Size of output array.

    Returns
    -------
    noise : ndarray
        Gaussian noise array, shape given by argument.
    """
    rng = np.random.default_rng()
    noise = rng.normal(scale=std, size=shape)
    return noise
  

def lorenz(t, xyz, *, sigma=10, rho=28, beta=8 / 3):
    """Find time derivative of Lorenz attractor at given coordinates x, y, z.

    Parameters
    ----------
    t : any
        Placeholder for syntax compatibility with scipy.integrate.solve_ivp().
    xyz : array-like, shape (3, n)
        n coordinate vectors.
    sigma : float
        Prandtl number.
    rho : float
        Rayleigh number.
    beta : float
        Parameter related to fluid dimensions.

    Returns
    -------
    xyz_dot : array, shape (3, n)
        Time derivatives of x, y, z at the n coordinate vectors.
    """
    x, y, z = xyz
    x_dot = sigma * (y - x)
    y_dot = x * (rho - z) - y
    z_dot = x * y - beta * z
    xyz_dot = np.array([x_dot, y_dot, z_dot])
    return xyz_dot
  
  
def integrate_ode(x_dot_fun, x_0, t_arr, *, args=None, **options):
  """Wrapper of scipy.integrate.solve_ivp() with simpler call signature and output.

  Parameters
  ----------
  x_dot_fun : function
      Time derivative of x, with call signature f(t, x, *args).
  x_0 : array-like, shape (n,)
      Initial value of x.
  t_arr : array-like, shape (n_points,)
      Array of points at which x is evaluated.
  args : tuple, optional
      Extra arguments for x_dot_fun, by default None.
  **options
      Options passed to scipy.integrate.solve_ivp().

  Returns
  -------
  x : ndarray, shape (n, n_points)
      x evaluated at points specified by t_arr.
  """
  t_span = t_arr[[0, -1]]
  solution = intg.solve_ivp(
      x_dot_fun, t_span, x_0, t_eval=t_arr, args=args, **options
  )
  x = solution.y
  return x


def denoise(x_list, dt, *, filter_order=4, cutoff_freq=0.025, **options):
    """Denoise a list of one-dimensional arrays of data measured at fixed time interval. Then take time derivative of each array.
    
    The arrays in a list are typically the components of a vector.

    Parameters
    ----------
    x_list : ndarray, shape (n, num_steps)
        A list of n one-dimensional arrays containing noisy data.
    dt : float
        Time interval, aka step size.
    filter_order : int, optional
        Order of Butterworth filter, by default 4.
    cutoff_freq : float, optional
        Cutoff frequency of Butterworth filter, by default 0.025.

    Returns
    -------
    x_denoised_list : ndarray, shape (n, num_steps)
        Denoised data.
    x_dot_denoised_list : ndarray, shape (n, num_steps)
        Time derivative of denoised data.
    """    
    options["filter_order"] = filter_order
    options["cutoff_freq"] = cutoff_freq

    x_denoised_list = np.zeros_like(x_list)
    x_dot_denoised_list = np.zeros_like(x_list)
    for n in range(len(x_list)):
        x = x_list[n, :]
        x_denoised, x_dot_denoised = smoothfd.butterdiff(x, dt, **options)
        x_denoised_list[n, :] = x_denoised
        x_dot_denoised_list[n, :] = x_dot_denoised

    return x_denoised_list, x_dot_denoised_list


def library_function(X, n, **kwargs):
    """
    Library function that takes matrix X, and n (the maximum degree of the polynomial features)
    Returns Theta, the library which is  a matrix of shape rows: time steps, cols: number of features in the library
    **kwargs includes print: if True, it will print the shape of Theta and the Theta matrix itself.
    """

    rows_X, cols_X = X.shape
    # listing the polynomial features in library
    variables = np.arange(1, cols_X + 1).astype(str)

    polynomials_list = [
        combo
        for r in range(1, n + 1)
        for combo in itertools.combinations_with_replacement(variables, r)
    ]

    Theta_no_ones = np.zeros((rows_X, len(polynomials_list)))

    for i, poly in enumerate(list(polynomials_list)):
        # Convert strings to 0-based integer indices
        indices = [int(s) - 1 for s in poly]
        # select the columns of X corresponding to the current polynomial feature and compute their product
        Theta_no_ones[:, i] = np.prod(X[:, indices], axis=1)

    Theta = np.concatenate((np.ones((rows_X, 1)), Theta_no_ones), axis=1)

    if "print" in kwargs and kwargs["print"] == True:
        print("Theta shape: ", Theta.shape)
        print("Theta: ", Theta)

    return Theta


def test_x():
    X = np.array([[1, 2, 8], [3, 4, 9], [5, 6, 10]])
    return X


# CLasses
class Trajectory:
    def __init__(self, x_dot_fun, x_0, dt, num_steps, noise_std=0.0):
        self.x_dot_fun = x_dot_fun
        self.x_0 = x_0
        
        self.dt = dt
        self.num_steps = num_steps
        self.t_end = self.dt * self.num_steps
        self.t_arr = np.linspace(0, self.t_end, self.num_steps + 1)
        
        self.x = integrate_ode(self.x_dot_fun, self.x_0, self.t_arr)
        self.x_dot = self.x_dot_fun(None, self.x)
        self.shape = np.shape(self.x)
        
        self.noise_std = noise_std
        if self.noise_std == 0:
            self.x_noisy = self.x
        else:
            self.x_noisy = self.x + generate_gaussian_noise(self.noise_std, self.shape)
            
        self.x_denoised, self.x_dot_denoised = denoise(self.x_noisy, self.dt)


# Runtime info
if __name__ == "__main__":
    pass
