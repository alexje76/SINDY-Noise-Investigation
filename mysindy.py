"""
Module that contains all the functions we need for our project.
"""

# IMPORTS
import itertools
from typing import TypeAlias, Any, Callable, ParamSpec, Concatenate, Union
from collections.abc import Sequence

import numpy as np
import numpy.random as npr
import numpy.typing as npt

from scipy import integrate as intg
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import pynumdiff as nd  # Some submodules requrie cvxpy or tqdm
from pynumdiff import smooth_finite_difference as smoothfd
import pandas as pd


# TYPING
FloatArr: TypeAlias = npt.NDArray[np.floating]
FloatData: TypeAlias = Union[FloatArr, pd.DataFrame, pd.Series]

P = ParamSpec("P")
DiffFun: TypeAlias = Callable[
    Concatenate[float | None, FloatArr, P],
    FloatArr,
]


# FUNCTIONS
def generate_gaussian_noise(
    std: float, shape: tuple[int, ...], seed: int = 571
) -> FloatArr:
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
    rng: npr.Generator = npr.default_rng(seed=seed)
    noise: FloatArr = rng.normal(scale=std, size=shape)
    return noise

def lorenz_params() -> FloatArr:
    """Return the parameters of the Lorenz attractor as a numpy array.

    Returns
    -------
    params : ndarray, shape (3,)
        The parameters of the Lorenz attractor: [sigma, rho, beta].
    """
    sigma: float = 10
    rho: float = 28
    beta: float = 8 / 3
    params: FloatArr = np.array([sigma, rho, beta])
    return params

def lorenz_advance(ics, time_eval, dt): #TODO add docsring and typing
    # Ensure time_chunks matches the expected count (e.g., 21 points for 0.02s)
    num_steps = int(round(time_eval / dt)) + 1
    time_chunks = np.linspace(0, time_eval, num_steps)
    
    # sol.y will be shape (3, 21)
    sol = intg.solve_ivp(
        lorenz,           # Your function
        [0, time_eval],        # Time span
        ics,                   # Initial conditions
        t_eval=time_chunks,    # Specific points to return
        method='RK45'
    )
    
    # Transpose to (21, 3) to match your plotting logic
    return sol.y.T


def lorenz(
    _: Any, xyz: FloatArr, *, sigma: float = 10, rho: float = 28, beta: float = 8 / 3
) -> FloatArr:
    """Find time derivative of Lorenz attractor at given coordinates x, y, z.

    Call signature is as expected by :func:`scipy.integrate.solve_ivp`
    and related solvers. The time argument is accepted but unused.

    Parameters
    ----------
    _ : Any
        Placeholder for syntax compatibility.
    xyz : FloatArr, shape (3, n)
        Array containing ``n`` 3D vectors.
    sigma : float
        Prandtl number.
    rho : float
        Rayleigh number.
    beta : float
        Parameter related to fluid dimensions.

    Returns
    -------
    xyz_dot : array, shape (3, n)
        Time derivative :math:`(\\dot x, \\dot y, \\dot z)` at the ``n`` vectors in
        ``xyz``.
    """
    lor_par = lorenz_params()
    sigma, rho, beta = lor_par[0], lor_par[1], lor_par[2]

    is_3_by_n: bool = xyz.ndim <= 2 and xyz.shape[0] == 3
    if not is_3_by_n:
        raise ValueError(
            f"Expected xyz.shape to be (3,) or (3, n), but it is {xyz.shape}."
        )

    x: FloatArr
    y: FloatArr
    z: FloatArr
    x, y, z = xyz
    x_dot: FloatArr = sigma * (y - x)
    y_dot: FloatArr = x * (rho - z) - y
    z_dot: FloatArr = x * y - beta * z
    xyz_dot: FloatArr = np.array([x_dot, y_dot, z_dot])
    return xyz_dot


def integrate_ode(
    x_dot_fun: DiffFun, x_0: FloatArr, t_arr: FloatArr, *, args: tuple = (), **options
) -> FloatArr:
    """Wrapper of scipy.integrate.solve_ivp() with simpler call signature and output. Only supports real variables.

    Parameters
    ----------
    x_dot_fun : function
        Time derivative of x, with call signature f(t, x, *args).
    x_0 : ndarray, shape (n,)
        Initial value of x.
    t_arr : ndarray, shape (n_points,)
        Array of points at which x is evaluated.
    args : tuple, optional
        Extra arguments for x_dot_fun, by default an empty tuple.
    **options
        Options passed to scipy.integrate.solve_ivp().

    Returns
    -------
    x : ndarray, shape (n, n_points)
        x evaluated at points specified by t_arr.
    """
    t_span: Sequence[float] = t_arr[[0, -1]].tolist()
    solution = intg.solve_ivp(
        x_dot_fun, t_span, x_0, t_eval=t_arr, args=args, **options
    )
    x: FloatArr = solution.y.real
    return x


def denoise(
    x_list: FloatArr,
    dt: float,
    *,
    filter_order: int = 4,
    butterworth_cutoff: float = 0.025,
    **options,
) -> tuple[FloatArr, FloatArr]:
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
    butterworth_cutoff : float, optional
        Cutoff frequency of Butterworth filter, by default 0.025.

    Returns
    -------
    x_denoised_list : ndarray, shape (n, num_steps)
        Denoised data.
    x_dot_denoised_list : ndarray, shape (n, num_steps)
        Time derivative of denoised data.
    """
    options["filter_order"] = filter_order
    options["cutoff_freq"] = butterworth_cutoff

    x_denoised_list: FloatArr
    x_dot_denoised_list: FloatArr
    x_denoised_list, x_dot_denoised_list = map(
        np.array, zip(*[smoothfd.butterdiff(x, dt, **options) for x in x_list])
    )

    return x_denoised_list, x_dot_denoised_list


def library_function(
    X: FloatData,
    n: int,
    *,
    df: bool = True,
    Print: bool = False,
    **options,
) -> FloatData:
    """
    Construct a library of polynomial features up to degree n from input data X. optionally print the library and its shape.
    If the input data is a pandas DataFrame, the output will also be a DataFrame with appropriate column names. Otherwise, the output will be a numpy array.

    Parameters
    ----------
    X : numpy array or pandas DataFrame, shape (m, d)
        The input data, where m is the number of samples and d is the number of features.
    n : int
        The maximum degree of the polynomial features in the library.
        For example, if n=2, the library will include all monomials of degree 0, 1, and 2

    **options
        df : bool, optional
            If True, will add headers to the printed Theta. By default True.

        Print : bool, optional
            If True, will print Theta and its shape. By default False.

    Returns
    -------
    Theta: numpy array or pandas DataFrame, shape (m, num_features)
        The library
    """
    if df and not isinstance(X, pd.DataFrame):
        raise ValueError("If df is True, X must be a pandas DataFrame.")
    elif df:
        # Get column names as the variables
        variables = X.columns

        polynomials_list = [
            combo
            for r in range(1, n + 1)
            for combo in itertools.combinations_with_replacement(variables, r)
        ]

        Theta_no_ones = pd.DataFrame(index=X.index)
        for combo in polynomials_list:
            # Create a name like 'Col1*Col2' or 'Col1^2'
            name = "*".join(combo)
            # Multiply the selected columns row-wise
            Theta_no_ones[name] = X[list(combo)].prod(axis=1)

        # Add the intercept (ones) column at the start
        Theta = Theta_no_ones.copy()
        Theta.insert(0, "1", 1.0)
    else:  # The numpy array case

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

    if Print:
        print("Theta shape: ", Theta.shape)
        print("Theta: ", Theta)

    return Theta
def stls(
    lib: FloatData,
    x_dot: FloatArr,
    threshold: float,
    *,
    max_iter: int = 100
) -> FloatArr:
    m, ell = lib.shape
    n = x_dot.shape[1]

    # Initial guess: Least-squares
    sol = np.linalg.lstsq(lib, x_dot, rcond=None)[0]
    mask = np.ones((ell, n), dtype=bool)

    iter: int = 0
    for iter in range(max_iter):
        # Find small coefficients
        mask_new = np.abs(sol) >= threshold
        # Break if mask does not change between iterations
        if np.array_equal(mask_new, mask):
            # print(f"threshold: {threshold}")
            # print()
            # print(sol)
            # print()
            # print(mask_new)
            break
        mask = mask_new

        sol[~mask] = 0  # and threshold
        for k in range(n):
            nonzero = mask[:, k]

            # Skip update if all entries of sol are small
            if not np.any(nonzero):
                continue

            sub_lib = lib[:, nonzero]
            # If underdetermined, lstsq will use dense minimum-norm solution, so skip
            if np.linalg.matrix_rank(sub_lib) < nonzero.sum():
                continue

            # Regress onto remaining terms
            sol[nonzero, k] = np.linalg.lstsq(sub_lib, x_dot[:, k], rcond=None)[0]

    print(f"Exited at {iter} iterations.")

    return sol

def lorenz_array(
    Xi: FloatData,
) -> FloatData:
    """
    Return the Lorenz attractor array with the same shape as X, for MSE calculation.

    Parameters
    ----------
    Xi : numpy array or pandas DataFrame, shape (m, d)
        Given array to match size wise for MSE calculation, Xi

    Returns
    -------
    FloatData
        The Lorenz attractor array wih the same shape as Xi, for MSE calculation.

        #TODO: Perhpaps we want to adjust this to be transposed - but only if it would then match our custom sindy.
    """

    n, m = Xi.shape

    # x_dot, y_dot, z_dot
    lor_par = lorenz_params()
    sigma, rho, beta = lor_par[0], lor_par[1], lor_par[2]
    
    classic_lorenz = np.array(
        [
            [0, 0, 0],
            [-sigma, rho, 0],
            [sigma, -1, 0],
            [0, 0, -beta],
            [0, 0, 0],
            [0, 0, 1],
            [0, -1, 0],
        ]
    )
    padded_lorenz = np.pad(classic_lorenz, ((0, n - 7), (0, m - 3)), mode="constant")
    return padded_lorenz


def test_x():
    """
    Simply returns a 3 by 3 array for testing purposes.

    Returns
    -------
    X: np.ndarray , shape (3, 3)
        Hardcoded array for testing purposes.
    """
    X = np.array([[1, 2, 8], [3, 4, 9], [5, 6, 10]])
    return X


def test_x_df():
    """
    Returns a 3 by 3 pandas DataFrame for testing purposes.

    Returns
    -------
    X_df: pd.DataFrame
        hardcoded DataFrame with columns 'x1', 'x2', 'x3'.
    """
    X = np.array([[1, 2, 8], [3, 4, 9], [5, 6, 10]])

    # Convert to DataFrame to trigger the 'if df' branch in your logic
    X_df = pd.DataFrame(X, columns=["x1", "x2", "x3"])

    return X_df


# Runtime info
if __name__ == "__main__":
    pass
