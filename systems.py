from typing import TypeAlias, Union, ParamSpec, Callable, Concatenate, Any, Protocol

import numpy as np
import numpy.typing as npt
from scipy import integrate as intg
import pandas as pd


FloatArr: TypeAlias = npt.NDArray[np.floating]
FloatData: TypeAlias = Union[FloatArr, pd.DataFrame, pd.Series]

P = ParamSpec("P")
DiffFun: TypeAlias = Callable[
    Concatenate[float | None, FloatArr, P],
    FloatArr,
]

class HasShape(Protocol):
    """Protocol for array-like objects with properties ``ndim`` and ``shape``.

    Examples of objects satisfying this protocal include ``numpy.ndarray`` and
    ``pandas.DataFrame``.
    """

    @property
    def ndim(self) -> int: ...
    @property
    def shape(self) -> tuple[int, ...]: ...


def validate_leading_dim(arr: HasShape, dim: int) -> None:
    """Validate that the leading dimension of an array matches an expected size.

    Parameters
    ----------
    arr : HasShape
        Array-like object with ``ndim`` and ``shape`` properties.
    dim : int
        Expected size of the leading dimension.

    Raises
    ------
    ValueError
        If ``arr.ndim > 2`` or ``arr.shape[0] != dim``.
    """
    if not (arr.ndim <= 2 and arr.shape[0] == dim):
        raise ValueError(
            f"Expected object shape to be ({dim},) or ({dim}, n), "
            f"but it is {arr.shape}."
        )

    
class Hopf:
    """Hopf normal form in Cartesian coordinates.

    .. math::

        \\dot x &= \\mu x + y - Ax (x^2 + y^2), \\\\
        \\dot y &= -x + \\mu y - Ay (x^2 + y^2).

    We specify :math:`A > 0`. The bifurcation occurs when :math:`\\mu > 0`, in which
    case an attractor exists at :math:`r_0 = \\sqrt{\\mu / A}`.

    Parameters
    ----------
    bif_param : float, optional
        Bifurcation parameter :math:`\\mu`. Default is ``1``.
    lyapunov : float, optional
        First Lyapunov coefficient :math:`A`. Default is ``1``.

    Attributes
    ----------
    bif_param : float
        Bifurcation parameter :math:`\\mu`.
    lyapunov : float
        First Lyapunov coefficient :math:`A`.
    """

    def __init__(self, bif_param: float = 1, lyapunov: float = 1):
        self.bif_param: float = bif_param
        self.lyapunov: float = lyapunov

    def __repr__(self) -> str:
        return (
            f"Hopf object {hex(id(self))}\n"
            f"    Bifurcation parameter,       mu:  {self.bif_param}\n"
            f"    First Lyapunov coefficient,  A:   {self.lyapunov}"
        )

    @property
    def params(self) -> tuple[float, float]:
        """Return ``(bif_param, lyapunov)`` as a tuple.

        Returns
        -------
        tuple[float, float]
            The bifurcation parameter and the first Lyapunov coefficient.
        """
        return (self.bif_param, self.lyapunov)

    def params_array(self, lib: FloatData) -> FloatArr:
        """Construct the ground-truth SINDy coefficient matrix :math:`\\Xi`.
        
        The nonzero entries encode

        .. math::

            \\dot x &= \\mu x + y - Ax (x^2 + y^2), \\\\
            \\dot y &= -x + \\mu y - Ay (x^2 + y^2), \\\\
            \\dot\\mu &= 0.

        The columns of the SINDy feature library should be defined as

        .. math::

            1,\\; x,\\; y,\\; \\mu,\\; x^2,\\; xy,\\; x\\mu,\\; y^2,\\; y\\mu,\\;
            \\mu^2,\\; x^3,\\; x^2 y,\\; x^2\\mu,\\; xy^2,\\; xy\\mu,\\; x\\mu^2,\\;
            y^3,\\; \\ldots

        Parameters
        ----------
        lib : FloatArr, shape (n_features, n_timesteps)
            The feature library of SINDy, used to determine the required shape of the
            output.  Must satisfy ``n_features >= 17`` and ``n_timesteps >= 3``. 

        Returns
        -------
        arr : FloatArr, shape (n_features, n_timesteps)
            Sparse coefficient matrix :math:`\\Xi`, zero-padded to match the shape of
            ``lib``.

        Raises
        ------
        AssertionError
            If ``lib.shape[0] < 17`` or ``lib.shape[1] < 3``.
        """
        n, m = lib.shape
        assert n >= 17 and m >= 3

        arr = np.zeros((17, 3))
        arr[1, 1] = -1
        arr[2, 0] = 1
        arr[6, 0] = arr[8, 1] = self.bif_param
        arr[10, 0] = arr[11, 1] = arr[13, 0] = arr[16, 1] = -self.lyapunov

        arr = np.pad(arr, ((0, n - 17), (0, m - 3)), mode="constant")

        return arr

    def diff_fun(self, _: Any, xyu: FloatArr) -> FloatArr:
        """Find the Hopf normal form system's time differential.

        Call signature is as expected by :func:`scipy.integrate.solve_ivp`
        and related solvers. The time argument is accepted but unused.

        Parameters
        ----------
        _ : Any
            Placeholder for syntax compatibility.
        xyu : FloatArr, shape (3, n)
            Array containing ``n`` 3D vectors of shape :math:`(x, y, \\mu)`, where
            :math:`\\mu` is bifurcation parameter, a constant.

        Returns
        -------
        xyu_dot : FloatArr, shape (3, n)
            Time derivative :math:`(\\dot x, \\dot y, 0)` at the ``n`` vectors in
            ``xyu``. The third component is always zero since :math:`\\mu` is a constant.

        Raises
        ------
        ValueError
            If ``xy.shape[0] != 2`` or ``xy.ndim > 2``.
        """
        validate_leading_dim(xyu, 3)

        x: FloatArr
        y: FloatArr
        x, y, _ = xyu

        a_times_r_sq: FloatArr = self.lyapunov * (x**2 + y**2)
        xyu_dot: FloatArr = np.zeros_like(xyu)
        xyu_dot[0] = self.bif_param * x + y - x * a_times_r_sq
        xyu_dot[1] = -x + self.bif_param * y - y * a_times_r_sq

        return xyu_dot