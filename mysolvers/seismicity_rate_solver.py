from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


def make_stressing_rate_interpolant(
    t: NDArray[np.float64], dtau_dt: NDArray[np.float64]
):
    """
    Build a time interpolant for the stressing rate field.

    Parameters
    ----------
    t       : (n_times,) time vector
    dtau_dt : (n_points, n_times) stressing rate field

    Returns
    -------
    callable : dtau_dt_interp(t) → (n_points,)
    """
    return interp1d(
        t,
        dtau_dt,
        axis=1,
        bounds_error=False,
        fill_value=(dtau_dt[:, 0], dtau_dt[:, -1]),
    )


def seismicity_rate_ode(
    t: np.float64,
    R: NDArray[np.float64],
    dtau_dt_interp: Callable[[float], NDArray[np.float64]],
    dtau_dt_0: float,
    t_a: float,
) -> NDArray[np.float64]:
    """
    Dieterich (1994) / Segall & Lu (2015) seismicity rate ODE.

    dR/dt = R/t_a * (dτ/dt / dτ/dt_0 - R)

    Parameters
    ----------
    t              : current time
    R              : (n_points,) seismicity rate (normalized)
    dtau_dt_interp : callable, returns (n_points,) stressing rate at time t
    dtau_dt_0      : background stressing rate [Pa/s] (scalar)
    t_a            : characteristic relaxation time [s] (scalar)
    """
    stressing_ratio = np.maximum(dtau_dt_interp(t), 1e-30) / dtau_dt_0
    return R / t_a * (stressing_ratio - R)


def solve_seismicity_rate(
    t: NDArray[np.float64],
    dtau_dt: NDArray[np.float64],
    dtau_dt_0: float,
    t_a: float,
    rtol: float = 1e-6,
    atol: float = 1e-10,
):
    """
    Solve the seismicity rate ODE for all fault points.

    Parameters
    ----------
    t         : (n_times,) time vector
    dtau_dt   : (n_points, n_times) Coulomb stressing rate [Pa/s]
    dtau_dt_0 : background stressing rate [Pa/s]
    t_a       : characteristic relaxation time [s]
    rtol, atol: solver tolerances

    Returns
    -------
    result : ODE solution object; result.y has shape (n_points, n_times)
    """
    dtau_dt_interp = make_stressing_rate_interpolant(t, dtau_dt)
    n_points = dtau_dt.shape[0]
    R0 = np.ones(n_points)

    result = solve_ivp(
        fun=lambda t, R: seismicity_rate_ode(t, R, dtau_dt_interp, dtau_dt_0, t_a),
        t_span=(t[0], t[-1]),
        y0=R0,
        t_eval=t,
        method="RK45",
        rtol=rtol,
        atol=atol,
    )
    assert result.success, f"ODE solver failed: {result.message}"
    return result
