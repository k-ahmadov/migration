"""Run the FVM aperture solve + elastic half-space solve and store the result.

python scripts/run_simulation.py            # single run, constant rate

Set ``SHUT_IN`` in ``main`` to stop injection at that fraction of t_c.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fracinj import paths, physics
from fracinj.io import save_run
from fracinj.solvers import aperture, elastic
from fracinj.types import Field, Parameters, Vector


@dataclass(frozen=True)
class FVMResult:
    x: Vector
    t: Vector
    p: Field
    w: Field


@dataclass(frozen=True)
class ElasticResult:
    x: Vector
    t: Vector
    sn: Field


def run_fvm(params: Parameters) -> FVMResult:
    nx, nt = int(params.Nx_p), int(params.Nt)
    if nx <= 0 or nt <= 0 or params.T < 0:
        raise ValueError("need Nx_p > 0, Nt > 0, T >= 0")

    scales = physics.cubic_scales(params, pin="length")
    w_hat = aperture.solve_diffusion(
        num_nodes=nx,
        num_steps=nt,
        w_initial=params.w_i / scales.w_c,
        t_final=params.T / scales.t_c,
        k_func=lambda w: w**3,
        left_bc_constant_rate=physics.is_constant_rate(params),
        t_stop=params.t_stop / scales.t_c if params.t_stop > 0 else None,
    )
    w = w_hat * scales.w_c
    t = np.linspace(params.T / (nt - 1), params.T, nt - 1)
    if w.shape[0] != t.size:
        raise ValueError("Time array size does not patch solver size")
    return FVMResult(
        x=np.linspace(0, params.L, nx),
        t=t,
        p=params.k_n * (w - params.w_i),
        w=w,
    )


class _InterpRHS:
    """Stateful forcing term for ``elastic.FIE_log_sing`` (pressure on [-1, 1])."""

    def __init__(self, x_mirror: Vector, L: float) -> None:
        self.x_mirror = x_mirror
        self.L = L
        self.y: Vector | None = None

    def __call__(self, s: np.ndarray) -> np.ndarray:
        assert self.y is not None
        return np.interp(s * self.L, self.x_mirror, self.y)


def run_elastic(params: Parameters, x_fvm: Vector, p_tx: Field) -> ElasticResult:
    nt = int(p_tx.shape[0])
    if x_fvm.ndim != 1 or p_tx.ndim != 2 or p_tx.shape[1] != x_fvm.size:
        raise ValueError("expected p_tx of shape (Nt, Nx) with Nx == x_fvm.size")

    E_plane = physics.plane_strain_modulus(params)
    lam = (4.0 / np.pi) * params.k_n * params.L / E_plane
    sn_char = physics.cubic_scales(params, pin="length").sn_c

    x_mirror = np.concatenate((-x_fvm[:0:-1], x_fvm))  # avoid double-counting x=0
    rhs = _InterpRHS(x_mirror, params.L)

    def set_pressure(p_x: np.ndarray) -> None:
        rhs.y = np.concatenate((p_x[:0:-1], p_x)) / sn_char

    set_pressure(p_tx[0])
    x_hat, sn_hat, _ = elastic.FIE_log_sing(lam, rhs, int(params.Nx_sn))
    x_out = x_hat * params.L

    sn_tx = np.empty((nt, sn_hat.size))
    sn_tx[0] = sn_hat * sn_char
    for i in range(1, nt):
        set_pressure(p_tx[i])
        _, sn_hat, _ = elastic.FIE_log_sing(lam, rhs, int(params.Nx_sn))
        sn_tx[i] = sn_hat * sn_char

    t = np.linspace(params.T / nt, params.T, nt)
    keep = x_out >= 0
    return ElasticResult(x=x_out[keep], t=t, sn=sn_tx[:, keep])


def run_simulation(
    params: Parameters, out_filepath: Path, shut_in: float | None = None
) -> None:
    """``shut_in``: stop injecting at ``shut_in * t_c`` (``None``: inject throughout)."""
    t_c = physics.cubic_scales(params, pin="length").t_c
    params.T = t_c * 2
    params.t_stop = shut_in * t_c if shut_in is not None else 0.0
    print(f"Simulation duration: {params.T:.3g} s")
    if params.t_stop > 0:
        print(f"Shut-in at t = {params.t_stop:.3g} s")

    fvm = run_fvm(params)
    print("FVM simulation finished")

    el = run_elastic(params, fvm.x, fvm.p)
    print("Elastic half-space solution finished")

    save_run(
        out_filepath,
        t=fvm.t,
        x_vert=fvm.x,
        w=fvm.w,
        p=fvm.p,
        x_sc=el.x,
        sn=el.sn,
        params=params,
    )
    print(f"Results stored in {out_filepath}")


def sweep(
    params: Parameters,
    field: str,
    values: np.ndarray,
    out_dir: Path,
) -> None:
    """Run one simulation per value of ``params.<field>``.

    ``run_simulation`` recomputes ``params.T`` unconditionally on every call:
    since `params` is shared across iterations, a "only set T if unset" guard
    would freeze T at the first value's duration for the rest of the sweep.
    """
    for value in values:
        setattr(params, field, float(value))
        print(f"running {field}={value:.1e}")
        run_simulation(params, out_dir / f"{field}-{value:.0e}.hdf5")


def default_parameters(bc_param: str) -> Parameters:
    params = Parameters(
        k_n=200e9,
        L=1000.0,
        mu=1e-3,
        w_i=1e-5,
        E=60e9,
        nu=0.25,
        Nx_p=1024,
        Nx_sn=512,
        Nt=2000,
        # T=500.0,
    )
    setattr(params, bc_param, 1e-13)
    return params


SHUT_IN = 0.25  # stop injection at SHUT_IN * t_c (t_c pinned to L); None: never


def main() -> None:
    bc = "m_q"
    params = default_parameters(bc)
    sub = "long" if SHUT_IN is None else f"shut-in-{SHUT_IN:g}tc"
    out_dir = paths.results_dir("halfspace", "ramp", sub)

    # q0_values = np.geomspace(1e-10, 1e-6, 5)
    # sweep(params=params, field="q_0", values=q0_values, out_dir=out_dir)

    # m_q_values = np.geomspace(1e-14, 1e-7, 8)
    # sweep(params=params, field="m_q", values=m_q_values, out_dir=out_dir, bc=bc)

    out_file = out_dir / f"{bc}-{getattr(params, bc):.0e}.hdf5"
    run_simulation(params, out_file, shut_in=SHUT_IN)


if __name__ == "__main__":
    main()
