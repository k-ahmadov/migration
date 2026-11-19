"""Run the FVM aperture solve + elastic half-space solve and store the result.

python scripts/run_simulation.py            # single run, constant rate

Set ``SHUT_IN`` in ``main`` to stop injection at that fraction of t_c.
"""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from fracinj import paths, physics
from fracinj.io import save_run
from fracinj.solvers import aperture, elastic_fft
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


def run_elastic(params: Parameters, x_fvm: Vector, p_tx: Field) -> ElasticResult:
    """Induced normal stress via the FFT elastic-joint solver, on the FVM grid.

    The pressure is mirrored onto [-L, L] (symmetric about the injection point)
    and the half-space problem is solved spectrally, then the x >= 0 half is kept.
    """
    nt = int(p_tx.shape[0])
    if x_fvm.ndim != 1 or p_tx.ndim != 2 or p_tx.shape[1] != x_fvm.size:
        raise ValueError("expected p_tx of shape (Nt, Nx) with Nx == x_fvm.size")

    # eps = L_k / L_p = E' / (4 k_n L), with the fracture half-length as L_p
    eps = physics.plane_strain_modulus(params) / (4.0 * params.k_n * params.L)

    X = np.concatenate((-x_fvm[:0:-1], x_fvm)) / params.L  # avoid double-counting x=0
    n_half = x_fvm.size - 1  # index of x=0 in the mirrored grid

    sn_tx = np.empty_like(p_tx)
    for i in range(nt):
        Pi = np.concatenate((p_tx[i, :0:-1], p_tx[i]))  # Pa; the solve is linear
        Sigma, _ = elastic_fft.solve_elastic_joint(X, Pi, eps)
        sn_tx[i] = Sigma[n_half:]

    t = np.linspace(params.T / nt, params.T, nt)
    return ElasticResult(x=x_fvm.copy(), t=t, sn=sn_tx)


TIP_TOL = 1e-3  # same threshold as elastic_fft's "not decayed" warning


def truncate_at_tip(fvm: FVMResult, tol: float = TIP_TOL) -> FVMResult:
    """Drop every step from the first one where the pressure at x = L exceeds
    ``tol * max(p)``: beyond it the FFT stress solve (infinite fracture,
    zero pressure past +-L) is no longer valid."""
    reached = fvm.p[:, -1] > tol * fvm.p.max(axis=1)
    if not reached.any():
        return fvm
    n_keep = int(np.argmax(reached))
    if n_keep == 0:
        raise ValueError("pressure is at x = L from the first step; increase L")
    print(f"Front reached x = L at t = {fvm.t[n_keep]:.3g} s; truncating")
    return FVMResult(x=fvm.x, t=fvm.t[:n_keep], p=fvm.p[:n_keep], w=fvm.w[:n_keep])


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

    fvm = truncate_at_tip(fvm)
    # record what is actually stored; a copy, so a sweep's shared `params` keeps
    # its requested Nt (t has Nt - 1 entries, see run_fvm)
    saved = replace(params, T=float(fvm.t[-1]), Nt=float(fvm.t.size + 1))
    if params.t_stop > saved.T:
        print(f"warning: run ends at t = {saved.T:.3g} s, before shut-in")

    el = run_elastic(saved, fvm.x, fvm.p)
    print("Elastic half-space solution finished")

    save_run(
        out_filepath,
        t=fvm.t,
        x_vert=fvm.x,
        w=fvm.w,
        p=fvm.p,
        x_sc=el.x,
        sn=el.sn,
        params=saved,
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
        L=100.0,
        mu=1e-3,
        w_i=1e-5,
        E=60e9,
        nu=0.25,
        Nx_p=1024,
        Nx_sn=1024,
        Nt=10000,
        # T=500.0,
    )
    setattr(params, bc_param, 1e-13)
    return params


def main() -> None:
    bc = "m_q"
    params = default_parameters(bc)
    out_dir = paths.results_dir("halfspace", "ramp")

    # q0_values = np.geomspace(1e-10, 1e-6, 5)
    # sweep(params=params, field="q_0", values=q0_values, out_dir=out_dir)

    m_q_values = np.geomspace(1e-14, 1e-7, 8)
    sweep(params=params, field="m_q", values=m_q_values, out_dir=out_dir)

    # out_file = out_dir / f"{bc}-{getattr(params, bc):.0e}.hdf5"
    # run_simulation(params, out_file, shut_in=SHUT_IN)


if __name__ == "__main__":
    main()
