"""Run the FVM aperture solve + elastic half-space solve and store the result.

python scripts/run_simulation.py            # settings in ``main``

Two geometries, chosen by ``params.radial``:

- plane strain (``radial = 0``): line source at x = 0, ``q_0`` / ``m_q`` per unit
  length (m^2/s, m^2/s^2); FFT stress solve on the mirrored domain [-L, L].
- axisymmetric (``radial = 1``): point source at r = 0, ``q_0`` / ``m_q`` total
  volume rate (m^3/s, m^3/s^2); Hankel-transform stress solve on [0, L].

Pass ``shut_in`` to ``run_simulation`` to stop injection at that fraction of t_c.
"""

from dataclasses import dataclass, replace
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


def run_fvm(params: Parameters) -> FVMResult:
    nx, nt = int(params.Nx_p), int(params.Nt)
    if nx <= 1 or nt <= 1 or params.T <= 0:
        raise ValueError("need Nx_p > 1, Nt > 1, T > 0")

    scales = physics.cubic_scales(params, pin="length")
    solve = aperture.solve_radial_diffusion if physics.is_radial(params) else aperture.solve_diffusion
    w_hat = solve(
        num_nodes=nx,
        num_steps=nt,
        w_initial=params.w_i / scales.w_c,
        t_final=params.T / scales.t_c,
        k_func=lambda w: w**3,
        left_bc_constant_rate=physics.is_constant_rate(params),
        t_stop=params.t_stop / scales.t_c if params.t_stop > 0 else None,
    )
    w = w_hat * scales.w_c
    # the solvers drop the initial condition: rows are t = dt, 2 dt, ..., T
    t = np.linspace(params.T / (nt - 1), params.T, nt - 1)
    if w.shape[0] != t.size:
        raise ValueError("time array size does not match solver output")
    return FVMResult(
        x=np.linspace(0, params.L, nx),
        t=t,
        p=params.k_n * (w - params.w_i),
        w=w,
    )


def run_elastic(params: Parameters, x: Vector, p_tx: Field) -> Field:
    """Induced normal stress change on the FVM grid, shape ``(n_t, n_x)``, in Pa.

    The solves are linear, so the pressure goes in dimensional (Pa) and the
    stress comes out in Pa; only the lengths are scaled, by L. Plane strain
    mirrors the pressure onto [-L, L] (symmetric about the injection point)
    and keeps the x >= 0 half; the axisymmetric solve works on r in [0, L].
    """
    if x.ndim != 1 or p_tx.ndim != 2 or p_tx.shape[1] != x.size:
        raise ValueError("expected p_tx of shape (Nt, Nx) with Nx == x.size")

    # eps = L_k / L_p = E' / (4 k_n L), with the fracture (half-)length as L_p
    eps = physics.plane_strain_modulus(params) / (4.0 * params.k_n * params.L)

    if physics.is_radial(params):
        sn, _ = elastic.solve_elastic_joint_axisym(x / params.L, p_tx, eps)
        return sn

    X = np.concatenate((-x[:0:-1], x)) / params.L  # avoid double-counting x=0
    Pi = np.concatenate((p_tx[:, :0:-1], p_tx), axis=1)
    sn, _ = elastic.solve_elastic_joint(X, Pi, eps)
    return sn[:, x.size - 1 :]  # x >= 0 half


TIP_TOL = elastic.DECAY_TOL  # the stress solvers' "not decayed" threshold


def truncate_at_tip(fvm: FVMResult, tol: float = TIP_TOL) -> FVMResult:
    """Drop every step from the first one where the pressure at x = L exceeds
    ``tol * max(p)``: beyond it the stress solve (unbounded fracture, zero
    pressure past L) is no longer valid."""
    reached = np.abs(fvm.p[:, -1]) > tol * np.abs(fvm.p).max(axis=1)
    if not reached.any():
        return fvm
    n_keep = int(np.argmax(reached))
    if n_keep == 0:
        raise ValueError("pressure is at x = L from the first step; increase L")
    print(f"Front reached x = L at t = {fvm.t[n_keep]:.3g} s; truncating")
    return FVMResult(x=fvm.x, t=fvm.t[:n_keep], p=fvm.p[:n_keep], w=fvm.w[:n_keep])


def run_simulation(
    params: Parameters,
    out_filepath: Path,
    shut_in: float | None = None,
    duration: float = 2.0,
) -> None:
    """Simulate ``duration * t_c`` and save to ``out_filepath``.

    ``shut_in``: stop injecting at ``shut_in * t_c`` (``None``: inject throughout).
    ``params`` is not modified; the saved copy records what was actually run.
    """
    t_c = physics.cubic_scales(params, pin="length").t_c
    params = replace(
        params,
        T=duration * t_c,
        t_stop=shut_in * t_c if shut_in is not None else 0.0,
    )
    geometry = "axisymmetric" if physics.is_radial(params) else "plane strain"
    print(f"{geometry}: simulation duration {params.T:.3g} s (t_c = {t_c:.3g} s)")
    if params.t_stop > 0:
        print(f"Shut-in at t = {params.t_stop:.3g} s")

    fvm = run_fvm(params)
    print("FVM simulation finished")

    fvm = truncate_at_tip(fvm)
    # record what is actually stored (t has Nt - 1 entries, see run_fvm)
    saved = replace(params, T=float(fvm.t[-1]), Nt=float(fvm.t.size + 1))
    if params.t_stop > saved.T:
        print(f"warning: run ends at t = {saved.T:.3g} s, before shut-in")

    sn = run_elastic(saved, fvm.x, fvm.p)
    print("Elastic half-space solution finished")

    save_run(
        out_filepath,
        t=fvm.t,
        x_vert=fvm.x,
        w=fvm.w,
        p=fvm.p,
        x_sc=fvm.x,
        sn=sn,
        params=saved,
    )
    print(f"Results stored in {out_filepath}")


def sweep(
    params: Parameters,
    field: str,
    values: np.ndarray,
    out_dir: Path,
    **kwargs,
) -> None:
    """Run one simulation per value of ``params.<field>`` (``params`` is not modified)."""
    for value in values:
        run_params = replace(params, **{field: float(value)})
        print(f"running {field}={value:.1e}")
        run_simulation(run_params, out_dir / f"{field}-{value:.0e}.hdf5", **kwargs)


def default_parameters(bc_param: str, value: float, radial: bool = False) -> Parameters:
    params = Parameters(
        k_n=200e9,
        L=3000.0,
        mu=1e-3,
        w_i=1e-5,
        E=60e9,
        nu=0.25,
        Nx_p=2**12,
        Nx_sn=2**12,
        Nt=20000,
        radial=1.0 if radial else 0.0,
    )
    setattr(params, bc_param, value)
    return params


def main() -> None:
    bc = "m_q"
    radial = True
    params = default_parameters(bc, 1e-10, radial=radial)
    out_dir = paths.results_dir("halfspace", "radial", "ramp")

    run_simulation(params, out_dir / f"{bc}-{getattr(params, bc):.0e}.hdf5")

    # sweep(params, "q_0", np.geomspace(1e-10, 1e-6, 5), out_dir)
    # sweep(params, "m_q", np.geomspace(1e-14, 1e-7, 8), out_dir)
    # run_simulation(params, out_dir / "shut-in.hdf5", shut_in=0.5)


if __name__ == "__main__":
    main()
