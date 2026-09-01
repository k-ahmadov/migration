"""Run only the FVM aperture solve (no elastic half-space) and store it."""

from fracinj import paths
from fracinj.io import save_run
from fracinj.types import Parameters
from run_simulation import RateBC, run_fvm


def main() -> None:
    params = Parameters(
        k_n=50e9,
        L=1000.0,
        mu=1e-3,
        w_i=1e-4,
        q_0=1e-5,
        T=10_000.0,
        Nx_p=10_000,
        Nt=20_000,
    )

    fvm = run_fvm(params, bc=RateBC.CONST)
    out_file = paths.results_dir("fvm", "runs") / f"run-L-{params.L:.0f}.hdf5"
    save_run(out_file, t=fvm.t, x_vert=fvm.x, w=fvm.w, p=fvm.p, params=params)
    print(f"Results stored in {out_file}")


if __name__ == "__main__":
    main()
