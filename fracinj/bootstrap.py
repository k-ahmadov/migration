"""Bootstrap uncertainty of a quantity fitted to a seismicity percentile envelope.

The envelope points all come from one catalogue, so they are not independent
and a least-squares covariance underestimates the error. Instead: resample the
events with replacement (keeping time order), optionally shift where the
first bin starts (the bin boundaries are arbitrary), rebuild the envelope,
refit, and take the spread of the fitted quantity.

Example (``explore/basel/basel_crossover.py``)::

    from fracinj import bootstrap

    def t_c_of(t_env, r_env):
        a, b = fit_front(t_env, r_env)
        return physical_t_c(a, b) if a > 0 and b > 0 else np.nan

    samples = bootstrap.envelope_bootstrap(t_inj, r_inj, t_c_of,
                                           bin_size=50, percentile=75)
    print(bootstrap.summary(samples))
"""

from collections.abc import Callable

import numpy as np

from fracinj import detection
from fracinj.types import Vector


def shifted_envelope(
    t: Vector, r: Vector, *, bin_size: int, percentile: float, offset: int = 0
) -> tuple[Vector, Vector]:
    """Percentile envelope with the bin boundaries shifted by ``offset`` events.

    The first bin absorbs the shift (``bin_size + offset`` events), so no early
    event is dropped; the later bins hold ``bin_size`` events each.
    """
    n0 = bin_size + offset
    t0, r0 = detection.percentile_envelope(t[:n0], r[:n0], bin_size=n0,
                                           percentile=percentile)
    t1, r1 = detection.percentile_envelope(t[n0:], r[n0:], bin_size=bin_size,
                                           percentile=percentile)
    return np.concatenate([t0, t1]), np.concatenate([r0, r1])


def envelope_bootstrap(
    t: Vector,
    r: Vector,
    statistic: Callable[[Vector, Vector], float],
    *,
    bin_size: int,
    percentile: float,
    n_boot: int = 500,
    random_offset: bool = True,
    seed: int = 0,
) -> Vector:
    """Bootstrap samples of ``statistic(t_env, r_env)``.

    ``t``, ``r`` are the event times and distances, sorted in time.
    ``statistic`` may return NaN for a failed fit; those samples are dropped.
    """
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        i = np.sort(rng.integers(0, len(t), len(t)))  # keep time order
        offset = int(rng.integers(bin_size)) if random_offset else 0
        env = shifted_envelope(t[i], r[i], bin_size=bin_size,
                               percentile=percentile, offset=offset)
        samples.append(statistic(*env))
    samples = np.asarray(samples, dtype=float)
    return samples[np.isfinite(samples)]


def summary(samples: Vector) -> str:
    """Median with 68% and 95% ranges."""
    q16, q50, q84, q025, q975 = np.percentile(samples, [16, 50, 84, 2.5, 97.5])
    return (f"median {q50:.3g}, 68%: {q16:.3g}-{q84:.3g}, "
            f"95%: {q025:.3g}-{q975:.3g} ({samples.size} samples)")
