import matplotlib.pyplot as plt
import numpy as np

from fracinj import io

# %%

t_s, d_s = io.load("soultz")

# %%

n_points = 1000
min_events = 5
t_eval = np.linspace(t_s[1], t_s[-1], n_points)
percentile = 90

p_vals = np.full_like(t_eval, np.nan, dtype=float)
for i in range(n_points - 1):
    mask = (t_eval[i] < t_s) & (t_s <= t_eval[i + 1])
    n = mask.sum()
    if n >= min_events:
        p_vals[i] = np.percentile(d_s[mask], percentile, method='nearest')


# %%

plt.figure()
plt.scatter(t_s, d_s)
plt.scatter(t_eval, p_vals)
plt.xlabel("Time [s]")
plt.ylabel("Distance [m]")
plt.title("Soultz 1993")
plt.tight_layout()
plt.show()


# %%
def seismic_envelope(t_s, d_s, percentile=90, min_events=20, max_dt=None,
                      n_eval=1000, t_eval=None):
    """
    Adaptive time-window envelope.

    For each evaluation time, use a backward-looking window that is
    widened in time until it contains at least `min_events` events
    (or capped at `max_dt`), then take the percentile of distance
    within that window.

    Assumes t_s is sorted ascending.
    """
    t_s = np.asarray(t_s)
    d_s = np.asarray(d_s)

    if t_eval is None:
        t_eval = np.linspace(t_s.min(), t_s.max(), n_eval)

    if max_dt is None:
        max_dt = t_s.max() - t_s.min()  # effectively no cap

    d_env = np.full(len(t_eval), np.nan)
    t_env = np.full(len(t_eval), np.nan)

    # searchsorted indices let us binary-search window edges fast
    for i, t_now in enumerate(t_eval):
        # right edge: last event at or before t_now
        right = np.searchsorted(t_s, t_now, side="right")
        if right == 0:
            continue  # no events yet

        # find left edge: smallest window (in time, capped at max_dt)
        # containing >= min_events events, searching backward
        t_left_cap = t_now - max_dt
        left_cap_idx = np.searchsorted(t_s, t_left_cap, side="left")

        left = right - min_events
        left = max(left, left_cap_idx, 0)

        window_d = d_s[left:right]

        if len(window_d) == 0:
            continue

        d_e = np.percentile(window_d, percentile)
        d_env[i] = d_e
        t_env[i] = t_now  # use evaluation time directly, not argmin matching

    valid = ~np.isnan(d_env)
    return t_env[valid], d_env[valid]

# %%

t_env, d_env = seismic_envelope(t_s, d_s, min_events=200, max_dt=1*86400)

plt.figure()
plt.scatter(t_s / 86400, d_s)
plt.scatter(t_env / 86400, d_env)
plt.xlabel("Time [s]")
plt.ylabel("Distance [m]")
plt.title("Soultz 1993")
plt.tight_layout()
plt.show()
