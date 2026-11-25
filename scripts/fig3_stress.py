import matplotlib.pyplot as plt
import numpy as np

from fracinj import analysis, io, paths, plotting

HOUR = 3600.0

# %%
# Early time, rigid regime (linear diffusion): front ∝ t^(1/2).
run_rigid = io.read_hdf5(paths.results_dir("halfspace", "ramp") / "m_q-1e-14.hdf5")
res_rigid = analysis.front_rigid_ramp(run_rigid, stress_front=True)
print(f"rigid: alpha_emp = {res_rigid.alpha_emp:.3f}  (theory {res_rigid.alpha_ana})")

# Late time, soft regime (n = 3 nonlinear diffusion, ramp): front ∝ t^(7/5).
# `front_late_ramp` keeps t > t_c and stops before the front reaches L.
run_soft = io.read_hdf5(paths.results_dir("halfspace", "ramp") / "m_q-1e-07.hdf5")
res_soft = analysis.front_late_ramp(run_soft, stress_front=True)
print(f"soft:  alpha_emp = {res_soft.alpha_emp:.3f}  (theory {res_soft.alpha_ana})")

# %%

# Distance-vs-time record section. Each profile is a vertical trace at its own
# time t_k: tension deflects towards later time, compression towards earlier
# time. Each trace is hung from its peak: its most-tensile point sits exactly at
# (t_k, x_front(t_k)), so the front marker and the power law pass through every
# peak. The zero line (sigma_n = 0) therefore sits at t_k - scale * peak_k.
#
# Two amplitude conventions:
# * ``offset_frac=None``: tension clipped at zero, only the tensile lobe drawn;
#   the tallest trace spans ``amp`` time spacings.
# * ``offset_frac`` given (same as explore/ramp_injection/
#   stress_profiles_stacked.py): full profiles including compression, trace
#   spacing equal to ``offset_frac * ptp(tension)``, so neighbours may touch.
# * ``on_peak=True``: full profiles including compression, but scaled on the
#   tensile peak (tallest peak spans ``amp`` time spacings). Use when
#   compression dwarfs the peak (rigid: ~35x): the compression near the well
#   then runs across the earlier traces, and the axes are sized on the tension.


def stack_panel(ax, run, result, *, t_unit, unit, law, n_profiles=10, amp=0.5,
                offset_frac=None, on_peak=False, label_every=2, front_line=False,
                drop_last=0, anchor_frac=None):
    """Stacked normal-stress profiles with the stress front and its power law.

    ``t_unit``: seconds per displayed time unit, ``unit``: its label.
    ``drop_last``: number of latest profiles to leave out.
    ``anchor_frac``: if given, the power law keeps its exponent but its
    prefactor is fitted on the latest ``anchor_frac`` of the front points (up
    to the last profile shown), so it reads as the late-time asymptote.
    Otherwise it is ``result.analytical()`` (prefactor averaged over the window).
    ``front_line``: draw the stress front as a line through the zero lines
    instead of markers.
    """
    x, sn = io.sort_fields(run.x_sc, run.sn)
    # result.t[j] is run.t[j0 + j]: map back to rows of sn
    j0 = int(np.searchsorted(run.t, result.t[0]))
    idx = np.unique(np.linspace(0, len(result.t) - 1, n_profiles + 1).astype(int)[1:])
    idx = idx[: len(idx) - drop_last]
    t_k = result.t[idx] / t_unit
    spacing = np.diff(t_k).mean()
    tension = -sn[j0 + idx]
    if on_peak:
        scale = amp * spacing / tension.max()  # time units per Pa
    elif offset_frac is None:
        tension = np.clip(tension, 0.0, None)
        scale = amp * spacing / tension.max()
    else:
        scale = spacing / (offset_frac * np.ptp(tension))
    t_zero = t_k - scale * tension.max(axis=1)  # where sigma_n = 0 lands
    traces = t_zero[:, None] + scale * tension

    for k, (tk, tz, trace) in enumerate(zip(t_k, t_zero, traces)):
        ax.axvline(tz, color="0.85", lw=0.4, zorder=0)  # sigma_n = 0 for this trace
        ax.plot(trace, x, color="k", lw=0.7, zorder=2)
        if (len(t_k) - 1 - k) % label_every:  # always label the latest
            continue
        # anchored at the zero line, i.e. under the trace's far-field part
        ax.text(tz, -0.01, f"{tk:.0f} {unit}", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize="small")

    # power law, drawn up to the last profile shown
    t_law = result.t[: idx[-1] + 1]
    alpha = result.alpha_ana
    if anchor_frac is None:
        A = result.A_ana
    else:
        n_anchor = max(2, int(anchor_frac * len(t_law)))
        t_a, x_a = t_law[-n_anchor:], result.y[: idx[-1] + 1][-n_anchor:]
        A = float(np.exp(np.mean(np.log(x_a) - alpha * np.log(t_a))))
    x_law = A * t_law**alpha

    if front_line:
        ax.plot(t_k, result.y[idx], color="C0", lw=1.5, zorder=3, label="Stress front")
        ax.plot(t_law / t_unit, x_law, color="C1", lw=1.0, ls="--", zorder=4, label=law)
    else:
        ax.plot(t_law / t_unit, x_law, color="C0", lw=1.0, ls="--", zorder=3, label=law)
        ax.plot(t_k, result.y[idx], "x", color="k", ms=3, zorder=4, label="Stress front")

    # scale bar for the stress amplitude (in time-axis units, so horizontal).
    # Length: the largest 1-2-5 value not exceeding one trace spacing, so it is
    # readable; placed in a right margin past the latest trace, label on top.
    per_spacing = spacing / scale  # Pa spanned by one trace spacing
    mag = 10 ** np.floor(np.log10(per_spacing))
    bar = max(m * mag for m in (1, 2, 5) if m * mag <= per_spacing)
    bar_label = f"{bar / 1e3:g} kPa" if bar >= 1e3 else f"{bar:g} Pa"
    bar_len = scale * bar

    lo, hi = traces.min(), traces.max()
    if on_peak:  # size on the tension; compression is clipped by the axes
        lo = t_zero[0] - spacing
    lo = min(0.0, lo)
    pad = 0.06 * (hi - lo)
    t_bar = hi + pad + bar_len / 2  # bar centre, right of every trace
    ylim = (0, min(1.5 * result.y[-1], run.params.L))
    ax.set(
        xticks=[],
        ylabel="Distance $x$ (m)",
        xlim=(lo - 0.04 * (hi - lo), t_bar + bar_len / 2 + pad),
        ylim=ylim,
    )
    x_bar = 0.80 * ylim[1]
    ax.plot([t_bar - bar_len / 2, t_bar + bar_len / 2], [x_bar] * 2,
            color="k", lw=1, solid_capstyle="butt", marker="|", ms=3, mew=0.8)
    # label starts just left of the bar's left end (offset in points, so both
    # panels shift alike) and runs right, clear of the traces
    ax.annotate(bar_label, (t_bar - bar_len / 2, x_bar + 0.02 * ylim[1]),
                xytext=(-5, 0), textcoords="offset points",
                ha="left", va="bottom", fontsize="small", annotation_clip=False)

    # legend in the title row, so it hides no trace
    ax.legend(fontsize="small", loc="lower right", bbox_to_anchor=(1.0, 1.0),
              ncol=2, frameon=False, borderaxespad=0.0, handlelength=1.5,
              columnspacing=1.0)
    ax.tick_params(axis="y", right=False, which="both")
    ax.text(0.9, -0.01, "Time", transform=ax.transAxes, ha="left", va="top",
            fontsize="small")
    for side in ("right", "top", "bottom"):
        ax.spines[side].set_visible(False)


# %%

CW, GR = 3.25, 1.618
fig = plt.figure(figsize=(2 * CW, 1.2 * CW / GR), dpi=200, layout="constrained",
                 clear=True, num=1)
ax_rigid, ax_soft = fig.subplots(1, 2)

stack_panel(ax_rigid, run_rigid, res_rigid, t_unit=HOUR, unit="h",
            law=r"$\ell \propto t^{1/2}$", on_peak=True, amp=0.5, n_profiles=15)
stack_panel(ax_soft, run_soft, res_soft, t_unit=1.0, unit="s",
            law=r"$\ell \propto t^{7/5}$", n_profiles=15, offset_frac=1,
            label_every=3, front_line=False, drop_last=1, anchor_frac=0.1)


def m_q_label(run):
    """``m_q = a×10^n m²/s²`` from the run's parameters (``10^n`` when a = 1)."""
    exp = int(np.floor(np.log10(run.params.m_q)))
    mant = run.params.m_q / 10**exp
    num = rf"10^{{{exp}}}" if np.isclose(mant, 1) else rf"{mant:g}\times10^{{{exp}}}"
    return rf"$m_q = {num}$ m$^2$/s$^2$"


# titles sit above the legend row
ax_rigid.set_title(f"(a) Early time, rigid, {m_q_label(run_rigid)}", loc="left",
                   fontsize="small", pad=18)
ax_soft.set_title(f"(b) Late time, soft, {m_q_label(run_soft)}", loc="left",
                  fontsize="small", pad=18)

fig.canvas.draw_idle()
plt.pause(0.01)

# %%

plotting.save_figure(fig, "stress-profiles-stacked-rigid-soft", ext='pdf')

# %%

