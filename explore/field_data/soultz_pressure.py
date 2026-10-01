import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fracinj import paths, plotting

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
# %%

df = pd.read_csv(
    paths.DATA
    / "SSFS1993-Hydraulic/SSFS1993-Hydraulic_september/SSFS1993-Hydraulic_september.csv",
    sep=";",
    header=0,
    names=[
        "datetime",
        "q_in_bgr (l/s)",
        "q_ex_bgr (l/s)",
        "p_wellhead (bar)",
        "p_tubing (bar)",
        "p_annulus (bar)",
        "temperature (°C)",
        "q_in_csma (l/s)",
        "q_ex_csma (l/s)",
        "q_in_comb_max (l/s)",
        "q_ex_comb_max (l/s)",
    ],
)

df["datetime"] = pd.to_datetime(df["datetime"])

date_mask = (df["datetime"].iloc[0] + pd.Timedelta(hours=0) <= df["datetime"]) & (
    df["datetime"] <= df["datetime"].iloc[0] + pd.Timedelta(days=15)
)

df = df.loc[date_mask]

t = (df["datetime"] - df["datetime"].iloc[0]).dt.total_seconds()
t_days = t / 86_400
p = df["p_wellhead (bar)"]
Q = df["q_in_csma (l/s)"]


# %%

style = {
    "marker": "o",
    "ms": 3,
    "mew": 0.5,
    "mfc": "none",
    "ls": "none",
}

CW, GR = 3.4, 1.618
fig = plt.figure(
    figsize=(2 * CW, CW / GR), dpi=150, layout="constrained", clear=True, num=1
)
ax = fig.subplots()
ax.plot(t / 86_400, Q, color=colors[0], label="Flow rate", **style)
ax_twinx = ax.twinx()
ax_twinx.plot(t / 86_400, p, color=colors[1], label="Wellhead pressure", **style)
ax.set(
    xlabel="Time (days)",
    ylabel="Injection rate (l/s)",
    title = "Soultz 1993",
)
ax_twinx.set(ylabel="Wellhead pressure (bars)")
fig.legend(loc="outside upper center", ncols=2)
fig.canvas.draw_idle()
# fig.savefig("/home/kahmadov/phd/migration/figures/field-data/soultz-hydraulic.eps")
plt.pause(0.01)

# %%
CW, GR = 3.4, 1.618
fig = plt.figure(
    figsize=(2 * CW, CW / GR), dpi=200, layout="constrained", clear=True, num=1
)
ax = fig.subplots()
ax.plot(t_days[35:], p[35:], color=colors[1], label="Wellhead pressure", **style)
ax.set(
    xlabel="Time (days)",
    ylabel="Wellhead pressure (bars)",
    title="Evolution of wellhead pressure in log scale",
    xscale="log",
    yscale="log",
)
ax.legend()
plotting.click_slope(ax)
fig.canvas.draw_idle()
plt.pause(0.01)


# %%


colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

m_Q, b_Q = np.polyfit(t, Q, deg=1)
print(f"m_Q = {m_Q:.3g} l/s^2  =  {m_Q / 1000:.3g} m^3/s^2")

fig = plt.figure(
    figsize=(2 * CW, CW / GR), dpi=150, layout="constrained", clear=True, num=2
)
ax = fig.subplots()
ax.plot(t_days, Q, color=colors[0], label="Flow rate", **style)
ax.plot(
    [t_days.iloc[0], t_days.iloc[-1]],
    [m_Q * t.iloc[0] + b_Q, m_Q * t.iloc[-1] + b_Q],
    "k-",
    # lw=1.5,
    label=rf"Linear fit, $m_Q={m_Q:.2g}\,\mathrm{{l/s^2}}$",
)
ax.set(xlabel="Time (days)", ylabel="Injection rate (l/s)", title="Soultz 1993")
ax.legend()
fig.canvas.draw_idle()
# fig.savefig("/home/kahmadov/phd/migration/figures/field-data/soultz-rate.eps")
plt.pause(0.01)

