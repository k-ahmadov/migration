import matplotlib.pyplot as plt
import numpy as np

from mypackages import plotting

# %%

# %%


duration = np.geomspace(1e3, 1e7, 10)

fig = plt.figure(
    figsize=(6.4 / 1.5, 4.8 / 1.5), dpi=150, layout="tight", clear=True, num=1
)
ax = fig.subplots()
for alpha in [0.5, 0.8, 1.0]:
    distance = duration**alpha
    velocity = distance / duration
    ax.plot(duration, velocity, label=rf"From $x_f \propto t^{{ {alpha}}}$")
    exponent, prefactor = np.polyfit(np.log(duration), np.log(velocity), deg=1)
    plotting.slope_triangle(
        ax, duration[5], np.exp(prefactor), round(exponent, 2), dx_log=0.8, inverse=True
    )
ax.set(
    xlabel="Duration [s]",
    ylabel="Velocity [m/s]",
    title="Velocity vs Duration",
    xscale="log",
    yscale="log",
)
ax.legend()
plt.show()

# %%
