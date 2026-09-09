import matplotlib.pyplot as plt

# %%

alphas = [
    0.0,
    0.0,
    0.0,
    1 / 3,
    1 / 3,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.7,
    0.7,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
]


alphas_unique = list(dict.fromkeys(alphas))
counts = [alphas.count(alpha) for alpha in alphas_unique]

# %%
fig, ax = plt.subplots(figsize=(6.4 / 2.5, 4.8 / 2), dpi=200)
ax.stem(alphas_unique, counts, linefmt="C7", basefmt="C7")
ax.set(
    xlabel="Scaling exponent, $\\alpha$",
    ylabel="Number of sites",
    # title="title",
    xticks=(0, 0.3, 0.5, 0.7, 1),
)
fig.tight_layout()
plt.show()
