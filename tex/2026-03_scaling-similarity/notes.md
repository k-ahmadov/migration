# Introduction

This work studies how a fracture aperture (or fluid pressure) propagates
along a deformable fracture during fluid injection. The main goal is to
determine how the aperture front position depends on time, and how this
relationship varies with injection conditions.

Three injection cases are considered: constant pressure, constant flow
rate, and a flow rate that increases linearly with time. The fracture is
assumed to be semi-infinite. The role of initial conditions is also
examined by considering both zero and non-zero initial aperture.

The aperture is governed by the nonlinear diffusion equation
([@MurphyEtAl2004]): $$\begin{equation}
    M \frac{\partial}{\partial x} \left( w^3 \frac{\partial w}{\partial x} \right) = \frac{\partial w}{\partial t},
    \label{eq:nonlinear-diffusion-equation}
\end{equation}$$ where $w$ is the aperture, $x$ is distance, and $t$ is
time. The parameter $M = k_n / (12\mu)$, where $k_n$ is stiffness and
$\mu$ is fluid viscosity.

The aperture front position is expected to follow a power-law in time:
$$\begin{equation}
    x_f \propto t^{\alpha}
    \label{eq:power-law}
\end{equation}$$ where $x_f$ is the front position and $\alpha$ is the
exponent of time.

Self-similar solutions provide a direct link between space and time in
nonlinear diffusion problems and arise under certain simplifying
conditions ([@Barenblatt1996]). Here, self-similarity is used to
determine how the front position scales with time, i.e., to find the
exponent $\alpha$, and how it depends on the injection condition.

# Self-similarity

A time-dependent phenomenon (e.g., aperture, pressure, temperature) is
**self-similar** if its spatial distribution changes with time while
retaining the same shape ([@Barenblatt1996]). In this case, there exist
time-dependent scales such that, when expressed in these scales, the
solution becomes time-independent. As a result, the spatial distribution
at different times collapses onto a single curve when written in
self-similar variables.

Self-similar solutions typically arise in idealized problems with simple
geometry (e.g., infinite or semi-infinite domains) and simple initial
and boundary conditions (e.g., an initially unperturbed state). For such
problems, finding a self-similar solution provides a complete
description of the system.

To construct a self-similar solution, one must first identify
appropriate similarity variables. These are obtained from
transformations that leave the governing partial differential equation
invariant. For the nonlinear diffusion equation
(Eq. [\[eq:nonlinear-diffusion-equation\]](#eq:nonlinear-diffusion-equation){reference-type="ref"
reference="eq:nonlinear-diffusion-equation"}), scaling (or dilation)
transformations are particularly relevant
([@PakdemirliYurusoy1998; @BlumanAnco2002]).

## Scaling transformation

To find similarity variables, we seek a transformation that leaves the
governing PDE
(Eq. [\[eq:nonlinear-diffusion-equation\]](#eq:nonlinear-diffusion-equation){reference-type="ref"
reference="eq:nonlinear-diffusion-equation"}) unchanged. We consider the
following **scaling (or dilation) transformation**: $$\begin{equation*}
    \tilde{x} = \varepsilon^a x, \quad
    \tilde{t} = \varepsilon^b t, \quad
    \tilde{w} = \varepsilon^c w,
\end{equation*}$$ where $\varepsilon > 0$ is an arbitrary parameter.
Substituting these scaled variables into the PDE allows us to determine
the exponents $a$, $b$, and $c$ by requiring that the transformed
equation matches the original.

For convenience, we first rewrite the PDE as: $$\begin{equation*}
    M \Big(3 w^2 (w_x)^2 + w^3 w_{xx}\Big) = w_t,
\end{equation*}$$ where $w_x = \partial w / \partial x$ and
$w_{xx} = \partial^2 w / \partial x^2$.

Substituting the scaled variables gives: $$\begin{equation*}
    \varepsilon^{2a - 4c} M \Big[ 3 \tilde{w}^2 (\tilde{w}_{\tilde{x}})^2 + \tilde{w}^3 \tilde{w}_{{\tilde{x}\tilde{x}}} \Big] = \varepsilon^{b - c} \tilde{w}_{\tilde{t}}.
\end{equation*}$$ For the PDE to be **invariant under scaling**, the
powers of $\varepsilon$ must match, which leads to: $$\begin{equation}
    2a - 4c = b - c.
    \label{eq:equality-for-invariance}
\end{equation}$$ This condition ensures that the scaled PDE has the same
form as the original.

## Similarity solution

The similarity solution of the nonlinear diffusion equation can be
written as $$\begin{equation*}
    w(x, t) = t^{c / b} \, \theta\big(x / t^{a / b}\big),
\end{equation*}$$ where $\zeta = x / t^{a / b}$ and
$\theta(\zeta) = w(x,t)/ t^{c / b}$ are the self-similar variables.
Substituting this form into the PDE and using the exponent relation from
Eq. [\[eq:equality-for-invariance\]](#eq:equality-for-invariance){reference-type="ref"
reference="eq:equality-for-invariance"}, the PDE successfully reduces to
an ODE in $\theta(\zeta)$. The exponents $a$, $b$, and $c$ are
determined by boundary and initial conditions.

# Initially closed fracture

For a fracture with zero initial aperture, $$\begin{equation*}
    w(x,0) = 0, \quad x>0, \qquad w(x,t) \to 0, \quad x \to \infty,
\end{equation*}$$ in similarity form, this becomes $$\begin{equation*}
    t^{c / b} \theta(\zeta) \to 0, \quad \zeta \to \infty \quad (\text{as } t \to 0),
\end{equation*}$$ which implies $\theta(\zeta) \to 0$ as
$\zeta \to \infty$ and $c / b \ge 0$. The values of $a$, $b$, and $c$
depend on the injection boundary condition at $x=0$. I consider the
following **injection conditions**:

- **Constant aperture (pressure):** $w(0,t) = w_0$ $$\begin{equation*}
                t^{c / b} \theta(0) = w_0 \implies c/b = 0
  \end{equation*}$$ Using $b = 2a - 3c$
  (Eq. [\[eq:equality-for-invariance\]](#eq:equality-for-invariance){reference-type="ref"
  reference="eq:equality-for-invariance"}) gives $b = 2a$, leading to
  $$\begin{equation*}
                w(x,t) = \theta(\zeta), \quad \zeta = \frac{x}{t^{1/2}}
  \end{equation*}$$

- **Constant injection rate:** $q(0,t) = q_0$ $$\begin{equation*}
                t^{(4c-a)/b} \theta(0)^3 \theta'(0) = q_0 \implies 4c-a = 0
  \end{equation*}$$ Combined with $b = 2a - 3c$, this gives
  $$\begin{equation*}
                c/b = 1/5, \quad a/b = 4/5, \quad w(x,t) = t^{1/5} \theta(\zeta), \quad \zeta = x / t^{4/5}
  \end{equation*}$$

- **Linearly increasing injection rate:** $q(0,t) = q_0t$
  $$\begin{equation*}
                t^{(4c-a)/b} \theta(0)^3 \theta'(0) = q_0t \implies 4c-a = b
  \end{equation*}$$ Combined with $b = 2a - 3c$, this gives
  $$\begin{equation*}
                c/b = 3/5, \quad a/b = 7/5, \quad w(x,t) = t^{3/5} \theta(\zeta), \quad \zeta = x / t^{7/5}
  \end{equation*}$$

# Initially Open Fracture

For a semi-infinite fracture with non-zero initial aperture $w_i$, the
initial condition is $$\begin{equation*}
    w(x,0) = w_i, \quad x>0, \qquad w(x,t) \to 0, \quad x \to \infty.
\end{equation*}$$ In similarity variables, this becomes
$$\begin{equation*}
    t^{c / b} \theta(\zeta) \to w_i, \quad \zeta \to \infty \quad (\text{as } t \to 0).
\end{equation*}$$ If $c / b > 0$, the right-hand side depends on time:
$$\begin{equation*}
    \theta(\zeta \to \infty) \to \frac{w_i}{t^{c / b}},
\end{equation*}$$ which violates the self-similarity assumption.

For constant pressure, $c/b = 0$, so the initial aperture does not
affect the similarity solution. For constant or linearly increasing
injection rates, $c/b > 0$, so the initial aperture prevents a strictly
self-similar solution.

However, at **late times** $t \gg t_c$ (where $t_c$ is a critical time),
$w_i / t^{c/b} \to 0$, and the similarity solutions derived for zero
initial aperture apply. At **early times** $t \ll t_c$, the aperture
increase ($w$) is small compared to $w_i$, so the fracture behaves
almost as rigid, and a linear diffusion solution is appropriate.

The critical time $t_c$ depends on the physical parameters of the
problem. The dimensional analysis can be used to find its dependence on
the parameters of the problem. For constant injection rate condition the
dimensional analysis yields $$\begin{equation*}
    t_c = \frac{k_n w_i^5}{12 \mu q_0^2}
\end{equation*}$$ which shows that, the higher is the applied injection
rate, the smaller is the characteristic time.

# Conclusions

The time exponents $\alpha$ governing aperture propagation derived under
all conditions considered are summarized in
Table [1](#tab:exponents){reference-type="ref"
reference="tab:exponents"}.

+----------------+------------------+----------------+----------------------+
| **Initial      | **Injection      | **Time         | **Exponent           |
| condition**    | condition**      | regime**       | $\alpha$**           |
+:===============+:=================+:===============+:====================:+
| $w_i = 0$      | $\Delta P(t)$    | ---            | $1/2$                |
|                | constant         |                |                      |
|                +------------------+----------------+----------------------+
|                | $q(t)$ constant  | ---            | $4/5$                |
|                +------------------+----------------+----------------------+
|                | $q(t) \propto t$ | ---            | $7/5$                |
+----------------+------------------+----------------+----------------------+
| $w_i \neq 0$   | $\Delta P(t)$    | ---            | $1/2$                |
|                | constant         |                |                      |
|                +------------------+----------------+----------------------+
|                | $q(t)$ constant  | Early          | $1/2$                |
|                |                  +----------------+----------------------+
|                |                  | Intermediate   | $1/2 < \alpha < 4/5$ |
|                |                  +----------------+----------------------+
|                |                  | Late           | $4/5$                |
|                +------------------+----------------+----------------------+
|                | $q(t) \propto t$ | Early          | $1/2$                |
|                |                  +----------------+----------------------+
|                |                  | Intermediate   | $1/2 < \alpha < 7/5$ |
|                |                  +----------------+----------------------+
|                |                  | Late           | $7/5$                |
+----------------+------------------+----------------+----------------------+

: Predicted time exponents $\alpha$ for fracture aperture propagation.
{#tab:exponents}
