import numpy as np
from scipy.integrate import solve_ivp  # For solving ordinary differential equations (ODEs)
from scipy.optimize import newton  # For finding roots of equations
from scipy.special import erf  # For the error function

# --- Exact Solutions ---

# Solves the similarity solution for n=0 (linear diffusion)
def solve_exact_similarity_n0(bcRight):
    """
    Solves the exact similarity solution for the case n=0.

    Args:
        bcRight: Boundary condition at the right end (zeta = infinity).

    Returns:
        zeta: Array of zeta values (similarity variable).
        theta: Array of theta values (solution).
    """
    zeta = np.arange(0, 15, 0.01)  # Create zeta values
    theta = (bcRight - 1) * erf(zeta/2) + 1  # Calculate theta using the error function
    return zeta, theta

# Solves the exact dimensional solution for n=0
def solve_exact_dimensional_n0(x, t, w0, wi, kn, mu=1e-3):
    """
    Solves the exact dimensional solution for the case n=0.

    Args:
        x: Array of x values (spatial coordinate).
        t: Time value.
        w0: Aperture at x=0.
        wi: Initial aperture.
        kn: Joint normal stiffness.
        mu: Fluid viscosity.

    Returns:
        w: Array of aperture values.
    """
    D = (kn * wi**3) / (12 * mu)  # Calculate hydraulic diffusivity
    w = w0 * (1 - erf(x/(2*np.sqrt(D*t)))) + wi*erf(x/(2*np.sqrt(D*t)))  # Calculate aperture using the error function
    return w

# --- Numerical Solutions (n > 0) ---

# Solves the similarity solution for n=1
def solve_n1(bcRight):
    """
    Solves the similarity solution for the case n=1 numerically.

    Args:
        bcRight: Boundary condition at the right end (zeta = infinity).

    Returns:
        zeta: Array of zeta values.
        theta: Array of theta values (solution).
    """

    def objective_n1(u1_0):
        """Objective function for root finding (n=1)."""
        sol = solve_ivp(F_n1, zeta_span, [u0, u1_0], t_eval=zeta_eval)  # Solve the ODE
        uf = sol.y[0][-1]  # Get the final value of u0
        return uf - (bcRight)**2  # The objective is to find u1_0 such that uf = bcRight^2

    def F_n1(zeta, u):
        """ODE system for n=1."""
        u0, u1 = u
        base = max(u0, 1e-5)  # Avoid division by zero or negative values
        du0 = u1
        du1 = -zeta*u1/2/np.power(base, 1/2)  # The ODE for n=1
        return [du0, du1]

    u0 = 1  # Initial condition for u0
    R = 8  # Right boundary for zeta
    zeta_span = [0, R]  # Interval for zeta
    zeta_eval = np.linspace(0, R, int(1e2))  # Zeta values for evaluation

    u1_0 = newton(objective_n1, -1)  # Find the initial condition for u1 using Newton's method
    sol_n1 = solve_ivp(F_n1, zeta_span, [u0, u1_0], t_eval=zeta_eval, method='RK45', atol=1e-8, rtol=1e-6)  # Solve the ODE
    return sol_n1.t, np.power(sol_n1.y[0], 1/2)  # Return zeta and theta (theta = u0^(1/2))

# Solves the similarity solution for n=3 (similar structure to solve_n1)
def solve_n3(bcRight):
    """
    Solves the similarity solution for the case n=3 numerically.

    Args:
        bcRight: Boundary condition at the right end (zeta = infinity).

    Returns:
        zeta: Array of zeta values.
        theta: Array of theta values (solution).
    """
    def objective_n3(u1_0):
        sol = solve_ivp(F_n3, zeta_span, [u0, u1_0], t_eval=zeta_eval)
        uf = sol.y[0][-1]
        return uf - (bcRight)**4

    def F_n3(zeta, u):
        u0, u1 = u
        base = max(u0, 1e-5)
        du0 = u1
        du1 = -zeta*u1/2/np.power(base, 3/4)
        return [du0, du1]

    u0 = 1
    R = 5
    zeta_span = [0, R]
    zeta_eval = np.linspace(0, R, int(1e3))  # Increased resolution for n=3

    u1_0 = newton(objective_n3, -1)
    sol_n3 = solve_ivp(F_n3, zeta_span, [u0, u1_0], t_eval=zeta_eval, method='RK45', atol=1e-8, rtol=1e-6)
    return sol_n3.t, np.power(sol_n3.y[0], 1/4)  # theta = u0^(1/4)


""" The 'n' in the function names (solve_n1, solve_n3) refers
to the exponent in a non-linear diffusion equation.  
This file solves that equation for different values of 'n'
(specifically, n=0, n=1, and n=3).  The solutions are found 
both analytically (for n=0) and numerically (for n=1 and n=3).
The similarity solutions transform the partial differential equation
into an ordinary differential equation, which is solved using `solve_ivp`.
The boundary conditions are handled using `newton` to find
the correct initial condition for the ODE"""

def dimensionalize(zeta, theta, t, D, w0, L=100):
    x = zeta * np.sqrt(D * t)
    w_x = theta * w0

    valid_indices = x <= L
    x = x[valid_indices]
    w_x = w_x[valid_indices]

    dx = x[1] - x[0]
    x_extended = np.arange(x[-1] + dx, L + dx, dx)
    w_x_extended = np.full_like(x_extended, w_x[-1])    

    x = np.concatenate([x, x_extended])
    w_x = np.concatenate([w_x, w_x_extended])

    return x, w_x

def findRowIndexWithSpecificTime(df, time):
    return np.where(df.iloc[1:, 0].astype(float) >= time)[0][0]