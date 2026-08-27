from mypackages import typesdefs


def crossover(
    x: typesdefs.Vector,
    a: float,
    alpha: float,
    b: float,
    beta: float,
    x0: float,
    delta: float,
) -> typesdefs.Vector:
    """
    A function that combines two different power-laws separated by x0
    Inputs:
        x: Independent variable,
        a, b: Prefactor of respective power-law functions
        alpha, beta: Exponents of respective power-law functions
        x0: Crossover (transition) scale that separates the two power-law functions
        delta: Controls how sharp/smoot the transition is
    Output:
        Crossover function
    """
    return (a * x**alpha) / (1 + (x / x0) ** delta) + (b * x**beta) / (
        1 + (x0 / x) ** delta
    )
