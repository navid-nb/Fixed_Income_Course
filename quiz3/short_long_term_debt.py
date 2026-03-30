"""
Question 5 model: short-term and long-term debt with equal seniority.
Implements Equations (9.24) and (9.25).
"""

import math
from scipy.stats import norm, multivariate_normal


def _phi2(x, y, corr):
    """Bivariate standard normal CDF with correlation corr."""
    cov = [[1.0, corr], [corr, 1.0]]
    return multivariate_normal(mean=[0.0, 0.0], cov=cov).cdf([x, y])


def _credit_spread_bps(debt_value, nominal, maturity, r):
    """Credit spread in bps from debt price."""
    if debt_value <= 0.0:
        return float("nan")
    return (-(1.0 / maturity) * math.log(debt_value / nominal) - r) * 10000.0


def short_long_term_debt_model(v, m, t, M, T, r, sigma, theta=1.0):
    """
    Compute short-term and long-term debt values and spreads for Question 5.

    Parameters
    ----------
    v : float
        Current asset value.
    m : float
        Short-term debt nominal.
    t : float
        Short-term maturity.
    M : float
        Long-term debt nominal.
    T : float
        Long-term maturity.
    r : float
        Risk-free rate.
    sigma : float
        Asset volatility.
    theta : float, optional
        Recovery fraction. Use theta=1.0 when default costs are ignored.

    Returns
    -------
    dict
        Short-term debt value, long-term debt value, and both spreads in bps.
    """
    if t <= 0.0 or T <= 0.0:
        raise ValueError("Maturities t and T must be strictly positive.")
    if sigma <= 0.0:
        raise ValueError("sigma must be strictly positive.")
    if m <= 0.0 or M <= 0.0 or v <= 0.0:
        raise ValueError("v, m, and M must be strictly positive.")

    # Eq. (9.26)
    z1 = (math.log(v / m) + (r + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
    z2 = (math.log(v / m) + (r - 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))

    # Eq. (9.27)
    z3 = (math.log(v / M) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    z4 = (math.log(v / M) + (r - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

    corr = math.sqrt(t / T)

    # Eq. (9.24)
    d_t = m * math.exp(-r * t) * norm.cdf(z2) + (m / (m + M)) * theta * v * norm.cdf(-z1)

    # Eq. (9.25)
    D_T = (
        M * math.exp(-r * T) * _phi2(z2, z4, corr)
        + theta * v * _phi2(z1, -z3, -corr)
        + (M / (m + M)) * theta * v * norm.cdf(-z1)
    )

    return {
        "short_term_debt": d_t,
        "long_term_debt": D_T,
        "short_term_spread_bps": _credit_spread_bps(d_t, m, t, r),
        "long_term_spread_bps": _credit_spread_bps(D_T, M, T, r),
        "z1": z1,
        "z2": z2,
        "z3": z3,
        "z4": z4,
    }
