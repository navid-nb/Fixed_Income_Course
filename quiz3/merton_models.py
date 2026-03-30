"""
Merton (1974, 1976) Credit Spread Models
Implementing zero-coupon debt valuation with and without jumps
"""

import numpy as np
import math
from scipy.stats import norm


def black_scholes_call(v, T, M, r, sigma):
    """
    Black-Scholes call option pricing formula
    
    Parameters:
    -----------
    v : float
        Current asset value
    T : float
        Time to maturity (years)
    M : float
        Strike price (debt face value)
    r : float
        Risk-free rate
    sigma : float
        Volatility of assets
        
    Returns:
    --------
    float
        Call option value
    """
    if sigma == 0 or T == 0:
        return max(v - M * np.exp(-r * T), 0)
    
    d1 = (np.log(v / M) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call = v * norm.cdf(d1) - M * np.exp(-r * T) * norm.cdf(d2)
    return call


def merton_1974(v, T, M, r, sigma, theta=1.0):
    """
    Merton (1974) Model - Zero-coupon debt with geometric Brownian motion
    
    Parameters:
    -----------
    v : float
        Current firm asset value
    T : float
        Time to maturity (years)
    M : float
        Debt face value
    r : float
        Risk-free rate
    sigma : float
        Asset volatility
    theta : float
        Recovery rate (default=1.0, no losses)
        
    Returns:
    --------
    dict with keys:
        - equity: Equity value
        - debt: Debt value
        - credit_spread: Credit spread in basis points
    """
    
    # Calculate d1 and d2
    d1 = (np.log(v / M) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Equity is a call option on assets (Equation 9.3)
    equity = v * norm.cdf(d1) - M * np.exp(-r * T) * norm.cdf(d2)
    
    # Debt value with recovery rate theta (Equation 9.4)
    # D = M * exp(-rT) * N(d2) + θ * v * N(-d1)
    debt = M * np.exp(-r * T) * norm.cdf(d2) + theta * v * norm.cdf(-d1)
    
    # Credit spread (Equation 9.6)
    # s(T) = -1/T * ln(D/M) - r
    credit_spread = -np.log(debt / M) / T - r
    
    # Convert to basis points
    credit_spread_bps = credit_spread * 10000
    
    return {
        'equity': equity,
        'debt': debt,
        'credit_spread': credit_spread,
        'credit_spread_bps': credit_spread_bps,
        'd1': d1,
        'd2': d2
    }


def merton_1976_with_jumps(v, T, M, r, sigma, lambda_param, mu_j, sigma_j, n_max=50):
    """
    Merton (1976) Jump-Diffusion Model for zero-coupon debt valuation.

    Extends the Merton (1974) framework by allowing the firm's asset value to
    follow a jump-diffusion process (geometric Brownian motion + compound Poisson
    jumps). Equity is priced as a Poisson-weighted sum of Black-Scholes call
    values, each conditioned on n jumps occurring over [0, T]. Debt is recovered
    via the structural identity D = v - E, which guarantees E + D = v exactly
    (i.e., no partial recovery / no deadweight loss).

    The risk-neutral Poisson intensity is adjusted to lambda * (1 + mu_j) to
    preserve the martingale condition on the asset process.

    Parameters
    ----------
    v : float
        Current firm asset value (V_0).
    T : float
        Time to debt maturity in years.
    M : float
        Face value of zero-coupon debt (the strike price).
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Diffusion (continuous) component of asset volatility.
    lambda_param : float
        Poisson jump arrival rate (expected number of jumps per year).
    mu_j : float
        Mean percentage jump size (e.g., -0.2 for a 20% average downward jump).
        Must satisfy mu_j > -1.
    sigma_j : float
        Standard deviation of the log jump size (jump volatility).
    n_max : int, optional
        Maximum number of jumps to include in the Poisson sum (default=50).
        Convergence is typically achieved well before n_max for reasonable
        values of lambda_param * T.

    Returns
    -------
    dict with keys:
        equity : float
            Market value of equity (call option on firm assets).
        debt : float
            Market value of risky zero-coupon debt, derived as D = v - E.
            Satisfies E + D = v exactly (full-recovery structural identity).
        credit_spread : float
            Annualised credit spread in decimal form:
            s(T) = -ln(D / M) / T - r.
        credit_spread_bps : float
            Credit spread expressed in basis points (credit_spread * 10000).

    Notes
    -----
    - Reference: Merton, R.C. (1976). "Option pricing when underlying stock
      returns are discontinuous." Journal of Financial Economics, 3(1-2), 125-144.
    - The effective risk-free rate per jump state n is:
          r_eff = r - lambda * mu_j + n * ln(1 + mu_j) / T
    - The effective volatility per jump state n is:
          sigma_eff = sqrt(sigma^2 + n * sigma_j^2 / T)
    - Partial recovery (theta < 1) is not part of the original Merton (1976)
      model. Use merton_1974() with theta < 1 if deadweight bankruptcy costs
      are required.
    """
    equity = 0.0

    for n in range(n_max + 1):
        # Poisson weight for n jumps under risk-neutral measure
        poisson_weight = (
            (lambda_param * (1 + mu_j) * T) ** n / math.factorial(n)
        ) * np.exp(-lambda_param * (1 + mu_j) * T)

        # State-n effective parameters
        r_eff = r - lambda_param * mu_j + (n * np.log(1 + mu_j)) / T
        sigma_eff = np.sqrt(sigma**2 + n * sigma_j**2 / T)

        # Equity contribution from jump state n
        call_value = black_scholes_call(v, T, M, r_eff, sigma_eff)
        equity += poisson_weight * call_value

    # Structural identity: D = v - E  (guarantees E + D = v exactly)
    debt = v - equity

    # Credit spread: s(T) = -ln(D/M)/T - r
    credit_spread = -np.log(debt / M) / T - r
    credit_spread_bps = credit_spread * 10000

    return {
        'equity': equity,
        'debt': debt,
        'credit_spread': credit_spread,
        'credit_spread_bps': credit_spread_bps,
    }
