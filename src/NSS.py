import numpy as np
import pandas as pd
from typing import Union
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

def calculate_nss(
    T: Union[float, np.ndarray], 
    a: float, b: float, c: float, d: float, 
    tau: float, theta: float
) -> pd.DataFrame:
    """
    Calculates NSS yields, forward rates, and discount factors.
    Handles T=0 by isolating calculations to prevent division errors.
    
    Parameters:
    -----------
    T     : Maturity (Time to maturity in years)
    a     : Level parameter (long-term yield limit)
    b     : Slope parameter (short-term factor)
    c     : First curvature parameter (medium-term)
    d     : Second curvature parameter (additional medium-term)
    tau   : First curvature maturity parameter (decay)
    theta : Second curvature maturity parameter (steepness)

    Returns:
    --------
    pd.DataFrame : DataFrame with columns:
        - 'zcy_NSS': Zero-coupon yield
        - 'f_NSS': Instantaneous forward rate
        - 'p_NSS': Discount factor
    """
    T = np.atleast_1d(T).astype(float)
    
    zcy = np.zeros_like(T)
    fwd = np.zeros_like(T)
    
    # Isolate indices for T=0 and T>0
    idx_zero = (T == 0)
    idx_pos = (T > 0)
    
    # Handle the T=0 case (The division by zero case)
    zcy[idx_zero] = a + b
    fwd[idx_zero] = a + b
    
    # Handle the T>0 case (The standard formula)
    t_v = T[idx_pos] 
    e_tau = np.exp(-t_v / tau)
    e_theta = np.exp(-t_v / theta)

    L_b = (1 - e_tau) / (t_v / tau)
    L_c = L_b - e_tau
    L_d = ((1 - e_theta) / (t_v / theta)) - e_theta
    
    zcy[idx_pos] = a + (b * L_b) + (c * L_c) + (d * L_d)
    fwd[idx_pos] = a + (e_tau / tau) * (b * tau + c * t_v) + (e_theta / theta) * d * t_v

    # Final step for all maturities
    discount_factor = np.exp(-zcy * T)

    return pd.DataFrame({
        'zcy_NSS': zcy,
        'f_NSS': fwd,
        'p_NSS': discount_factor
    }, index=T).rename_axis('maturity')







# this function is to be revised later


def fit_nss(market_data: pd.DataFrame, weight_method: str = 'none') -> dict:
    """
    Fits the NSS model to observed yields using the General Weighting Method
    from HEC Montreal 60201 (Chapter 2.2.2).
    
    Parameters:
    -----------
    market_data   : pd.DataFrame with 'maturity' and 'yield' columns.
    weight_method : 'none' (Uniform), 'short' (more weight on shorter maturities), or 'long' (more weight on longer maturities)
    """
    T = market_data['maturity'].values
    y_obs = market_data['yield'].values

    if weight_method == 'short':
        w = (1.0 / T) / np.sum(1.0 / T) 
    elif weight_method == 'long':
        w = T / np.sum(T) 
    else:
        w = np.ones_like(T) / len(T) # Uniform weights


    def objective(params):
        a, b, c, d, tau, theta = params
        
        # Stability: a and Decay parameters must be positive 
        if a < 0 or tau <= 0 or theta <= 0:
            return 1e12

        # Avoid division by zero
        T_safe = np.where(T == 0, 1e-10, T) 

        # NSS Yield Calculation
        exp_tau = np.exp(-T_safe / tau)
        exp_theta = np.exp(-T_safe / theta)
        
        L_b = (1 - exp_tau) / (T_safe / tau)
        L_c = L_b - exp_tau
        L_d = ((1 - exp_theta) / (T_safe / theta)) - exp_theta
        
        y_nss = a + (b * L_b) + (c * L_c) + (d * L_d)
        
        # Weighted Squared Error calculation
        return np.sum(w * (y_nss - y_obs)**2)

    # # Optimization using Nelder-Mead for robustness
    # initial_guess = [y_obs[-1], y_obs[0] - y_obs[-1], 0.0, 0.0, 1.0, 2.0]
    # res = minimize(objective, initial_guess, method='Nelder-Mead', tol=1e-6)
    # 1. Define Parameter Bounds (Min, Max) instead of an initial guess
    # a: long term yield, b: slope, c/d: humps, tau/theta: decay
    bounds = [
        (0.0, 0.15),      # a
        (-0.15, 0.15),    # b
        (-0.5, 0.5),    # c
        (-0.5, 0.5),    # d
        (1e-6, 100.0),      # tau (min 0.1 to avoid division by zero)
        (1e-6, 100.0)       # theta (min 0.1 to avoid division by zero)
    ]

    # 2. Global Optimization (Differential Evolution)
    res = differential_evolution(objective, bounds, strategy='best1bin', tol=1e-7)

    return {
        'params': {'a': res.x[0], 'b': res.x[1], 'c': res.x[2], 
                   'd': res.x[3], 'tau': res.x[4], 'theta': res.x[5]},
        'success': res.success,
        'weighted_sse': res.fun,
        'message': res.message
    }
