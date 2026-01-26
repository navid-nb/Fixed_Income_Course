import numpy as np
import pandas as pd
from typing import Union
from scipy.stats import norm
from scipy.optimize import brentq

def vasicek_zcb(
    T: Union[float, np.ndarray], 
    rt: float, 
    kappa: float, 
    theta: float, 
    sigma: float,
    t: float = 0
) -> pd.DataFrame:
    """
    Calculates Vasicek zero-coupon bond prices and yields.
    Based on formulas (3.8), (3.9), and (3.10) [cite: 419-421].
    
    Parameters:
    -----------
    T     : Maturity date
    rt    : instantaneous risk-free rate at time t
    kappa : Speed of mean reversion
    theta : Long-term mean level
    sigma : Volatility of interest rate
    t     : valuation time (defaults to 0)

    Returns:
    --------
    pd.DataFrame : DataFrame with columns:
        - 'zcb_price_vk': Zero-coupon bond price
        - 'zcy_vk': Zero-coupon yield
    """
    T_arr = np.atleast_1d(T).astype(float)
    tau = T_arr - t
    
    # b_vk(t, T) formula (3.10) 
    b_vk = (1 - np.exp(-kappa * tau)) / kappa
    
    # a_vk(t, T) formula (3.9) 
    term1 = (theta - (sigma**2 / (2 * kappa**2))) * (b_vk - tau)
    term2 = (sigma**2 / (4 * kappa)) * (b_vk**2)
    a_vk = np.exp(term1 - term2)
    
    # P_vk(t, T) formula (3.8) 
    price = a_vk * np.exp(-b_vk * rt)
    
    # zcy_vk(t, T) formula (3.11) 
    # Handle tau=0 to avoid division by zero
    zcy = np.zeros_like(tau)
    idx = tau > 0
    zcy[idx] = -(np.log(a_vk[idx]) - b_vk[idx] * rt) / tau[idx]
    zcy[~idx] = rt
    
    return pd.DataFrame({
        'zcb_price_vk': price,
        'zcy_vk': zcy
    }, index=T_arr).rename_axis('maturity')


def vasicek_coupon_bond(
    maturity: float,
    coupon_rate: float,
    frequency: int,
    rt: float,
    kappa: float,
    theta: float,
    sigma: float,
    face_value: float = 1,
    t: float = 0
) -> pd.Series:
    """
    Calculates the value, Macaulay duration, and Stochastic duration of a coupon bond.
    """
    # 1. Generate the payment schedule
    num_payments = int(maturity * frequency)
    maturities = np.linspace(1/frequency, maturity, num_payments)

    # 2. Generate the cash flow amounts
    coupon_payment = (coupon_rate * face_value) / frequency
    cash_flows = np.full(num_payments, coupon_payment)
    cash_flows[-1] += face_value
    
    # 3. Get ZCB prices and B(t, T) factors
    # We need b_vk for Stochastic Duration: B(t, T) = (1 - e^-kappa*tau) / kappa
    tau = maturities - t
    b_vk = (1 - np.exp(-kappa * tau)) / kappa
    
    df_zcb = vasicek_zcb(maturities, rt, kappa, theta, sigma, t=t)
    zcb_prices = df_zcb['zcb_price_vk'].values
    
    # 4. Calculate Price
    pv_cfs = cash_flows * zcb_prices
    total_price = np.sum(pv_cfs)
    
    # 5. Calculate Macaulay Duration (D_mac)
    # Formula: sum(ti * PV(cf_i)) / Price
    d_macaulay = np.sum(maturities * pv_cfs) / total_price
    
    # 6. Calculate Stochastic Duration (D_stoch)
    # Formula: sum(B(t, Ti) * PV(cf_i)) / Price
    d_stochastic = np.sum(b_vk * pv_cfs) / total_price
    
    return pd.Series({
        'price': float(total_price),
        'macaulay_duration': float(d_macaulay),
        'stochastic_duration': float(d_stochastic)
    }, name="bond_metrics")

def vasicek_zcb_option(
    T: float,
    T_call: float, 
    K: float, 
    rt: float, 
    kappa: float, 
    theta: float, 
    sigma: float,
    option_type: str = 'call',
    t: float = 0
) -> float:
    """
    Calculates European option price on a zero-coupon bond in Vasicek model.

    Parameters:
    -----------
    T         : Maturity of the underlying zero-coupon bond
    T_call    : Maturity of the option
    K         : Strike price of the option (assuming face value = 1 for ZCB)
    rt        : instantaneous risk-free rate at time t
    kappa     : Speed of mean reversion
    theta     : Long-term mean level
    sigma     : Volatility of interest rate
    option_type : 'call' or 'put' to specify option type
    t         : valuation time (defaults to 0)

    Returns:
    --------
    float : Price of the European option on the zero-coupon bond
    """
    # 1. Get bond prices at time t using your existing vasicek_zcb
    # We use .iloc[0] to extract the scalar float from the DataFrame
    P_t_T = vasicek_zcb(T, rt, kappa, theta, sigma, t=t)['zcb_price_vk'].iloc[0]
    P_t_Tcall = vasicek_zcb(T_call, rt, kappa, theta, sigma, t=t)['zcb_price_vk'].iloc[0]
    
    # 2. sigma_p formula generalized for t (Formula 3.14)
    vol_term = (sigma / kappa) * (1 - np.exp(-kappa * (T - T_call)))
    sqrt_term = np.sqrt((1 - np.exp(-2 * kappa * (T_call - t))) / (2 * kappa))
    sigma_p = vol_term * sqrt_term
    
    # 3. h formula generalized for t (Formula 3.13)
    h = (sigma_p / 2) + (1 / sigma_p) * np.log(P_t_T / (K * P_t_Tcall))
    
    # 4. Price calculation
    if option_type.lower() == 'call':
        # Formula (3.12)
        price = P_t_T * norm.cdf(h) - K * P_t_Tcall * norm.cdf(h - sigma_p)
    elif option_type.lower() == 'put':
        # Formula (3.15)
        price = K * P_t_Tcall * norm.cdf(sigma_p - h) - P_t_T * norm.cdf(-h)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return price




def vasicek_coupon_bond_option(
    T: float,
    T_call: float,
    coupon_rate: float,
    frequency: int,
    K: float,
    rt: float,
    kappa: float,
    theta: float,
    sigma: float,
    t: float = 0,
    option_type: str = 'call'
) -> float:
    """
    Values a European option on a coupon-bearing bond using Jamshidian decomposition.
    Based on textbook Formula (3.23). Assumes unit face value of 1.

    Parameters:
    -----------
    T            : Maturity of the coupon bond
    coupon_rate  : Annual coupon rate (as a decimal, e.g., 0.05 for 5%)
    frequency    : Number of coupon payments per year
    T_call       : Maturity of the option
    K            : Strike price of the option
    rt           : instantaneous risk-free rate at time t
    kappa       : Speed of mean reversion
    theta       : Long-term mean level
    sigma       : Volatility of interest rate
    t            : valuation time (defaults to 0)
    option_type  : 'call' or 'put' to specify option type

    Returns:
    --------
    float : Price of the European option on the coupon bond

    """
    # 1. Generate payment schedule (T_i) and cash flows (c_i)
    # Final c_i includes the unit face value (1.0)
    num_payments = int(T * frequency)
    maturities = np.linspace(1/frequency, T, num_payments)
    c_i_val = coupon_rate / frequency
    c_i = np.full(num_payments, c_i_val)
    c_i[-1] += 1.0 # Add face value of 1
    
    # Filter for cash flows occurring after the option maturity (T_i > T_call)
    mask = maturities > T_call
    relevant_cfs = c_i[mask]
    relevant_mats = maturities[mask]

    # 2. Solve for r* such that sum(c_i * P(T_call, T_i, r*)) = K [cite: 447]
    def objective(r_star):
        prices = vasicek_zcb(relevant_mats, r_star, kappa, theta, sigma, t=T_call)['zcb_price_vk'].values
        return np.sum(relevant_cfs * prices) - K

    r_star = brentq(objective, -1.0, 1.0)

    # 3. Calculate individual strikes K_i = P(T_call, T_i, r*) [cite: 453, 467]
    Ki_strikes = vasicek_zcb(relevant_mats, r_star, kappa, theta, sigma, t=T_call)['zcb_price_vk'].values
    
    # 4. Final Summation: sum [ c_i * ZCB_Option(T_i, T_call, K_i) ] [cite: 459]
    total_val = 0.0
    for i in range(len(relevant_cfs)):
        total_val += relevant_cfs[i] * vasicek_zcb_option(
            T=relevant_mats[i], T_call=T_call, K=Ki_strikes[i],
            rt=rt, kappa=kappa, theta=theta, sigma=sigma,
            t=t, option_type=option_type
        )
        
    return total_val