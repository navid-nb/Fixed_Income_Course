import numpy as np
import pandas as pd
from typing import Union
from scipy.stats import norm
from scipy.optimize import brentq

def hull_white_zcb(
    T: Union[float, np.ndarray],
    rt: float,
    kappa: float,
    sigma: float,
    t: float,
    market_zcb_func,
    market_fwd_func
) -> pd.DataFrame:
    """
    Calculates Hull-White ZCB prices and yields for single or multiple maturities.
    Calibrated to market functions (e.g., from NSS model).
    """
    T_arr = np.atleast_1d(T).astype(float)
    tau = T_arr - t
    
    # b_vk is the same as in Vasicek model (3.10)
    # We use np.where to handle the tau=0 case for safety
    b_vk = np.where(tau > 0, (1 - np.exp(-kappa * tau)) / kappa, 0)
    
    # a_hw(t, T) formula (3.42) 
    # We vectorized the market function calls
    pm_0_T = np.array([market_zcb_func(m) for m in T_arr])
    pm_0_t = market_zcb_func(t)
    fm_0_t = market_fwd_func(t)
    
    term1 = pm_0_T / pm_0_t
    term2 = np.exp(b_vk * fm_0_t - (sigma**2 * (1 - np.exp(-2*kappa*t)) * b_vk**2) / (4 * kappa))
    a_hw = term1 * term2
    
    # P_hw(t, T) formula (3.41)
    price = a_hw * np.exp(-b_vk * rt)
    
    # zcy_hw(t, T)
    zcy = np.where(tau > 0, -np.log(price) / tau, rt)
    
    return pd.DataFrame({
        'zcb_price_hw': price,
        'zcy_hw': zcy
    }, index=T_arr).rename_axis('maturity')


def hull_white_coupon_bond(
    maturity: float,
    coupon_rate: float,
    frequency: int,
    rt: float,
    kappa: float,
    sigma: float,
    t: float,
    market_zcb_func,
    market_fwd_func,
    face_value: float = 1.0
) -> pd.DataFrame:
    """
    Calculates the price of a coupon bond in the Hull-White model.
    Calibrated to NSS market functions.
    """
    # 1. Generate the payment schedule (maturities)
    num_payments = int(maturity * frequency)
    maturities = np.linspace(1/frequency, maturity, num_payments)
    
    # 2. Generate the cash flow amounts (c_i)
    # Following textbook convention: final payment includes face value
    coupon_payment = (coupon_rate * face_value) / frequency
    cash_flows = np.full(num_payments, coupon_payment)
    cash_flows[-1] += face_value
    
    # 3. Discount each cash flow using the HW ZCB function
    # We only care about payments happening after current time t
    mask = maturities > t
    relevant_mats = maturities[mask]
    relevant_cfs = cash_flows[mask]
    
    # Calculate HW prices for all relevant maturities
    # We use a list comprehension to call your HW ZCB function for each date
    pv_cfs = []
    for m, cf in zip(relevant_mats, relevant_cfs):
        hw_df = hull_white_zcb(
            T=m, rt=rt, kappa=kappa, sigma=sigma, t=t, 
            market_zcb_func=market_zcb_func, 
            market_fwd_func=market_fwd_func
        )
        pv_cfs.append(cf * hw_df['zcb_price_hw'].iloc[0])
    
    total_price = np.sum(pv_cfs)
    
    return (total_price)


def hull_white_zcb_option(
    T: float,
    T_call: float, 
    K: float, 
    rt: float, 
    kappa: float, 
    sigma: float,
    t: float,
    market_zcb_func,
    market_fwd_func,
    option_type: str = 'call'
) -> float:
    """
    Calculates the European option price on a single ZCB in Hull-White.
    Returns a float.
    """
    # 1. Get HW Bond Prices at time t
    # Using your existing hull_white_zcb and extracting the float price
    P_t_T = hull_white_zcb(T, rt, kappa, sigma, t, market_zcb_func, market_fwd_func)['zcb_price_hw'].iloc[0]
    P_t_Tcall = hull_white_zcb(T_call, rt, kappa, sigma, t, market_zcb_func, market_fwd_func)['zcb_price_hw'].iloc[0]

    # 2. Sigma_p formula (3.14)
    vol_term = (sigma / kappa) * (1 - np.exp(-kappa * (T - T_call)))
    sqrt_term = np.sqrt((1 - np.exp(-2 * kappa * (T_call - t))) / (2 * kappa))
    sigma_p = vol_term * sqrt_term
    
    # 3. h formula (3.13)
    h = (sigma_p / 2) + (1 / sigma_p) * np.log(P_t_T / (K * P_t_Tcall))
    
    # 4. Price calculation
    if option_type.lower() == 'call':
        price = P_t_T * norm.cdf(h) - K * P_t_Tcall * norm.cdf(h - sigma_p)
    elif option_type.lower() == 'put':
        price = K * P_t_Tcall * norm.cdf(sigma_p - h) - P_t_T * norm.cdf(-h)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return float(price)
    



def hull_white_coupon_bond_option(
    maturity: float,
    coupon_rate: float,
    frequency: int,
    T_call: float,
    K: float,
    rt: float,
    kappa: float,
    sigma: float,
    t: float,
    market_zcb_func,
    market_fwd_func,
    option_type: str = 'call'
) -> float:
    """
    Values a European option on a coupon bond in HW using Jamshidian decomposition.
    Returns a float.
    """
    # 1. Generate bond payment schedule
    num_payments = int(maturity * frequency)
    T_all = np.linspace(1/frequency, maturity, num_payments)
    c_all = np.full(num_payments, (coupon_rate / frequency))
    c_all[-1] += 1.0 # Standard unit face value
    
    # Filter for cash flows after option expiry
    mask = T_all > T_call
    c_i = c_all[mask]
    T_i = T_all[mask]

    # 2. Find critical rate r* such that sum(c_i * P_hw(T_call, T_i, r*)) = K
    def objective(r_star):
        prices = [hull_white_zcb(m, r_star, kappa, sigma, T_call, 
                                 market_zcb_func, market_fwd_func)['zcb_price_hw'].iloc[0] 
                  for m in T_i]
        return np.sum(c_i * prices) - K

    r_star = brentq(objective, -0.5, 0.5)

    # 3. Sum individual ZCB options
    total_option_price = 0.0
    for i in range(len(c_i)):
        # Calculate mini-strike Ki = P_hw(T_call, Ti, r*)
        Ki = hull_white_zcb(T_i[i], r_star, kappa, sigma, T_call, 
                             market_zcb_func, market_fwd_func)['zcb_price_hw'].iloc[0]
        
        # Call the simplified ZCB option function
        total_option_price += c_i[i] * hull_white_zcb_option(
            T=T_i[i], T_call=T_call, K=Ki, rt=rt, 
            kappa=kappa, sigma=sigma, t=t, 
            market_zcb_func=market_zcb_func, 
            market_fwd_func=market_fwd_func, 
            option_type=option_type
        )
        
    return float(total_option_price)