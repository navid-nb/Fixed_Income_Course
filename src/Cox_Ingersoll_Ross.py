import numpy as np
import pandas as pd
from typing import Union
from scipy.stats import ncx2
from scipy.optimize import brentq

def cir_zcb(
    T: Union[float, np.ndarray], 
    rt: float, 
    kappa: float, 
    theta: float, 
    sigma: float,
    t: float = 0
) -> pd.DataFrame:
    """
    Calculates CIR zero-coupon bond prices and yields.
    Based on formulas (3.34) - (3.37)

    Parameters:
    -----------
    T     : Maturity or array of maturities
    rt    : Short rate at time t
    kappa : Speed of mean reversion
    theta : Long-term mean level
    sigma : Volatility parameter
    t     : Current time (default is 0)

    Returns:
    --------
    pd.DataFrame with columns:
        - 'zcb_price_cir': Zero-coupon bond prices
        - 'zcy_cir': Zero-coupon yields 
    """
    T_arr = np.atleast_1d(T).astype(float)
    tau = T_arr - t
    
    # gamma formula (3.37)
    gamma = np.sqrt(kappa**2 + 2 * sigma**2)
    
    # Helper terms for a_cir and b_cir
    exp_gamma = np.exp(gamma * tau)
    denom = (kappa + gamma) * (exp_gamma - 1) + 2 * gamma
    
    # a_cir(t, T) formula (3.35) 
    term_a = (2 * gamma * np.exp((kappa + gamma) * tau / 2)) / denom
    a_cir = term_a**(2 * kappa * theta / sigma**2)
    
    # b_cir(t, T) formula (3.36) 
    b_cir = 2 * (exp_gamma - 1) / denom
    
    # P_cir(t, T) formula (3.34) 
    price = a_cir * np.exp(-b_cir * rt)
    
    # zcy_cir(t, T) formula (3.38)
    zcy = np.zeros_like(tau)
    idx = tau > 0
    zcy[idx] = -(np.log(a_cir[idx]) - b_cir[idx] * rt) / tau[idx]
    zcy[~idx] = rt
    
    return pd.DataFrame({
        'zcb_price_cir': price,
        'zcy_cir': zcy
    }, index=T_arr).rename_axis('maturity')


def cir_coupon_bond(
    maturity: float,
    coupon_rate: float,
    frequency: int,
    rt: float,
    kappa: float,
    theta: float,
    sigma: float,
    t: float = 0,
    face_value: float = 1,
) -> float:
    """
    Values a CIR coupon bond assuming unit face value of 1.

    Parameters:
    -----------
    maturity     : Maturity of the bond in years
    coupon_rate  : Annual coupon rate (as a decimal, e.g., 0.05 for 5%)
    frequency    : Number of coupon payments per year
    rt           : Short rate at time t
    kappa        : Speed of mean reversion
    theta        : Long-term mean level
    sigma        : Volatility parameter
    t            : Current time (default is 0)
    face_value   : Face value of the bond (default is 1)

    Returns:
    --------
    float : Present value of the coupon bond
    """
    num_payments = int(maturity * frequency)
    maturities = np.linspace(1/frequency, maturity, num_payments)
    c_i = np.full(num_payments, coupon_rate / frequency) * face_value
    c_i[-1] += face_value # Standard textbook unit face value
    
    df_zcb = cir_zcb(maturities, rt, kappa, theta, sigma, t=t)
    return np.sum(c_i * df_zcb['zcb_price_cir'].values)


def cir_zcb_option(
    T: float,
    T_call: float, 
    K: float, 
    rt: float, 
    kappa: float, 
    theta: float, 
    sigma: float,
    t: float = 0,
    option_type: str = 'call'
) -> float:
    """
    European option on a ZCB in CIR model.
    Based on formula (3.39)
    """
    gamma = np.sqrt(kappa**2 + 2 * sigma**2)
    
    # Get bond prices at time t
    P_0_T = cir_zcb(T, rt, kappa, theta, sigma, t=t)['zcb_price_cir'].iloc[0]
    P_0_Tcall = cir_zcb(T_call, rt, kappa, theta, sigma, t=t)['zcb_price_cir'].iloc[0]
    
    
    phi = (2 * gamma) / (sigma**2 * (np.exp(gamma * (T_call - t)) - 1))
    psi = (kappa + gamma) / sigma**2
    
    # b_cir and a_cir at T_call for maturity T
    df_Tcall_T = cir_zcb(T, 0, kappa, theta, sigma, t=T_call)
    b_Tcall_T = 2 * (np.exp(gamma*(T-T_call))-1) / ((kappa+gamma)*(np.exp(gamma*(T-T_call))-1)+2*gamma)
    a_Tcall_T = ((2*gamma*np.exp((kappa+gamma)*(T-T_call)/2))/((kappa+gamma)*(np.exp(gamma*(T-T_call))-1)+2*gamma))**(2*kappa*theta/sigma**2)

    xi = phi + psi + b_Tcall_T
    df = 4 * kappa * theta / sigma**2
    
    
    nc_centrality = (2 * phi**2 * rt * np.exp(gamma * (T_call - t)))
    
    # Calculation of r_hat (strike rate)
    r_hat = (1 / b_Tcall_T) * np.log(a_Tcall_T / K)
    
    if option_type.lower() == 'call':
        term1 = P_0_T * ncx2.cdf(2 * r_hat * xi, df, nc_centrality / xi)
        term2 = K * P_0_Tcall * ncx2.cdf(2 * r_hat * (phi + psi), df, nc_centrality / (phi + psi))
        return term1 - term2
    elif option_type.lower() == 'put':
        # Corrected CIR Put formula based on symmetry of the chi-square CDF
        term1 = K * P_0_Tcall * ncx2.cdf(2 * r_hat * (phi + psi), df, nc_centrality / (phi + psi))
        term2 = P_0_T * ncx2.cdf(2 * r_hat * xi, df, nc_centrality / xi)
        
        # In a Put, we reverse the probabilities compared to a Call
        price = K * P_0_Tcall * (1 - ncx2.cdf(2 * r_hat * (phi + psi), df, nc_centrality / (phi + psi))) - \
                P_0_T * (1 - ncx2.cdf(2 * r_hat * xi, df, nc_centrality / xi))
        return price
    

    

def cir_coupon_bond_option(
    T: float,
    coupon_rate: float,
    frequency: int,
    T_call: float,
    K: float,
    rt: float,
    kappa: float,
    theta: float,
    sigma: float,
    t: float = 0,
    option_type: str = 'call'
) -> float:
    """
    Values European coupon bond options in CIR using Jamshidian decomposition.
    Based on Formula (3.23)

    Parameters:
    -----------
    T     : Maturity of the bond in years
    coupon_rate  : Annual coupon rate (as a decimal, e.g., 0.05 for 5%)
    frequency    : Number of coupon payments per year
    T_call       : Maturity of the option
    K            : Strike price of the option
    rt           : Short rate at time t
    kappa       : Speed of mean reversion
    theta       : Long-term mean level
    sigma       : Volatility parameter
    t            : Current time (default is 0)
    option_type  : 'call' or 'put' to specify option type

    Returns:
    --------
    float : Price of the European option on the coupon bond

    """
    num_payments = int(T * frequency)
    maturities = np.linspace(1/frequency, T, num_payments)
    c_i = np.full(num_payments, coupon_rate / frequency)
    c_i[-1] += 1.0 
    
    mask = maturities > T_call
    relevant_cfs = c_i[mask]
    relevant_mats = maturities[mask]

    # Find r* such that sum(c_i * P_cir(T_call, T_i, r*)) = K 
    def objective(r_star):
        prices = cir_zcb(relevant_mats, r_star, kappa, theta, sigma, t=T_call)['zcb_price_cir'].values
        return np.sum(relevant_cfs * prices) - K

    r_star = brentq(objective, 1e-6, 1.0) # r* must be positive in CIR

    # Calculate individual strikes K_i 
    Ki_strikes = cir_zcb(relevant_mats, r_star, kappa, theta, sigma, t=T_call)['zcb_price_cir'].values
    
    total_val = 0.0
    for i in range(len(relevant_cfs)):
        total_val += relevant_cfs[i] * cir_zcb_option(
            T=relevant_mats[i], T_call=T_call, K=Ki_strikes[i],
            rt=rt, kappa=kappa, theta=theta, sigma=sigma,
            t=t, option_type=option_type
        )
    return total_val