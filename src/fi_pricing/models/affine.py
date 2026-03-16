import numpy as np
from scipy.stats import norm, ncx2
from .one_factor import OneFactorModel
from ..curves.base import BaseYieldCurve

class VasicekModel(OneFactorModel):
    """
    Vasicek Short Rate Model implementation.
    Dynamics: dr_t = kappa * (theta - r_t) * dt + sigma * dW_t
    """

    def __init__(self, kappa, theta, sigma):
        """
        Args:
            kappa (float): Mean reversion speed.
            theta (float): Long-term mean level.
            sigma (float): Volatility of the short rate.
        """
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma

    def B(self, t, T):
        dt = np.maximum(T - t, 1e-9)
        return (1 - np.exp(-self.kappa * dt)) / self.kappa

    def A(self, t, T):
        dt = np.maximum(T - t, 1e-9)
        B_val = self.B(t, T)
        
        term1 = (self.theta - self.sigma**2 / (2 * self.kappa**2)) * (B_val - dt)
        term2 = (self.sigma**2 / (4 * self.kappa)) * B_val**2
        
        return np.exp(term1 - term2)

    def zcb_option(self, t, T, rt, T_expiry, K, option_type="call"):
        Pt_T = self.P(t, T, rt)
        Pt_Texpiry = self.P(t, T_expiry, rt) 

        tau_opt = T_expiry - t
        tau_bond = np.asanyarray(T) - T_expiry
        
        vol_factor = (1 - np.exp(-2 * self.kappa * tau_opt)) / (2 * self.kappa)
        sigma_p = (self.sigma / self.kappa) * (1 - np.exp(-self.kappa * tau_bond)) * np.sqrt(vol_factor)

        sigma_p = np.maximum(sigma_p, 1e-9) 

        h = (1 / sigma_p) * np.log(Pt_T / (K * Pt_Texpiry)) + sigma_p / 2
        
        if option_type == "call":
            return Pt_T * norm.cdf(h) - K * Pt_Texpiry * norm.cdf(h - sigma_p)
        else:
            return K * Pt_Texpiry * norm.cdf(sigma_p - h) - Pt_T * norm.cdf(-h)


class CIRModel(OneFactorModel):
    """
    Cox-Ingersoll-Ross (CIR) Short Rate Model implementation.
    Dynamics: dr_t = kappa * (theta - r_t) * dt + sigma * sqrt(r_t) * dW_t
    """

    def __init__(self, kappa, theta, sigma):
        """
        Args:
            kappa (float): Mean reversion speed.
            theta (float): Long-term mean level.
            sigma (float): Volatility parameter.
        """
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.gamma = np.sqrt(kappa**2 + 2 * sigma**2)

    def B(self, t, T):
        dt = np.maximum(T - t, 1e-9)
        exp_gamma = np.exp(self.gamma * dt)
        
        numerator = 2 * (exp_gamma - 1)
        denominator = (self.gamma + self.kappa) * (exp_gamma - 1) + 2 * self.gamma
        
        return numerator / denominator

    def A(self, t, T):
        dt = np.maximum(T - t, 1e-9)
        exp_half_dt = np.exp(dt * (self.kappa + self.gamma) / 2)
        exp_gamma = np.exp(self.gamma * dt)
        
        numerator = 2 * self.gamma * exp_half_dt
        denominator = (self.gamma + self.kappa) * (exp_gamma - 1) + 2 * self.gamma
        
        exponent = 2 * self.kappa * self.theta / self.sigma**2
        return (numerator / denominator) ** exponent

    def zcb_option(self, t, T, rt, T_expiry, K, option_type="call"):
        tau = T_expiry - t
        T_bond = np.asanyarray(T)
        
        B_Texp_T = self.B(T_expiry, T_bond) + 1e-9
        
        rho = 2 * self.gamma / (self.sigma**2 * (np.exp(self.gamma * tau) - 1))
        psi = (self.kappa + self.gamma) / self.sigma**2

        df = 4 * self.kappa * self.theta / self.sigma**2
        
        scale_1 = rho + psi + B_Texp_T
        nc_param_1 = (2 * rho**2 * rt * np.exp(self.gamma * tau)) / scale_1
        
        A_Texp_T = self.A(T_expiry, T_bond)
        strike_limit = 2 * scale_1 * np.log(A_Texp_T / K) / B_Texp_T

        scale_2 = rho + psi
        nc_param_2 = (2 * rho**2 * rt * np.exp(self.gamma * tau)) / scale_2
        strike_limit_2 = 2 * scale_2 * np.log(A_Texp_T / K) / B_Texp_T
        Pt_T = self.P(t, T, rt)
        Pt_Texpiry = self.P(t, T_expiry, rt)

        if option_type == "call":
            val_1 = ncx2.cdf(strike_limit, df, nc_param_1)
            val_2 = ncx2.cdf(strike_limit_2, df, nc_param_2)
            return Pt_T * val_1 - K * Pt_Texpiry * val_2
        else:
            call_price = Pt_T * ncx2.cdf(strike_limit, df, nc_param_1) - K * Pt_Texpiry * ncx2.cdf(strike_limit_2, df, nc_param_2)
            return call_price - Pt_T + K * Pt_Texpiry


class HullWhiteModel(OneFactorModel):
    """
    Hull-White One-Factor Model.
    Dynamics: dr_t = kappa * (phi(t) - r_t)dt + sigma * dW_t
    
    Distinctive feature:
    Fits the initial term structure of interest rates exactly using an input yield curve.
    """

    def __init__(self, kappa: float, sigma: float, yield_curve: BaseYieldCurve):
        """
        Args:
            kappa (float): The speed of mean reversion.
            sigma (float): The volatility of the short rate.
            yield_curve (BaseYieldCurve): An object (e.g., NSS) with .P(t, T) and .f(t, T).
        """
        self.kappa = kappa
        self.sigma = sigma
        self.curve = yield_curve

    def B(self, t, T):
        dt = np.maximum(T - t, 1e-9)
        return (1 - np.exp(-self.kappa * dt)) / self.kappa

    def A(self, t, T):
        P_0_T = self.curve.P(0, T)
        P_0_t = self.curve.P(0, t)
        f_0_t = self.curve.f(0, t)
        B_t_T = self.B(t, T)
        term_1 = B_t_T * f_0_t
        term_2 = (self.sigma**2 / (4 * self.kappa)) * (1 - np.exp(-2 * self.kappa * t)) * B_t_T**2
        exp_term = np.exp(term_1 - term_2)
        return (P_0_T / P_0_t) * exp_term

    def zcb_option(self, t, T, rt, T_expiry, K, option_type="call"):
        """
        Prices a ZCB option using the Hull-White explicit formula (Gaussian).

        Args:
            t (float): Valuation date.
            T (float or np.ndarray): Maturity date(s) of the underlying ZCB.
            rt (float): Current instantaneous short rate.
            T_expiry (float): Expiry date of the option.
            K (float or np.ndarray): Strike price of the option.
            option_type (str): "call" or "put".
        Returns:
            float or np.ndarray: The price of the option.
        """
        Pt_T = self.P(t, T, rt)
        Pt_Texpiry = self.P(t, T_expiry, rt) 

        tau_opt = T_expiry - t
        tau_bond = np.asanyarray(T) - T_expiry
        
        vol_factor = (1 - np.exp(-2 * self.kappa * tau_opt)) / (2 * self.kappa)
        sigma_p = (self.sigma / self.kappa) * (1 - np.exp(-self.kappa * tau_bond)) * np.sqrt(vol_factor)

        sigma_p = np.maximum(sigma_p, 1e-9) 

        h = (1 / sigma_p) * np.log(Pt_T / (K * Pt_Texpiry)) + sigma_p / 2
        
        if option_type == "call":
            return Pt_T * norm.cdf(h) - K * Pt_Texpiry * norm.cdf(h - sigma_p)
        else:
            return K * Pt_Texpiry * norm.cdf(sigma_p - h) - Pt_T * norm.cdf(-h)