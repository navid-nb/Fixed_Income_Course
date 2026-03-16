import numpy as np
from .base import BaseYieldCurve

class NelsonSiegelSvensson(BaseYieldCurve):
    """
    Nelson-Siegel-Svensson (NSS) Yield Curve Model.
    """

    def __init__(self, a, b, c, d, tau, theta):
        """
        Initialize the NSS model parameters.

        Args:
            a (float): Long-term level component (beta 0).
            b (float): Short-term component (beta 1).
            c (float): Medium-term component (beta 2).
            d (float): Additional long-term component (beta 3).
            tau (float): Decay factor 1.
            theta (float): Decay factor 2.
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.tau = tau
        self.theta = theta

    def zcy(self, t, T):
        """
        Calculates the zero-coupon spot rate for a given maturity.

        Args:
            t (float): Valuation date (in years).
            T (float or np.ndarray): Maturity date(s) (in years).

        Returns:
            float or np.ndarray: The zero rate(s) y(t, T).
        """
        dt = np.maximum(np.asanyarray(T) - t, 1e-9)
        
        ratio_tau = dt / self.tau
        ratio_theta = dt / self.theta
        
        term_b = (1 - np.exp(-ratio_tau)) / ratio_tau
        term_c = term_b - np.exp(-ratio_tau)
        term_d = (1 - np.exp(-ratio_theta)) / ratio_theta - np.exp(-ratio_theta)

        return self.a + self.b * term_b + self.c * term_c + self.d * term_d

    def f(self, t, T):
        """
        Calculates the instantaneous forward rate at t for maturity T.

        Args:
            t (float): Valuation date (in years).
            T (float or np.ndarray): Maturity date(s) (in years).

        Returns:
            float or np.ndarray: The instantaneous forward rate f(t, T).
        """
        dt = np.maximum(np.asanyarray(T) - t, 1e-9)
        
        exp_tau = np.exp(-dt / self.tau)
        exp_theta = np.exp(-dt / self.theta)

        term_c = (dt / self.tau) * exp_tau
        term_d = (dt / self.theta) * exp_theta

        return self.a + self.b * exp_tau + self.c * term_c + self.d * term_d

    def df_dT(self, t, T):
        """
        Calculates the slope of the forward rate curve (df/dT).

        Args:
            t (float): Valuation date (in years).
            T (float or np.ndarray): Maturity date(s) (in years).

        Returns:
            float or np.ndarray: The derivative of the forward rate with respect to T.
        """
        dt = np.maximum(np.asanyarray(T) - t, 1e-9)

        exp_tau = np.exp(-dt / self.tau)
        exp_theta = np.exp(-dt / self.theta)

        slope_tau = (exp_tau / self.tau) * (self.c * (1 - dt / self.tau) - self.b)
        slope_theta = (exp_theta / self.theta) * (self.d * (1 - dt / self.theta))

        return slope_tau + slope_theta