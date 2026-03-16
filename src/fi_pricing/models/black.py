import numpy as np
from scipy.stats import norm
from ..curves.base import BaseYieldCurve


class BlackCapModel:
    """
    Black (1976) Model for Cap and Floor pricing.

    cap(t_n) = M * sum_{j=alpha+1}^{n}
                    tau * P(0, t_j) * ( F(t_{j-1}, t_j) * Phi(d1)
                                        - K * Phi(d2) )

    with:
        d1  = (1/v_j) * ( ln(F(t_{j-1}, t_j) / K) + v_j^2 / 2 )    (Eq. 5.11)
        d2  = (1/v_j) * ( ln(F(t_{j-1}, t_j) / K) - v_j^2 / 2 )    (Eq. 5.12)
        v_j = sigma_cap * sqrt(t_{j-1} - t)                           (Eq. 5.13)

    where:
        - sigma_cap is the constant Black implied volatility quoted on the market
        - t_{j-1} is the reset date of caplet j  (in years from the curve-fitting date)
        - t       is the current pricing date
        - F(t_{j-1}, t_j) is the simply-compounded forward rate implied by the curve
        - P(0, t_j) are market ZCB prices from the fitted NSS curve

    Market convention (as noted in teaching notes):
        Caps are quoted with quarterly settlement (tau = 0.25) and the
        first reset date at t_0 = 0.25 (alpha=0, t_0=0.25, t_1=0.50, ...).

    Parameters
    ----------
    curve : BaseYieldCurve — the fitted initial (NSS) yield curve, provides P(0, .)
    """

    def __init__(self, curve: BaseYieldCurve):
        self.curve = curve   # P(0, .) — market ZCB prices


    def forward_rate(self, t_reset: float, t_pay: float) -> float:
        """
        Simply-compounded forward rate F(t_{j-1}, t_j) implied by the market curve.

            F(t_{j-1}, t_j) = (1 / tau) * ( P(0, t_{j-1}) / P(0, t_j) - 1 )

        Parameters
        ----------
        t_reset : float — reset date t_{j-1}
        t_pay   : float — payment date t_j
        """
        tau   = t_pay - t_reset
        Pm_reset = self.curve.P(0.0, t_reset)
        Pm_pay   = self.curve.P(0.0, t_pay)
        return (1.0 / tau) * (Pm_reset / Pm_pay - 1.0)

  
    def caplet_price(
        self,
        t: float,
        t_reset: float,
        t_pay: float,
        K: float,
        sigma_cap: float,
        nominal: float = 1.0,
    ) -> float:
        """
        Black price of a single caplet (one term of Eq. 5.10).

            caplet_j = M * tau * P(0, t_j) * ( F * Phi(d1) - K * Phi(d2) )

        Parameters
        ----------
        t         : float — current pricing time (years from curve-fitting date)
        t_reset   : float — caplet reset date t_{j-1}
        t_pay     : float — caplet payment date t_j
        K         : float — cap strike rate
        sigma_cap : float — Black flat implied volatility (e.g. 0.20 for 20%)
        nominal   : float — notional amount M (default 1.0)
        """
        tau = t_pay - t_reset
        F   = self.forward_rate(t_reset, t_pay)
        P  = self.curve.P(0.0, t_pay)

        # time to expiry of this caplet's option (Eq. 5.13)
        tau_opt = t_reset - t
        tau_opt = max(tau_opt, 1e-9)  # ensure no division by zero 

        v_j = sigma_cap * np.sqrt(tau_opt)                # Eq. 5.13

        d1 = (np.log(F / K) + 0.5 * v_j**2) / v_j        # Eq. 5.11
        d2 = d1 - v_j                                      # Eq. 5.12

        return nominal * tau * P * (F * norm.cdf(d1) - K * norm.cdf(d2))

  

    def cap_price(
        self,
        t: float,
        payment_dates: np.ndarray,
        K: float,
        sigma_cap: float,
        nominal: float = 1.0,
    ) -> float:
        """
        Black price of a cap as a sum of caplets (Eq. 5.10).

            cap(t_n) = M * sum_{j=alpha+1}^{n}
                           tau * P(0, t_j) * ( F(t_{j-1}, t_j) * Phi(d1) - K * Phi(d2) )

        Parameters
        ----------
        t             : float      — current pricing time
        payment_dates : np.ndarray — settlement dates [t_1, t_2, ..., t_n]
                                     (first reset date is inferred as t)
        K             : float      — cap strike rate
        sigma_cap     : float      — flat Black implied volatility
        nominal       : float      — notional M
        """
        payment_dates = np.asarray(payment_dates)

        # reset dates: t, t_1, t_2, ..., t_{n-1}
        reset_dates = np.concatenate(([t], payment_dates[:-1]))

        total = 0.0
        for t_reset, t_pay in zip(reset_dates, payment_dates):
            total += self.caplet_price(t, t_reset, t_pay, K, sigma_cap, nominal)

        return total

    # ------------------------------------------------------------------
    # Forward swap rate  (Eq. 5.15 / 1.49)
    # ------------------------------------------------------------------

    def forward_swap_rate(
        self,
        t_alpha: float,
        payment_dates: np.ndarray,
    ) -> float:
        """
        Forward swap rate K_f (Eq. 5.15):

            K_f = ( P(0, t_alpha) - P(0, t_n) )
                  / sum_{j=alpha+1}^{n}  tau_j * P(0, t_j)

        Parameters
        ----------
        t_alpha       : float      — first reset / start date of the swap
        payment_dates : np.ndarray — payment dates [t_{alpha+1}, ..., t_n]
        """
        payment_dates = np.asarray(payment_dates)
        reset_dates   = np.concatenate(([t_alpha], payment_dates[:-1]))
        taus          = payment_dates - reset_dates

        P_alpha  = self.curve.P(0.0, t_alpha)
        P_n      = self.curve.P(0.0, payment_dates[-1])
        annuity  = np.sum(taus * np.array([self.curve.P(0.0, tj) for tj in payment_dates]))

        return (P_alpha - P_n) / annuity
