import numpy as np
from scipy.stats import norm
from ..curves.base import BaseYieldCurve


class TwoFactorGaussianModel:
    """
    Two-Factor Gaussian Model.

    Dynamics under the EMM:
        dx_t = -a * x_t * dt + sigma * dW_{1t}
        dy_t = -b * y_t * dt + eta   * dW_{2t}

    Instantaneous risk-free rate:
        r_t = x_t + y_t + phi(t)

    where phi(t) is deterministic and chosen to fit the initial yield curve exactly.

    At the curve-fitting date (time 0): x_0 = y_0 = 0 since phi fits the initial curve.

    Parameters
    ----------
    a     : float  — mean-reversion speed of x
    b     : float  — mean-reversion speed of y
    sigma : float  — volatility of x
    eta   : float  — volatility of y
    rho   : float  — instantaneous correlation between W_1 and W_2  (-1 < rho < 1)
    curve : BaseYieldCurve — the fitted initial (NSS) yield curve P^m(0, .)
    x_t   : float  — current value of factor x
    y_t   : float  — current value of factor y
    """

    def __init__(
        self,
        a: float,
        b: float,
        sigma: float,
        eta: float,
        rho: float,
        curve: BaseYieldCurve,
        x_t: float = 0.0,
        y_t: float = 0.0,
    ):
        self.a = a
        self.b = b
        self.sigma = sigma
        self.eta = eta
        self.rho = rho
        self.curve = curve   # P^m(0, .) — the fitted initial curve
        self.x_t = x_t
        self.y_t = y_t


    def _V(self, t: float, T):
        """
        Variance of the integral int_t^T (x_u + y_u) du  (Eq. 4.5).
        
        Works with scalar or array T due to numpy broadcasting.
        """
        a, b, s, e, r = self.a, self.b, self.sigma, self.eta, self.rho
        tau = np.asarray(T) - t

        term_1 = (s**2 / a**2) * (
            tau
            + (2.0 / a) * np.exp(-a * tau)
            - (1.0 / (2.0 * a)) * np.exp(-2.0 * a * tau)
            - 3.0 / (2.0 * a)
        )

        term_2 = (e**2 / b**2) * (
            tau
            + (2.0 / b) * np.exp(-b * tau)
            - (1.0 / (2.0 * b)) * np.exp(-2.0 * b * tau)
            - 3.0 / (2.0 * b)
        )

        term_3 = (2.0 * r * s * e / (a * b)) * (
            tau
            + (np.exp(-a * tau) - 1.0) / a
            + (np.exp(-b * tau) - 1.0) / b
            - (np.exp(-(a + b) * tau) - 1.0) / (a + b)
        )

        return term_1 + term_2 + term_3

    def _Bx(self, t: float, T):
        """(1 - e^{-a(T-t)}) / a  — x-factor loading. Works with arrays."""
        tau = np.asarray(T) - t
        return (1.0 - np.exp(-self.a * tau)) / self.a

    def _By(self, t: float, T):
        """(1 - e^{-b(T-t)}) / b  — y-factor loading. Works with arrays."""
        tau = np.asarray(T) - t
        return (1.0 - np.exp(-self.b * tau)) / self.b

    def _B_full(self, t: float, T):
        """
        Exponent in the ZCB formula (Eq. 4.4). Works with arrays.

        B(t, T) = 1/2 * [V(t,T) - V(0,T) + V(0,t)]
                  - Bx(t,T) * x_t
                  - By(t,T) * y_t
        """
        half_V = 0.5 * (self._V(t, T) - self._V(0.0, T) + self._V(0.0, t))
        return half_V - self._Bx(t, T) * self.x_t - self._By(t, T) * self.y_t



    def P(self, t: float, T) -> float:
        """
        Zero-coupon bond price at time t for maturity T, given state (x_t, y_t).

        P_{2fg}(t, T) = P^m(0, T) / P^m(0, t) * exp( B(t, T) )       (Eq. 4.3)

        Parameters
        ----------
        t : float — current evaluation time (0 = curve-fitting date)
        T : float or array — bond maturity/maturities
        
        Returns
        -------
        float or array — bond price(s)
        """
        T = np.asarray(T)
        scalar_input = T.ndim == 0
        
        Pm_0T = self.curve.P(0.0, T)
        Pm_0t = self.curve.P(0.0, t)
        
        # _B_full already works with arrays via numpy broadcasting
        B_vals = self._B_full(t, T)
        
        result = (Pm_0T / Pm_0t) * np.exp(B_vals)
        return float(result) if scalar_input else result

    def zcy(self, t: float, T: float) -> float:
        """Continuously compounded zero-coupon yield for maturity T at time t."""
        tau = max(T - t, 1e-9)
        return -np.log(self.P(t, T)) / tau



    def _sigma_P(self, t: float, T_call, T_bond) -> float:
        """
        Bond return standard deviation for the ZCB option formula (Eq. 4.8).

        sigma_P^2 =   sigma^2 / (2a^3) * (1 - e^{-a(T_bond - T_call)})^2 * (1 - e^{-2a*(T_call - t)})
                    + eta^2   / (2b^3) * (1 - e^{-b(T_bond - T_call)})^2 * (1 - e^{-2b*(T_call - t)})
                    + 2*rho*sigma*eta / (ab(a+b))
                      * (1 - e^{-a(T_bond-T_call)}) * (1 - e^{-b(T_bond-T_call)}) * (1 - e^{-(a+b)*(T_call-t)})

        Parameters
        ----------
        t      : float — current time
        T_call : float or array — option expiry/expiries
        T_bond : float or array — bond maturity/maturities
        
        Returns
        -------
        float or array — bond volatility/volatilities
        """
        T_call = np.asarray(T_call)
        T_bond = np.asarray(T_bond)
        
        a, b, s, e, r = self.a, self.b, self.sigma, self.eta, self.rho
        tau_opt  = T_call - t           # time to option expiry from now
        tau_bond = T_bond - T_call      # bond life remaining after option expiry

        term1 = (s**2 / (2.0 * a**3)) * (
            (1.0 - np.exp(-a * tau_bond))**2
            * (1.0 - np.exp(-2.0 * a * tau_opt))
        )

        term2 = (e**2 / (2.0 * b**3)) * (
            (1.0 - np.exp(-b * tau_bond))**2
            * (1.0 - np.exp(-2.0 * b * tau_opt))
        )

        term3 = (2.0 * r * s * e / (a * b * (a + b))) * (
            (1.0 - np.exp(-a * tau_bond))
            * (1.0 - np.exp(-b * tau_bond))
            * (1.0 - np.exp(-(a + b) * tau_opt))
        )

        sigma_p2 = term1 + term2 + term3
        return np.sqrt(np.maximum(sigma_p2, 1e-18))



    def zcb_option(
        self,
        t: float,
        T_call: float,
        T_bond: float,
        K: float,
        option_type: str = "call",
    ) -> float:
        """
        European option on a zero-coupon bond  (Eq. 4.6).

        call_{2fg}(P, K) = P_{2fg}(t, T_bond) * Phi(h)
                         - K * P_{2fg}(t, T_call) * Phi(h - sigma_P)     (Eq. 4.6)

        h = sigma_P / 2 + (1 / sigma_P) * ln( P_{2fg}(t, T_bond)
                                             / (K * P_{2fg}(t, T_call)) ) (Eq. 4.7)

        Parameters
        ----------
        t      : float — current evaluation time
        T_call : float — option expiry (calendar time)
        T_bond : float — ZCB maturity (calendar time), T_bond > T_call
        K      : float — strike price on the ZCB
        option_type : "call" or "put"
        """
        P_bond  = self.P(t, T_bond)
        P_call  = self.P(t, T_call)
        sigma_p = self._sigma_P(t, T_call, T_bond)

        h = (sigma_p / 2.0
             + np.log(P_bond / (K * P_call)) / sigma_p)

        if option_type.lower() == "call":
            return P_bond * norm.cdf(h) - K * P_call * norm.cdf(h - sigma_p)
        elif option_type.lower() == "put":
            return K * P_call * norm.cdf(sigma_p - h) - P_bond * norm.cdf(-h)
        else:
            raise ValueError("option_type must be 'call' or 'put'")


    def caplet_price(
        self,
        t: float,
        T_reset: float,
        T_pay: float,
        K_rate: float,
        nominal: float = 1.0,
    ) -> float:
        """
        Price of a single caplet (Eq. 1.38 – 1.40).

        A caplet with reset T_reset and payment T_pay is a put on the ZCB P(T_reset, T_pay):

            Caplet(T_reset, T_pay, K) = M' * Put( P(T_reset, T_pay), K' )

        where:
            tau = T_pay - T_reset      (accrual period)
            K'  = 1 / (1 + K_rate * tau)   (ZCB strike, Eq. 1.39)
            M'  = nominal * (1 + K_rate * tau)   (scaled nominal, Eq. 1.40)
        """
        tau    = T_pay - T_reset
        K_prime = 1.0 / (1.0 + K_rate * tau)
        M_prime = nominal * (1.0 + K_rate * tau)
        return M_prime * self.zcb_option(t, T_reset, T_pay, K_prime, option_type="put")

    def cap_price(
        self,
        t: float,
        payment_dates: np.ndarray,
        K_rate: float,
        nominal: float = 1.0,
    ) -> float:
        """
        Price of a cap as a chain of caplets (Eq. 1.40).
        Assumes the first reset date is the current time t. 

        Cap(M, K) = sum_{j=alpha+1}^{n}  M' * Put( P(t_{j-1}, t_j), K' )

        Parameters
        ----------
        t             : float — current evaluation time
        payment_dates : array of settlement dates t_1, t_2, ..., t_n  (calendar years)
                        The first reset date is t (current time).
        K_rate        : fixed strike rate K
        nominal       : notional M
        """
        total = 0.0
        # reset dates: current time t, then all payment dates except the last
        reset_dates = np.concatenate(([t], payment_dates[:-1]))

        for T_reset, T_pay in zip(reset_dates, payment_dates):
            # skip caplets that have already reset (rate is known, no optionality)
            # or are degenerate (reset >= pay)
            if T_reset <= t + 1e-9 or T_reset >= T_pay:
                continue
            total += self.caplet_price(t, T_reset, T_pay, K_rate, nominal)

        return total

    def cap_prices_batch(
        self,
        t: float,
        payment_dates_list: list,
        strikes: np.ndarray,
        nominal: float = 1.0,
    ) -> np.ndarray:
        """
        Vectorized batch cap pricing for multiple maturities and strikes.
        
        Parameters
        ----------
        t                   : evaluation time
        payment_dates_list  : list of payment date arrays, one per maturity
        strikes             : 1D array of strike rates
        nominal             : notional
        
        Returns
        -------
        prices : (n_maturities, n_strikes) array of cap prices
        """
        n_mat = len(payment_dates_list)
        n_str = len(strikes)
        prices = np.zeros((n_mat, n_str))
        
        # Vectorize over maturities
        for i, pay_dates in enumerate(payment_dates_list):
            reset_dates = np.concatenate(([t], pay_dates[:-1]))
            mask = (reset_dates > t + 1e-9) & (reset_dates < pay_dates)
            valid_resets = reset_dates[mask]
            valid_pays = pay_dates[mask]
            
            if len(valid_resets) == 0:
                continue
            
            # Compute bond prices and vols ONCE per maturity (vectorized)
            P_bond = self.P(t, valid_pays)
            P_call = self.P(t, valid_resets)
            sigma_p = self._sigma_P(t, valid_resets, valid_pays)
            
            # Now vectorize across strikes: (n_caplets, n_strikes)
            tau = valid_pays - valid_resets  # (n_caplets,)
            strikes_2d = strikes[np.newaxis, :]  # (1, n_strikes)
            tau_2d = tau[:, np.newaxis]  # (n_caplets, 1)
            
            K_prime = 1.0 / (1.0 + strikes_2d * tau_2d)  # (n_caplets, n_strikes)
            M_prime = nominal * (1.0 + strikes_2d * tau_2d)
            
            # Broadcast bond prices and vols
            P_bond_2d = P_bond[:, np.newaxis]  # (n_caplets, 1)
            P_call_2d = P_call[:, np.newaxis]
            sigma_p_2d = sigma_p[:, np.newaxis]
            
            h = sigma_p_2d / 2.0 + np.log(P_bond_2d / (K_prime * P_call_2d)) / sigma_p_2d
            put_values = K_prime * P_call_2d * norm.cdf(sigma_p_2d - h) - P_bond_2d * norm.cdf(-h)
            
            prices[i, :] = np.sum(M_prime * put_values, axis=0)  # Sum over caplets
        
        return prices

    def forward_swap_rate(
        self,
        t_alpha: float,
        payment_dates: np.ndarray,
    ) -> float:
        """
        Forward swap rate K_f (Eq. 1.49).

        K_f = ( P(0, t_alpha) - P(0, t_n) )
              / sum_{j=alpha+1}^{n}  tau_j * P(0, t_j)

        where:
            t_alpha       — first reset date of the swap
            payment_dates — settlement dates t_{alpha+1}, ..., t_n
            tau_j         — accrual period t_j - t_{j-1}  (year fraction)

        Parameters
        ----------
        t_alpha       : float      — first reset / start date of the swap
        payment_dates : np.ndarray — array of payment dates [t_{alpha+1}, ..., t_n]

        Returns
        -------
        float — the forward swap rate K_f
        """
        payment_dates = np.asarray(payment_dates)
        t_n = payment_dates[-1]

        # accrual periods tau_j = t_j - t_{j-1}
        reset_dates = np.concatenate(([t_alpha], payment_dates[:-1]))
        taus = payment_dates - reset_dates

        # market ZCB prices from the fitted curve (all anchored at time 0)
        P_t_alpha = self.curve.P(0.0, t_alpha)
        P_t_n     = self.curve.P(0.0, t_n)
        denominator   = np.sum(taus * np.array([self.curve.P(0.0, tj) for tj in payment_dates]))

        return (P_t_alpha - P_t_n) / denominator
