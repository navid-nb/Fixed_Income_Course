from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import brentq

class OneFactorModel(ABC):
    """
    Abstract base class for one-factor affine short rate models (e.g., Vasicek, CIR).
    """

    @abstractmethod
    def A(self, t, T):
        """
        Calculates the deterministic coefficient A(t, T) for the affine bond price formula.
        
        Args:
            t (float): Valuation date (in years).
            T (float or np.ndarray): Maturity date(s) of the bond (in years).
            
        Returns:
            float or np.ndarray: The coefficient A(t, T).
        """
        pass

    @abstractmethod
    def B(self, t, T):
        """
        Calculates the deterministic coefficient B(t, T) for the affine bond price formula.
        
        Args:
            t (float): Valuation date (in years).
            T (float or np.ndarray): Maturity date(s) of the bond (in years).
            
        Returns:
            float or np.ndarray: The coefficient B(t, T).
        """
        pass

    @abstractmethod
    def zcb_option(self, t, T, rt, T_expiry, K, option_type="call"):
        """
        Prices a European option on a Zero-Coupon Bond using the model's closed-form solution.
        
        Args:
            t (float): Valuation date.
            T (float or np.ndarray): Maturity date(s) of the underlying Zero-Coupon Bond.
            rt (float): Current instantaneous short rate.
            T_expiry (float): Expiry date of the option.
            K (float or np.ndarray): Strike price of the option.
            option_type (str): "call" or "put".
            
        Returns:
            float or np.ndarray: The price of the option.
        """
        pass

    def P(self, t, T, rt):
        """Calculates the price of a Zero-Coupon Bond at time t with maturity T given short rate rt."""
        return self.A(t, T) * np.exp(-self.B(t, T) * rt)

    def zcy(self, t, T, rt):
        """
        Calculates the Zero-Coupon Yield (Spot Rate) for a given maturity.
        
        Args:
            t (float): Valuation date.
            T (float or np.ndarray): Maturity date(s).
            rt (float): Current instantaneous short rate.
            
        Returns:
            float or np.ndarray: The annualized continuously compounded yield.
        """
        dt = np.maximum(T - t, 1e-9)
        return -1/dt * np.log(self.P(t, T, rt))
    
    def coupon_bond_price(self, t, cash_flows, payment_dates, rt):
        """
        Calculates the price of a coupon-bearing bond as the sum of discounted cash flows.
        
        Args:
            t (float): Valuation date.
            cash_flows (np.ndarray): Array of coupon payments and principal.
            payment_dates (np.ndarray): Array of payment dates corresponding to cash_flows.
            rt (float): Current instantaneous short rate.   

        Returns:
            float: The price of the coupon-bearing bond.
        """        
        cash_flows = np.asanyarray(cash_flows)
        payment_dates = np.asanyarray(payment_dates)
        
        return np.sum(cash_flows * self.P(t, payment_dates, rt))
    

    def coupon_bond_option(self, t, T_expiry, rt, K_bond, 
                           cash_flows, payment_dates, 
                           option_type="call"):
        """
        Prices a European option on a coupon-bearing bond using Jamshidian's Decomposition.
        
        This method decomposes the option on a coupon bond into a sum of options 
        on Zero-Coupon Bonds (ZCB), determined by a critical rate r*.
        
        Args:
            t (float): Valuation date.
            T_expiry (float): Expiry date of the option.
            rt (float): Current instantaneous short rate.
            K_bond (float): Strike price relative to the whole coupon bond.
            cash_flows (np.ndarray): Array of coupon payments and principal.
            payment_dates (np.ndarray): Array of payment dates corresponding to cash_flows.
            option_type (str): "call" or "put".
            
        Returns:
            float: The price of the coupon bond option.
        """
        payment_dates = np.asanyarray(payment_dates)
        cash_flows = np.asanyarray(cash_flows)
        
        # Filter to only include future cash flows (those at or after T_expiry)
        future_mask = payment_dates >= T_expiry
        future_payment_dates = payment_dates[future_mask]
        future_cash_flows = cash_flows[future_mask]

        def objective(r):
            prices = self.P(T_expiry, future_payment_dates, r)
            return np.dot(prices, future_cash_flows) - K_bond

        try:
            r_star = brentq(objective, -0.9, 2.0)
        except ValueError:
            print("Error : Impossible to find r* for Jamshidian's Decomposition.")
            return np.nan
        print(f"Found r* = {r_star:.6f} for Jamshidian's Decomposition.")
        print( objective(r_star), " ....", objective(0.0100905) )

        strikes_Ki = self.P(T_expiry, future_payment_dates, r_star)
        option_prices = self.zcb_option(t, future_payment_dates, rt, T_expiry, strikes_Ki, option_type=option_type)

        return future_cash_flows @ option_prices

    def rate_option(self, t, rt, start_date, payment_dates, K_rate, nominal, option_type="cap"):
        """
        Prices a Cap or Floor product as a portfolio of Caplets/Floorlets (Options on ZCB).
        
        Args:
            t (float): Valuation date.
            rt (float): Current instantaneous short rate.
            start_date (float): Start date of the contract (first reset date).
            payment_dates (np.ndarray): Array of payment dates [T1, T2, ... Tn].
            K_rate (float): The strike rate (e.g., 0.03 for 3%).
            nominal (float): The notional amount.
            option_type (str): "cap" for Cap (protection against rise), "floor" for Floor (protection against fall).
            
        Returns:
            float: The total price of the Cap or Floor.
        """
        payment_dates = np.asanyarray(payment_dates)

        if start_date >= payment_dates[0]:
            raise ValueError("Start date must be strictly before the first payment date.")
        
        reset_dates = np.concatenate(([start_date], payment_dates[:-1]))
        
        future_mask = reset_dates > t
        
        if not np.any(future_mask):
            return 0.0
            
        eff_resets = reset_dates[future_mask]
        eff_payments = payment_dates[future_mask] 
        
        deltas = eff_payments - eff_resets
        
        K_bonds = 1.0 / (1.0 + deltas * K_rate)
        
        zcb_option_type = "put" if option_type.lower() == "cap" else "call"
        
        option_prices = self.zcb_option(t, eff_payments, rt, eff_resets, K_bonds, option_type=zcb_option_type)
        
        scale_factors = nominal * (1.0 + deltas * K_rate)
        
        return np.sum(option_prices * scale_factors)
    
    def coupon_bond_cashflow_calculator(self, coupon, maturity, frequency, face_value):
        """
        Calculates the cash flows and payment dates for a typical coupon-bearing bond.
        
        Args:
            coupon (float): The annual coupon rate (e.g., 0.05 for 5%).
            maturity (float): The maturity of the bond in years.
            frequency (int): The number of coupon payments per year (e.g., 2 for semi-annual).
            face_value (float): The face value of the bond (e.g., 1000).
            
        Returns:
            np.ndarray: An array of cash flows corresponding to each payment date.
            np.ndarray: An array of payment dates corresponding to each cash flow.
        """
        num_payments = int(maturity * frequency)
        cash_flows = np.full(num_payments, coupon * face_value / frequency)
        cash_flows[-1] += face_value  # Add principal to the last payment
        payment_dates = np.array([ (i + 1) / frequency for i in range(num_payments)])
        return cash_flows, payment_dates
