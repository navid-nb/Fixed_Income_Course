"""
Zero-Coupon Yield extraction from market yield data using bootstrapping.
Conventions from: Canadian Treasury yields, May 8, 2024
"""
import numpy as np
from typing import Dict


class ZCYExtractor:
    
    def __init__(self, freq=2):
        """
        Initialize ZCY extractor.
        
        Args:
            freq (int): Coupon frequency per year (2 for semi-annual).
        """
        self.freq = freq
        self.P = {}    # Discount factors P(0, t)
        self.zcy = {}  # Zero-coupon yields (simple rates)
    
    def extract_zcy(self, money_market_yields: Dict[float, float], 
                par_yields: Dict[float, float]) -> Dict[float, float]:
        """
        Bootstrap zero-coupon yields from market data.
        
        Args:
            money_market_yields (dict): {maturity: simple_rate} for T-bills. E.g., {1/13: 0.0491, ...}
            par_yields (dict): {maturity: coupon_rate} for bonds. E.g., {1: 0.0466, 2: 0.0419, ...}
        
        Returns:
            dict: {maturity: zero_coupon_yield} - continuous compounding: zcy(t) = -ln(P(t)) / t
        """
        # Process money market rates: P(0,t) = 1 - r(t) * t
        for t, rate in sorted(money_market_yields.items()):
            self.P[t] = 1 - rate * t
            self.zcy[t] = -np.log(self.P[t]) / t  # Convert to continuous compounding
        
        # Bootstrap coupon bonds
        for T, coupon_rate in sorted(par_yields.items()):
            tau = 1.0 / self.freq  # Semi-annual: tau = 0.5
            n = int(round(T * self.freq))
            
            # Cashflow times: 0.5, 1.0, 1.5, ..., T
            times = np.array([(k + 1) * tau for k in range(n)])
            
            # Coupon per period (on $100 face value)
            c = coupon_rate * tau * 100
            
            # Find last known time before T
            known_times = sorted([t for t in self.P.keys() if t < T])
            t_k = known_times[-1] if known_times else None
            
            # Separate intermediate times (between t_k and T) from known times
            pv_known = 0.0  # PV of completely known cashflows
            coeff_P_T = 0.0  # Coefficient of P(T) in the equation
            
            for t in times[:-1]:  # All times except T itself
                if t in self.P:  # Known discount factor
                    pv_known += c * self.P[t]
                else:  # Intermediate time: interpolate with P(T)
                    # P(t) = P(t_k) + (t - t_k)/(T - t_k) * (P(T) - P(t_k))
                    beta = (t - t_k) / (T - t_k)
                    pv_known += c * self.P[t_k] * (1 - beta)
                    coeff_P_T += c * beta
            
            # Final cashflow (coupon + principal)
            coeff_P_T += c + 100
            
            # Solve linear equation: 100 = pv_known + coeff_P_T * P(T)
            self.P[T] = (100 - pv_known) / coeff_P_T
            self.zcy[T] = -np.log(self.P[T]) / T  # Convert to continuous compounding
        
        return self.zcy.copy(), self.P.copy()
    
    def get_discount_factors(self) -> Dict[float, float]:
        """Return all discount factors."""
        return self.P.copy()
    
    def get_zero_yields(self) -> Dict[float, float]:
        """Return all zero-coupon yields."""
        return self.zcy.copy()





if __name__ == "__main__":
    """Test the ZCYExtractor with Canadian Treasury yields (May 8, 2024)
    based on the sample numerical calculation provided in slides for Yield Curve Smoothing pages 8 to 12
    """
    
    # Money market rates (simple rates, up to and including 1 year)
    money_market_yields = {
        4/52: 0.0491,      # 4-week
        3/12: 0.0490,      # 3-month
        6/12: 0.0480,      # 6-month
        1.0: 0.0466,       # 1-year
    }
    
    # Coupon bond par yields (rest)
    par_yields = {
        2.0: 0.0419,       # 2-year
        3.0: 0.0406,       # 3-year
        5.0: 0.0368,       # 5-year
        7.0: 0.0362,       # 7-year
        10.0: 0.0363,      # 10-year
        30.0: 0.0351,      # 30-year
    }
    
    # Extract zero-coupon yields
    extractor = ZCYExtractor(freq=2)
    zcys, Ps = extractor.extract_zcy(money_market_yields, par_yields)
    
    # Display results
    print("=" * 60)
    print(f"{'Maturity (years)':<20} {'ZCY':<15} {'Discount Factor':<15}")
    print("-" * 60)
    
    for mat in sorted(zcys.keys()):
        print(f"{mat:<20.4f} {zcys[mat]:<15.6f} {Ps[mat]:<15.6f}")
        
    print("=" * 60)
    
