from abc import ABC, abstractmethod
import numpy as np

class BaseYieldCurve(ABC):
    """
    Abstract base class defining the interface for yield curve models.
    """
    @abstractmethod
    def zcy(self, t, T):
        pass

    @abstractmethod
    def f(self, t, T):
        pass
    
    def P(self, t, T):
        """
        Calculates the discount factor P(t, T) (Zero-Coupon Bond price).

        Args:
            t (float): Valuation date (in years).
            T (float or np.ndarray): Maturity date(s) (in years).

        Returns:
            float or np.ndarray: The discount factor value.
        """
        return np.exp(-self.zcy(t, T) * (T-t))