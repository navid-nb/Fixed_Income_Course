"""
Calibration tools for yield curve models.
"""
import numpy as np
from scipy.optimize import differential_evolution
from typing import Dict
from .nss import NelsonSiegelSvensson
import matplotlib.pyplot as plt


class NSS_Calibrator:
    
    def __init__(self,):
        self.fitted_curve = None
        self.params = None
        self.optimization_result = None
    
    def fit(self, zcy_dict: Dict[float, float], bounds: list, strategy= 'best1bin',
            verbose=True):
        
        # Convert dict to arrays
        sorted_items = sorted(zcy_dict.items())
        maturities = np.array([mat for mat, _ in sorted_items])
        yields = np.array([yld for _, yld in sorted_items])
        
        
        
        if verbose:
            print("Starting NSS calibration with Differential Evolution. (this may take a few moments)...")
        
        result = differential_evolution(
            self._objective,
            bounds=bounds,
            args=(maturities, yields),
            strategy= strategy,
            updating='deferred',
            maxiter=10000,
            popsize=100,
            seed=42,
            atol=1e-10,
            tol=1e-10,
            workers=-1
        )
        
        # Store results
        a, b, c, d, tau, theta = result.x
        self.fitted_curve = NelsonSiegelSvensson(a, b, c, d, tau, theta)
        self.params = {
            'a': a, 'b': b, 'c': c, 'd': d, 
            'tau': tau, 'theta': theta
        }
        self.optimization_result = {
            'success': result.success,
            'fun': result.fun,
            'nfev': result.nfev,
            'nit': result.nit if hasattr(result, 'nit') else None,
            'message': result.message,
            'sse': result.fun
        }
        
        if verbose:
            print(f"\nOptimization completed!")
            print(f"  Success: {result.success}")
            print(f"  Function evaluations: {result.nfev}")
            print(f"  Iterations: {self.optimization_result['nit']}")
            print(f"  Objective function value at optimum (SSE): {result.fun:.6e}")
            print(f"\nFitted Parameters:")
            print(f"  a = {a:.6f}")
            print(f"  b = {b:.6f}")
            print(f"  c = {c:.6f}")
            print(f"  d = {d:.6f}")
            print(f"  tau = {tau:.6f}")
            print(f"  theta = {theta:.6f}")
        
        return self
    
    def _objective(self, params, maturities, market_zcy):
        a, b, c, d, tau, theta = params
        
        if tau <= 0 or theta <= 0:
            return 1e10
    
        nss = NelsonSiegelSvensson(a, b, c, d, tau, theta)
        model_zcy = nss.zcy(0, maturities)
        return np.sum((model_zcy - market_zcy) ** 2)

    def plot_fit(self, zcy_dict: Dict[float, float]):
        T_sorted = np.array(sorted(zcy_dict.keys()))
        y_obs = np.array([zcy_dict[t] for t in T_sorted])

        # Create dense grid for smooth NSS curve
        T_smooth = np.linspace(0.1, 20, 200)
        y_smooth = self.fitted_curve.zcy(0, T_smooth)


        plt.figure(figsize=(12, 7))

        # Plot market data (discrete points)
        plt.scatter(T_sorted, y_obs, 
                color='red', s=50, alpha=0.7, label='Market Zero-Coupon Yields', zorder=3)

        # Plot NSS fitted curve (smooth line)
        plt.plot(T_smooth, y_smooth, 
                color='blue', linewidth=2.5, label='NSS Fitted Curve', zorder=2)

        # Labels and formatting
        plt.xlabel('Maturity (years)', fontsize=12, fontweight='bold')
        plt.ylabel('Zero-Coupon Yield', fontsize=12, fontweight='bold')
        plt.title('Nelson-Siegel-Svensson Yield Curve Fit\n(US Treasury Market Data, Sept 18, 2025)', 
                fontsize=13, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(fontsize=11, loc='best', framealpha=0.95)

        # Add text box with parameters
        params_text = f"""NSS Parameters:
        a = {self.params['a']:.6f}
        b = {self.params['b']:.6f}
        c = {self.params['c']:.6f}
        d = {self.params['d']:.6f}
        tau = {self.params['tau']:.6f}
        theta = {self.params['theta']:.6f}

        SSE = {self.optimization_result['sse']:.6e}"""

        plt.text(0.98, 0.35, params_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                family='monospace')

        plt.tight_layout()
        plt.show()