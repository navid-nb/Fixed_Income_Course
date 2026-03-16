import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from src.fi_pricing.models import TwoFactorGaussianModel

_STRIKE_LABELS = ["0.85·K_f (ITM)", "K_f (ATM)", "1.15·K_f (OTM)"]


def cap_price_objective(params, curve, mkt_prices, payment_dates_list, strikes, t):
    """Sum of squared relative cap-price errors for the Two-Factor Gaussian model."""
    a, b, sigma, eta, rho, x_t, y_t = params
    model = TwoFactorGaussianModel(a=a, b=b, sigma=sigma, eta=eta, rho=rho,
                                   curve=curve, x_t=x_t, y_t=y_t)
    
    # Vectorized batch pricing: (n_maturities, n_strikes)
    model_prices = model.cap_prices_batch(t, payment_dates_list, strikes)
    
    # Compute squared relative errors
    rel_err = (model_prices - mkt_prices) / mkt_prices
    return np.sum(rel_err ** 2)


def plot_cap_prices(maturities, mkt_prices, model_prices, strikes, save_path=None):
    """Market vs model cap prices — one subplot per strike."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Cap Prices: Market vs Two-Factor Gaussian", fontsize=13)
    for j, (ax, lbl) in enumerate(zip(axes, _STRIKE_LABELS)):
        ax.plot(maturities, mkt_prices[:, j],  "o-",  label="Market", color="steelblue")
        ax.plot(maturities, model_prices[:, j], "s--", label="2F Gaussian", color="tomato")
        ax.set_title(f"{lbl}  ({strikes[j]:.3%})")
        ax.set_xlabel("Maturity (years)"); ax.set_ylabel("Cap price"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_implied_vols(maturities, imp_vols, model_vols, strikes, save_path=None):
    """Market vs model implied vols — one subplot per strike."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Implied Vols: Market vs Two-Factor Gaussian", fontsize=13)
    for j, (ax, lbl) in enumerate(zip(axes, _STRIKE_LABELS)):
        ax.plot(maturities, imp_vols[:, j]  * 100, "o-",  label="Market", color="steelblue")
        ax.plot(maturities, model_vols[:, j] * 100, "s--", label="2F Gaussian", color="tomato")
        ax.set_title(f"{lbl}  ({strikes[j]:.3%})")
        ax.set_xlabel("Maturity (years)"); ax.set_ylabel("Implied vol (%)"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
