"""
Part 2: Assignment 1 — NSS and CIR Analysis.

This module consolidates all Part 2 functionality:
  1. Load NSS coefficients from Excel
  2. Build weekly zero-coupon yields
  3. Compute descriptive statistics
  4. Fit CIR model via Extended Kalman Filter + MLE
  5. Plot and compare short-rate paths

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution

from fi_pricing.curves.nss import NelsonSiegelSvensson
from fi_pricing.models.affine import CIRModel


# ============================================================================
# Configuration
# ============================================================================

@dataclass(frozen=True)
class Part2Config:
    """Runtime configuration for Part 2."""

    repo_root: Path
    excel_path: Path
    sheet_name: str = "TP1 data"

    # Required maturities (years) for Part 2.1
    taus_years: tuple[float, ...] = (0.25, 0.5, 1.0, 3.0, 5.0, 10.0, 30.0)
    labels: tuple[str, ...] = ("3M", "6M", "1Y", "3Y", "5Y", "10Y", "30Y")

    out_dirname: str = "part2_outputs"

    # EKF / MLE
    max_iter: int = 400
    ftol: float = 1e-10

    @property
    def out_dir(self) -> Path:
        return self.repo_root / self.out_dirname


# ============================================================================
# NSS Coefficients Loading
# ============================================================================

REQUIRED_COLUMNS = {
    "Date",
    "BETA0",
    "BETA1",
    "BETA2",
    "BETA3",
    "TAU1",
    "TAU2",
}


@dataclass(frozen=True)
class NSSCoefficients:
    """Container for NSS coefficients time series."""

    df: pd.DataFrame

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.to_datetime(self.df["Date"])  # type: ignore[return-value]


@dataclass(frozen=True)
class WeeklyYields:
    """Weekly zero-coupon yields built from NSS coefficients."""

    df: pd.DataFrame  # columns: Date + labels

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.to_datetime(self.df["Date"])  # type: ignore[return-value]


def _detect_beta_scale_percent(nss_df: pd.DataFrame) -> float:
    """Detect whether NSS betas are in percent or decimal units."""
    beta0_med_abs = float(np.median(np.abs(nss_df["BETA0"].to_numpy(dtype=float))))
    if beta0_med_abs < 0.5:
        return 100.0

    if beta0_med_abs > 50.0:
        raise ValueError(
            f"Unit sanity check failed: median |BETA0|={beta0_med_abs:.3g}. "
            "This looks far too large for percent units. Verify the Excel file and units."
        )

    return 1.0


def load_nss_coefficients(excel_path: Path, sheet_name: str) -> NSSCoefficients:
    """Load St. Louis Fed NSS coefficient time series from Excel."""
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            "Excel sheet is missing required columns: "
            + ", ".join(sorted(missing))
            + f". Found columns: {list(df.columns)}"
        )

    # Basic cleaning
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.dropna(subset=["Date"])

    # Assert numeric columns are numeric
    for col in REQUIRED_COLUMNS - {"Date"}:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[list(REQUIRED_COLUMNS - {"Date"})].isna().any().any():
        bad = df[df[list(REQUIRED_COLUMNS - {"Date"})].isna().any(axis=1)].head(5)
        raise ValueError(
            "Found NaNs after numeric coercion in NSS coefficient columns. "
            "Example bad rows (first 5):\n" + bad.to_string(index=False)
        )

    return NSSCoefficients(df=df)


# ============================================================================
# NSS Weekly Yields
# ============================================================================

def _validate_tau_units(nss_df: pd.DataFrame) -> None:
    """Validate tau coefficients look like years and are strictly positive."""
    tau1 = nss_df["TAU1"].to_numpy(dtype=float)
    tau2 = nss_df["TAU2"].to_numpy(dtype=float)

    tau1_min = float(np.min(tau1))
    tau2_min = float(np.min(tau2))
    if tau1_min <= 0.0 or tau2_min <= 0.0:
        raise ValueError(
            f"Invalid TAU values: min(TAU1)={tau1_min:.6g}, min(TAU2)={tau2_min:.6g}. "
            "TAU1 and TAU2 must be strictly positive."
        )

    tau1_med = float(np.median(tau1))
    tau2_med = float(np.median(tau2))
    if tau1_med > 100.0 or tau2_med > 100.0:
        raise ValueError(
            f"Unit sanity check failed: TAU medians look huge (TAU1~{tau1_med:.3g}, TAU2~{tau2_med:.3g}). "
            "TAU parameters are in years and should be single-digit numbers. Check for a unit bug."
        )


def build_weekly_zero_coupon_yields(
    nss_df: pd.DataFrame,
    taus_years: tuple[float, ...],
    labels: tuple[str, ...],
) -> WeeklyYields:
    """Construct the weekly ZC yield series for the requested maturities."""
    if len(taus_years) != len(labels):
        raise ValueError("taus_years and labels must have same length")

    taus = np.array(taus_years, dtype=float)

    if float(np.max(taus)) > 30.0 + 1e-12:
        raise ValueError(
            f"Extrapolation check failed: requested max maturity={float(np.max(taus)):.4g} years. "
            "Part 2 only needs maturities up to 30 years; reduce taus_years."
        )

    beta_scale = _detect_beta_scale_percent(nss_df)
    _validate_tau_units(nss_df)

    out = pd.DataFrame({"Date": pd.to_datetime(nss_df["Date"])})

    y_mat = np.empty((len(nss_df), len(taus)), dtype=float)

    for i, row in enumerate(nss_df.itertuples(index=False)):
        curve = NelsonSiegelSvensson(
            a=beta_scale * float(getattr(row, "BETA0")),
            b=beta_scale * float(getattr(row, "BETA1")),
            c=beta_scale * float(getattr(row, "BETA2")),
            d=beta_scale * float(getattr(row, "BETA3")),
            tau=float(getattr(row, "TAU1")),
            theta=float(getattr(row, "TAU2")),
        )
        y_mat[i, :] = curve.zcy(0.0, taus)  # type: ignore[assignment]

    for j, lab in enumerate(labels):
        out[lab] = y_mat[:, j]

    # Output sanity checks
    y_min = float(np.nanmin(y_mat))
    y_max = float(np.nanmax(y_mat))
    if y_min < -10.0 or y_max > 35.0:
        raise ValueError(
            "Yield sanity check failed: computed weekly yields contain values outside a plausible range. "
            f"Observed min={y_min:.2f}%, max={y_max:.2f}%. "
            "This usually indicates a percent/decimal mismatch or unstable extrapolation."
        )

    if "10Y" in out.columns:
        med_10y = float(out["10Y"].median())
        if med_10y > 25.0 or med_10y < -5.0:
            raise ValueError(
                f"Unit sanity check failed: median 10Y yield={med_10y:.2f}% looks inconsistent. "
                "Check whether the NSS betas are expressed in percent or decimals."
            )

    return WeeklyYields(df=out)


def nss_instantaneous_short_rate_percent(nss_df: pd.DataFrame) -> np.ndarray:
    """NSS short rate in percent: beta0 + beta1."""
    beta_scale = _detect_beta_scale_percent(nss_df)
    beta0 = nss_df["BETA0"].to_numpy(dtype=float)
    beta1 = nss_df["BETA1"].to_numpy(dtype=float)
    return beta_scale * (beta0 + beta1)


# ============================================================================
# Descriptive Statistics
# ============================================================================

def descriptive_stats_table(yields_df: pd.DataFrame, labels: tuple[str, ...]) -> pd.DataFrame:
    """Return mean, std, and quartiles for each maturity column."""
    rows: list[dict[str, float | str]] = []
    for lab in labels:
        s = yields_df[lab]
        rows.append(
            {
                "Maturity": lab,
                "Mean": float(s.mean()),
                "Std": float(s.std(ddof=1)),
                "Q1": float(s.quantile(0.25)),
                "Median": float(s.quantile(0.50)),
                "Q3": float(s.quantile(0.75)),
            }
        )
    return pd.DataFrame(rows)


# ============================================================================
# CIR EKF + MLE Estimation
# ============================================================================

class CIRLoadings(NamedTuple):
    a: np.ndarray
    b: np.ndarray

@dataclass(frozen=True)
class CIRFitResult:
    success: bool
    message: str
    kappa: float
    theta: float
    lambda_risk: float
    sigma: float
    R_diag: np.ndarray
    max_loglik: float
    se_params: np.ndarray

def cir_ab(tau: float, kappa: float, theta: float, sigma: float) -> tuple[float, float]:
    g = np.sqrt(kappa**2 + 2.0 * sigma**2)
    num_a = 2.0 * g * np.exp((kappa + g) * tau / 2.0)
    den = (kappa + g) * (np.exp(g * tau) - 1.0) + 2.0 * g
    a_val = (num_a / den) ** (2.0 * kappa * theta / sigma**2)
    b_val = 2.0 * (np.exp(g * tau) - 1.0) / den
    return float(a_val), float(b_val)

def _measurement_loadings(taus: np.ndarray, kappa: float, theta: float, sigma: float) -> CIRLoadings:
    a_vec = np.zeros(len(taus))
    b_vec = np.zeros(len(taus))
    for i, tau in enumerate(taus):
        a_val, b_val = cir_ab(tau, kappa, theta, sigma)
        a_vec[i] = -np.log(a_val) / tau
        b_vec[i] = b_val / tau
    return CIRLoadings(a=a_vec, b=b_vec)

def _cir_transition_moments(r: float, kappa_p: float, theta_p: float, sigma: float, dt: float, phi: float) -> tuple[float, float]:
    r = max(r, 1e-10)
    mean = theta_p + (r - theta_p) * phi
    denom = max(kappa_p, 1e-12)
    var = r * (sigma**2 / denom) * phi * (1.0 - phi) + theta_p * (sigma**2 / (2.0 * denom)) * (1.0 - phi)**2
    return float(mean), float(max(var, 1e-15))

def _ekf_neg_loglik(params: np.ndarray, Y: np.ndarray, taus: np.ndarray, dt: float) -> float:
    n_obs = Y.shape[1]
    if len(params) != 4 + n_obs: return 1e10
    
    kappa, theta, lambda_risk, sigma = params[:4]
    R_diag = params[4:]

    kappa_p = kappa - sigma * lambda_risk
    if kappa_p <= 0.0 or kappa <= 0.0 or theta <= 0.0 or sigma <= 0.0: return 1e10
    if np.any(R_diag <= 0.0): return 1e10
    if 2.0 * kappa * theta <= sigma ** 2: return 1e10
    
    theta_p = (kappa * theta) / kappa_p
    if theta_p <= 0.0: return 1e10

    try:
        load = _measurement_loadings(taus, kappa, theta, sigma)
    except Exception:
        return 1e10
    
    a, b = load.a, load.b
    b_outer = np.outer(b, b)
    R_mat = np.diag(R_diag)
    phi = float(np.exp(-kappa_p * dt))

    # Strict Initialization to unconditional mean (NO r0 ESTIMATION)
    x = theta_p
    P = (sigma ** 2 * theta_p) / (2.0 * kappa_p)

    ll = 0.0
    for t in range(Y.shape[0]):
        x_pred, Q = _cir_transition_moments(x, kappa_p, theta_p, sigma, dt, phi)
        P_pred = (phi ** 2) * P + Q
        
        y_pred = a + b * x_pred
        v = Y[t] - y_pred
        S = P_pred * b_outer + R_mat
        
        try:
            sign, logdet = np.linalg.slogdet(S)
            if sign <= 0: return 1e10
            v_inv = np.linalg.inv(S)
            ll_step = -0.5 * (n_obs * np.log(2.0 * np.pi) + logdet + v.T @ v_inv @ v)
            if np.isnan(ll_step) or np.isinf(ll_step): return 1e10
            ll += ll_step
        except np.linalg.LinAlgError:
            return 1e10

        K = P_pred * b @ v_inv
        x = max(x_pred + float(K @ v), 1e-8)
        P = max(P_pred - float(K @ b * P_pred), 1e-15)

    return float(-ll)

def fit_cir_via_ekf_mle(Y: np.ndarray, taus: np.ndarray, dt: float, compute_se: bool = True) -> CIRFitResult:
    n_obs = Y.shape[1]
    bounds = [
        (0.001, 5.0),   # kappa
        (0.0001, 0.2),  # theta
        (-5.0, 5.0),    # lambda_risk
        (0.001, 1.0),   # sigma
    ] + [(1e-8, 0.005)] * n_obs

    # Strict Differential Evolution settings to guarantee positive-definite Hessian
    np.random.seed(42)
    result = differential_evolution(
        _ekf_neg_loglik, bounds, args=(Y, taus, dt),
        maxiter=100, tol=1e-12, polish=True, popsize=15,
        mutation=(0.5, 1.5), recombination=0.9, disp=False
    )

    params_hat = result.x
    kappa, theta, lambda_risk, sigma = params_hat[:4]
    R_diag = np.maximum(params_hat[4:], 1e-12)
    best_ll = -float(result.fun)

    se = np.full(4 + n_obs, np.nan)
    if compute_se:
        eps = np.maximum(np.abs(params_hat) * 1e-4, 1e-8)
        hess = np.zeros((len(params_hat), len(params_hat)))
        for i in range(len(params_hat)):
            for j in range(i, len(params_hat)):
                p_pp, p_pm, p_mp, p_mm = params_hat.copy(), params_hat.copy(), params_hat.copy(), params_hat.copy()
                p_pp[i] += eps[i]; p_pp[j] += eps[j]
                p_pm[i] += eps[i]; p_pm[j] -= eps[j]
                p_mp[i] -= eps[i]; p_mp[j] += eps[j]
                p_mm[i] -= eps[i]; p_mm[j] -= eps[j]
                
                f_pp = _ekf_neg_loglik(p_pp, Y, taus, dt)
                f_pm = _ekf_neg_loglik(p_pm, Y, taus, dt)
                f_mp = _ekf_neg_loglik(p_mp, Y, taus, dt)
                f_mm = _ekf_neg_loglik(p_mm, Y, taus, dt)
                
                val = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps[i] * eps[j])
                hess[i, j] = val
                hess[j, i] = val
        try:
            cov = np.linalg.inv(hess)
            se = np.sqrt(np.maximum(np.diag(cov), 0.0))
        except np.linalg.LinAlgError:
            pass

    return CIRFitResult(
        success=bool(result.success), message=str(result.message),
        kappa=float(kappa), theta=float(theta), lambda_risk=float(lambda_risk), sigma=float(sigma),
        R_diag=R_diag.astype(float), max_loglik=best_ll, se_params=se.astype(float)
    )

def ekf_filter_short_rate_path(fit: CIRFitResult, Y: np.ndarray, taus: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    load = _measurement_loadings(taus, fit.kappa, fit.theta, fit.sigma)
    a, b = load.a, load.b
    kappa_p = fit.kappa - fit.sigma * fit.lambda_risk
    theta_p = (fit.kappa * fit.theta) / max(kappa_p, 1e-12)
    
    b_outer = np.outer(b, b)
    R_mat = np.diag(fit.R_diag)
    phi = float(np.exp(-kappa_p * dt))
    
    x = theta_p
    P = (fit.sigma ** 2 * theta_p) / (2.0 * kappa_p)
    
    r_path = np.zeros(Y.shape[0], dtype=float)
    innovations = np.zeros_like(Y, dtype=float)
    
    for t in range(Y.shape[0]):
        x_pred, Q = _cir_transition_moments(x, kappa_p, theta_p, fit.sigma, dt, phi)
        P_pred = (phi ** 2) * P + Q
        
        y_pred = a + b * x_pred
        v = Y[t] - y_pred
        S = P_pred * b_outer + R_mat
        
        v_inv = np.linalg.inv(S)
        K = P_pred * b @ v_inv
        
        x = max(x_pred + float(K @ v), 1e-8)
        P = max(P_pred - float(K @ b * P_pred), 1e-15)
        
        r_path[t] = x
        innovations[t] = v
        
    return r_path, innovations

# ============================================================================
# High-Level Solver
# ============================================================================

class Part2Solver:
    def __init__(self, cfg: Part2Config) -> None:
        self.cfg = cfg
        self.nss = load_nss_coefficients(cfg.excel_path, cfg.sheet_name)
        self.weekly: WeeklyYields | None = None
        self.stats: pd.DataFrame | None = None
        self.fit_result = None
        self.r_path: np.ndarray | None = None
        self.innovations: np.ndarray | None = None

    def compute_weekly(self) -> pd.DataFrame:
        self.weekly = build_weekly_zero_coupon_yields(self.nss.df, self.cfg.taus_years, self.cfg.labels)
        return self.weekly.df

    def compute_stats(self) -> pd.DataFrame:
        if self.weekly is None: self.compute_weekly()
        assert self.weekly is not None
        self.stats = descriptive_stats_table(self.weekly.df, self.cfg.labels)
        return self.stats

    def estimate_cir(self):
        if self.weekly is None:
            self.compute_weekly()
        assert self.weekly is not None
        dates = self.weekly.dates
        dt = self._weekly_dt_years(dates)
        Y = self.weekly.df[list(self.cfg.labels)].to_numpy(dtype=float) / 100.0
        taus = np.array(self.cfg.taus_years, dtype=float)
        
        self.fit_result = fit_cir_via_ekf_mle(Y=Y, taus=taus, dt=dt, compute_se=True)
        self.r_path, self.innovations = ekf_filter_short_rate_path(fit=self.fit_result, Y=Y, taus=taus, dt=dt)
        return self.fit_result

    def compare_rates(self) -> tuple[np.ndarray, np.ndarray]:
        if self.r_path is None: self.estimate_cir()
        assert self.r_path is not None
        r_path = self.r_path 
        nss_sr = nss_instantaneous_short_rate_percent(self.nss.df)
        return r_path, nss_sr

    def get_parameter_table(self) -> pd.DataFrame:
        if self.fit_result is None: raise ValueError("Must call estimate_cir() first")
        fit = self.fit_result
        se = fit.se_params
        kappa_p = fit.kappa - fit.sigma * fit.lambda_risk
        theta_p = (fit.kappa * fit.theta) / kappa_p

        params_df = pd.DataFrame({
            "Parameter": ["kappa (Q)", "theta (Q)", "lambda_risk", "sigma", "kappa_P (derived)", "theta_P (derived)"],
            "Estimate": [fit.kappa, fit.theta, fit.lambda_risk, fit.sigma, kappa_p, theta_p],
            "Std. Error": [se if len(se)>0 else np.nan, se[1] if len(se)>1 else np.nan, se[2] if len(se)>2 else np.nan, se[3] if len(se)>3 else np.nan, np.nan, np.nan]
        })
        return params_df

    def get_measurement_error_table(self) -> pd.DataFrame:
        if self.innovations is None or self.fit_result is None: raise ValueError("Must call estimate_cir() first")
        fit = self.fit_result
        err_df = pd.DataFrame({
            "Maturity": self.cfg.labels,
            "Mean Error": self.innovations.mean(axis=0),
            "Variance Error": self.innovations.var(axis=0, ddof=1),
            "Measurement Variance (h_i)": fit.R_diag,
        })
        return err_df

    @staticmethod
    def _weekly_dt_years(dates: pd.DatetimeIndex) -> float:
        diffs = pd.Series(dates).diff().dt.days.dropna()
        if diffs.empty: raise ValueError("Not enough dates to compute dt")
        return float(np.median(diffs)) / 365.25