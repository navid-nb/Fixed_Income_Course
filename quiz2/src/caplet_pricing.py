"""Forward-looking and backward-looking caplet/floorlet pricing helpers."""


import numpy as np


def _validate_indices(reset_index: int, settlement_index: int, total_steps: int) -> None:
    if reset_index < 0 or settlement_index < 0:
        raise ValueError("reset_index and settlement_index must be non-negative")
    if settlement_index <= reset_index:
        raise ValueError("settlement_index must be strictly greater than reset_index")
    if settlement_index >= total_steps:
        raise ValueError("settlement_index is out of bounds for the provided rate paths")


def _path_discount_factors(
    rates: np.ndarray,
    dt: float,
    t1_index: int,
    t2_index: int,
) -> np.ndarray:
    """Return pathwise discount factors that discount cashflows from t2 to t1."""
    if t1_index < 0 or t2_index < 0:
        raise ValueError("t1_index and t2_index must be non-negative")
    if t2_index < t1_index:
        raise ValueError("t2_index must be greater than or equal to t1_index")
    if t2_index >= rates.shape[1]:
        raise ValueError("t2_index is out of bounds for the provided rate paths")

    if t2_index == t1_index:
        return np.ones(rates.shape[0], dtype=float)

    # Pathwise short-rate discounting from t2 back to t1.
    integrated = np.sum(rates[:, t1_index:t2_index], axis=1) * dt
    return np.exp(-integrated)


def _premium_from_rate(
    underlying_rate: np.ndarray,
    strike: float,
    tau: float, # time from reset to settlement
    discount_factor_t2_to_t1: np.ndarray,
    is_cap: bool,
) -> dict[str, float]:
    intrinsic = underlying_rate - strike if is_cap else strike - underlying_rate
    payoff = tau * np.maximum(intrinsic, 0.0)
    pv_paths = discount_factor_t2_to_t1 * payoff
    pv = float(np.mean(pv_paths))

    return  pv * 10000 # convert to bps of notional


def forward_looking_option_premium_bps(
    rates: np.ndarray,
    strike: float,
    reset_index: int = 275,
    settlement_index: int = 365,
    dt: float = 1.0 / 365.0,
    is_cap: bool = True,
) -> dict[str, float]:
    """Price a forward-looking caplet or floorlet using r(reset)."""
    if rates.ndim != 2:
        raise ValueError("rates must be a 2D array shaped (n_paths, n_steps)")

    _validate_indices(reset_index, settlement_index, rates.shape[1])

    discount_factors = _path_discount_factors(rates, dt, t1_index=0, t2_index=settlement_index)

    tau = (settlement_index - reset_index) * dt
    ref_rate = rates[:, reset_index]

    result = _premium_from_rate(ref_rate, strike, tau, discount_factors, is_cap=is_cap)
    return result


def backward_looking_option_premium_bps(
    rates: np.ndarray,
    strike: float,
    reset_index: int = 275,
    settlement_index: int = 365,
    dt: float = 1.0 / 365.0,
    is_cap: bool = True,
) -> dict[str, float]:
    """Price a backward-looking caplet or floorlet from realized average short rate."""
    if rates.ndim != 2:
        raise ValueError("rates must be a 2D array shaped (n_paths, n_steps)")

    _validate_indices(reset_index, settlement_index, rates.shape[1])

    discount_factors = _path_discount_factors(rates, dt, t1_index=0, t2_index=settlement_index)

    tau = (settlement_index - reset_index) * dt
    # Backward-looking rate proxy: arithmetic daily average over [reset, settlement).
    realized_rate = np.mean(rates[:, reset_index:settlement_index], axis=1)
    
    # if we wanted to include the settlement day rate in the average, we would use:
    # realized_rate = np.mean(rates[:, reset_index:settlement_index+1], axis=1) 

    result = _premium_from_rate(realized_rate, strike, tau, discount_factors, is_cap=is_cap)
    return result
