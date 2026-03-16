"""Explicit finite-difference pricing for one-factor short-rate PDEs.

The implementation follows the general explicit finite-difference scheme shown
in the user's notes:

- (7.19) short-rate diffusion: dr_t = mu(t, r_t) dt + sigma(t, r_t) dW_t
- (7.20) pricing PDE for g(t, r)
- (7.22)-(7.24) central-difference delta/gamma and backward theta
- (7.25) explicit backward recursion
- (7.26)-(7.28) coefficients A_i^t, B_i^t, C_i^t

CIR is then used as a specific case with:
mu(r) = kappa (theta - r), sigma(r) = sigma * sqrt(r).
"""

import numpy as np


def cir_zero_coupon_price(
    t: float,
    T: float,
    r: np.ndarray | float,
    kappa: float,
    theta: float,
    sigma: float,
) -> np.ndarray | float:
    """CIR zero-coupon bond price using the closed form from the lecture notes.

    P_cir(t, T) = a_cir(t, T) * exp(-b_cir(t, T) * r_t)

    where

    gamma = sqrt(kappa^2 + 2 sigma^2)
    a_cir(t, T) = 
        [2 gamma exp((kappa + gamma)(T-t)/2)
         / ((kappa + gamma)(exp(gamma(T-t)) - 1) + 2 gamma)]
        ^(2 kappa theta / sigma^2)
    b_cir(t, T) =
        2 (exp(gamma(T-t)) - 1)
        / ((kappa + gamma)(exp(gamma(T-t)) - 1) + 2 gamma)
    """
    if T < t:
        raise ValueError("T must be greater than or equal to t")
    if T == t:
        if np.isscalar(r):
            return 1.0
        return np.ones_like(np.asarray(r, dtype=float))

    # Compute the affine CIR bond-pricing coefficients a_cir(t,T) and b_cir(t,T),
    # then apply P(t,T) = a_cir(t,T) * exp(-b_cir(t,T) * r_t).
    sigma2 = sigma * sigma
    gamma = np.sqrt(kappa * kappa + 2.0 * sigma2)
    tau = T - t
    exp_gamma_tau = np.exp(gamma * tau)
    b_cir = 2.0 * (exp_gamma_tau - 1.0) / (
        (kappa + gamma) * (exp_gamma_tau - 1.0) + 2.0 * gamma
    )
    a_cir = (
        (2.0 * gamma * np.exp(0.5 * (kappa + gamma) * tau))
        / ((kappa + gamma) * (exp_gamma_tau - 1.0) + 2.0 * gamma)
    ) ** (2.0 * kappa * theta / sigma2)
    return a_cir * np.exp(-b_cir * np.asarray(r, dtype=float))


def explicit_scheme_coefficients(
    r: np.ndarray,
    mu: np.ndarray,
    vol: np.ndarray,
    dr: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Return explicit-scheme coefficients using formulas (7.25)-(7.28).

    With v1 = dt / dr^2 and v2 = dt / dr, formulas (7.26)-(7.28) give:
    A_i^t = 0.5 * ((sigma_i^t)^2 * v1 - mu_i^t * v2)
    B_i^t = 1 - (sigma_i^t)^2 * v1 - i*dr*dt
    C_i^t = 0.5 * (mu_i^t * v2 + (sigma_i^t)^2 * v1)

    The input arrays r, mu, vol correspond to the interior space nodes i=1,...,N-1
    for a fixed time layer t in the recursion.
    """
    if dr <= 0.0 or dt <= 0.0:
        raise ValueError("dr and dt must be strictly positive")

    # v1 and v2 are the two grid-scaling terms that appear repeatedly in
    # the explicit finite-difference coefficients.
    v1 = dt / (dr * dr)
    v2 = dt / dr
    vol2 = vol * vol
    a = 0.5 * (vol2 * v1 - mu * v2)
    b = 1.0 - vol2 * v1 - r * dt
    c = 0.5 * (mu * v2 + vol2 * v1)
    return a, b, c, v1, v2


def check_explicit_scheme_stability(
    r: np.ndarray,
    mu: np.ndarray,
    vol: np.ndarray,
    dr: float,
    dt: float,
) -> tuple[bool, str]:
    """Practical stability check using the coefficient bounds from (7.26)-(7.28).

    The notes state the explicit scheme is stable when the time grid is fine
    enough relative to the space grid. A practical check is to verify the
    interior coefficients remain in [0, 1].
    """
    # For an explicit scheme we want the update weights to remain well-behaved;
    # the lecture notes suggest keeping the time step small enough relative to
    # the space step so these coefficients stay between 0 and 1.
    a, b, c, v1, v2 = explicit_scheme_coefficients(r, mu, vol, dr, dt)
    stable = bool(
        np.all(a >= 0.0)
        and np.all(b >= 0.0)
        and np.all(c >= 0.0)
        and np.all(a <= 1.0)
        and np.all(b <= 1.0)
        and np.all(c <= 1.0)
    )
    msg = (
        f"v1={v1:.6f}, v2={v2:.6f}, "
        f"A in [{a.min():.6f}, {a.max():.6f}], "
        f"B in [{b.min():.6f}, {b.max():.6f}], "
        f"C in [{c.min():.6f}, {c.max():.6f}], stable={stable}"
    )
    return stable, msg


def solve_one_factor_explicit_pde(
    r_grid: np.ndarray,
    maturity: float,
    n_t_steps: int,
    dt: float,
    mu_fn,
    vol_fn,
    terminal_payoff_fn,
    lower_boundary_fn,
    upper_boundary_fn,
    exercise_value_fn=None,
) -> np.ndarray:
    """Solve the one-factor pricing PDE by the explicit scheme (7.25).

    Delta, gamma, and theta are discretized as in (7.22)-(7.24), giving the
    backward recursion in (7.25): g_i^{t-1} = A_i^t g_{i-1}^t + B_i^t g_i^t + C_i^t g_{i+1}^t.

    Index convention used here:
    - i: space node index on the short-rate grid
    - t: time-layer index, with t increasing forward in time
    """
    if n_t_steps <= 0:
        raise ValueError("n_t_steps must be positive")
    if maturity <= 0.0:
        raise ValueError("maturity must be positive")
    if not np.isclose(maturity / n_t_steps, dt, atol=1e-12):
        raise ValueError("dt must equal maturity / n_t_steps")

    dr = float(r_grid[1] - r_grid[0])
    # Start from the terminal condition at option maturity and march backward to t=0.
    values = terminal_payoff_fn(r_grid).astype(float)

    interior_r = r_grid[1:-1]
    mu = mu_fn(interior_r)
    vol = vol_fn(interior_r)
    # These are the left/center/right weights in the explicit recursion.
    a, b, c, _, _ = explicit_scheme_coefficients(interior_r, mu, vol, dr, dt)

    for t_step in range(n_t_steps):
        # Moving backward one time layer: from t to t-1 in formula notation.
        t_prev = maturity - (t_step + 1) * dt
        next_values = values.copy()

        # Apply boundary conditions at i=0 and i=N, then update interior i-nodes.
        next_values[0] = lower_boundary_fn(t_prev)
        next_values[-1] = upper_boundary_fn(t_prev)
        values[1:-1] = (
            a * next_values[:-2]
            + b * next_values[1:-1]
            + c * next_values[2:]
        )

        if exercise_value_fn is not None:
            # American exercise is handled by comparing continuation value with
            # immediate exercise value at each time step and grid node.
            values = np.maximum(values, exercise_value_fn(t_prev, r_grid))

    return values




#############################################
# CIR-specific functions
#############################################

def check_explicit_cir_stability(
    kappa: float,
    theta: float,
    sigma: float,
    r_max: float,
    dr: float,
    dt: float,
) -> tuple[bool, str]:
    """Apply the general explicit-scheme stability check to the CIR model."""
    if dr <= 0.0 or dt <= 0.0 or r_max <= 0.0:
        return False, "dr, dt, and r_max must be strictly positive"

    # Specialize the general one-factor inputs mu(r) and sigma(r) to CIR.
    r_grid = np.arange(int(round(r_max / dr)) + 1, dtype=float) * dr
    interior_r = r_grid[1:-1]
    mu = kappa * (theta - interior_r)
    vol = sigma * np.sqrt(np.maximum(interior_r, 0.0))
    return check_explicit_scheme_stability(interior_r, mu, vol, dr, dt)


def explicit_cir_put_on_bond_premiums_bps(
    kappa: float,
    theta: float,
    sigma: float,
    r0: float,
    option_maturity: float,
    bond_maturity: float,
    strike: float,
    n_r_steps: int = 100,
    dr: float = 0.001,
    n_t_steps: int = 2000,
    dt: float = 1.0 / 2000.0,
) -> dict[str, float | np.ndarray]:
    """Price European/American put options on a ZCB under CIR in bps.

    This uses the general explicit scheme (7.25), with CIR plugged into the
    generic drift/volatility slots from (7.19).
    """
    if n_r_steps <= 1:
        raise ValueError("n_r_steps must be greater than 1")
    if n_t_steps <= 0:
        raise ValueError("n_t_steps must be positive")
    if option_maturity <= 0.0 or bond_maturity <= option_maturity:
        raise ValueError("Require 0 < option_maturity < bond_maturity")
    if strike <= 0.0:
        raise ValueError("strike must be positive")

    if not np.isclose(option_maturity / n_t_steps, dt, atol=1e-12):
        raise ValueError("dt must equal option_maturity / n_t_steps")

    # Build the short-rate grid r = 0, dr, 2dr, ..., r_max.
    r = np.arange(n_r_steps + 1, dtype=float) * dr
    r_max = float(r[-1])

    stable, reason = check_explicit_cir_stability(kappa, theta, sigma, r_max, dr, dt)
    if not stable:
        raise ValueError(f"Explicit CIR grid appears unstable: {reason}")

    def mu_fn(r_vals: np.ndarray) -> np.ndarray:
        # CIR drift from dr_t = kappa(theta - r_t)dt + sigma sqrt(r_t)dW_t.
        return kappa * (theta - r_vals)

    def vol_fn(r_vals: np.ndarray) -> np.ndarray:
        # CIR diffusion coefficient evaluated pointwise on the rate grid.
        if np.any(r_vals < 0.0):
            raise ValueError("Negative rates observed clipping to zero for volatility calculation")
        return sigma * np.sqrt(np.maximum(r_vals, 0.0))

    def terminal_payoff_fn(r_vals: np.ndarray) -> np.ndarray:
        # At option maturity, the put pays max(K - P(T_opt, T_bond), 0).
        bond_at_expiry = cir_zero_coupon_price(option_maturity, bond_maturity, r_vals, kappa, theta, sigma)
        return np.maximum(strike - bond_at_expiry, 0.0)

    # Boundary constants from the notes' put setup (evaluated at t=0):
    # g_0^t = max(0, K*P(0,T_opt,0) - P(0,T_bond,0))
    # g_N^t = max(0, K*P(0,T_opt,r_max) - P(0,T_bond,r_max)).
    left_boundary_const = max(
        strike * float(cir_zero_coupon_price(0.0, option_maturity, 0.0, kappa, theta, sigma))
        - float(cir_zero_coupon_price(0.0, bond_maturity, 0.0, kappa, theta, sigma)),
        0.0,
    )
    right_boundary_const = max(
        strike * float(cir_zero_coupon_price(0.0, option_maturity, r_max, kappa, theta, sigma))
        - float(cir_zero_coupon_price(0.0, bond_maturity, r_max, kappa, theta, sigma)),
        0.0,
    )

    def lower_boundary_fn(t_now: float) -> float:
        # Put boundary from the notes image, constant through backward time steps.
        _ = t_now
        return left_boundary_const

    def upper_boundary_eur_fn(t_now: float) -> float:
        # Put boundary from the notes image, constant through backward time steps.
        _ = t_now
        return right_boundary_const

    def upper_boundary_am_fn(t_now: float) -> float:
        # Same right-edge boundary used in the notes for both Euro and American grids.
        _ = t_now
        return right_boundary_const

    def exercise_value_fn(t_now: float, r_vals: np.ndarray) -> np.ndarray:
        # Intrinsic value used for the American early-exercise comparison.
        bond_value = cir_zero_coupon_price(t_now, bond_maturity, r_vals, kappa, theta, sigma)
        return np.maximum(strike - bond_value, 0.0)

    v_eur = solve_one_factor_explicit_pde(
        r_grid=r,
        maturity=option_maturity,
        n_t_steps=n_t_steps,
        dt=dt,
        mu_fn=mu_fn,
        vol_fn=vol_fn,
        terminal_payoff_fn=terminal_payoff_fn,
        lower_boundary_fn=lower_boundary_fn,
        upper_boundary_fn=upper_boundary_eur_fn,
    )
    v_am = solve_one_factor_explicit_pde(
        r_grid=r,
        maturity=option_maturity,
        n_t_steps=n_t_steps,
        dt=dt,
        mu_fn=mu_fn,
        vol_fn=vol_fn,
        terminal_payoff_fn=terminal_payoff_fn,
        lower_boundary_fn=lower_boundary_fn,
        upper_boundary_fn=upper_boundary_am_fn,
        exercise_value_fn=exercise_value_fn,
    )

    eur_price = float(np.interp(r0, r, v_eur))
    am_price = float(np.interp(r0, r, v_am))

    return {
        "european_put_bps": eur_price * 10000.0,
        "american_put_bps": am_price * 10000.0,
        "r_grid": r,
        "eur_grid": v_eur,
        "am_grid": v_am,
        "stability_info": reason,
    }
