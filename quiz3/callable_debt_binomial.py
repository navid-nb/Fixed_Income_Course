"""
CRR tree valuation for four debt scenarios:
1) plain zero-coupon debt
2) callable debt
3) convertible debt
4) callable + convertible debt

The implementation follows the penultimate-date initialization and backward
recursions from the provided formulas for callable/convertible structures.
"""

import math
from scipy.stats import norm


def _credit_spread_bps(debt_value, nominal, maturity, r):
    if debt_value <= 0.0:
        return float("nan")
    return (-(1.0 / maturity) * math.log(debt_value / nominal) - r) * 10000.0


def _merton_debt_one_maturity(v, M, r, sigma, tau, theta=1.0):
    """Merton debt value for one maturity tau (used at penultimate date)."""
    if tau <= 0.0:
        return min(v, M)

    vol = sigma * math.sqrt(tau)
    d1 = (math.log(v / M) + (r + 0.5 * sigma**2) * tau) / vol
    d2 = d1 - vol

    return M * math.exp(-r * tau) * norm.cdf(d2) + theta * v * norm.cdf(-d1)


def _convertible_debt_european(v, M, q, r, sigma, tau):
    """Closed-form European convertible debt value over one horizon tau."""
    if tau <= 0.0:
        return max(min(v, M), q * v)

    vol = sigma * math.sqrt(tau)
    d1 = (math.log(v / M) + (r + 0.5 * sigma**2) * tau) / vol
    d2 = d1 - vol

    strike_conv = M / q
    d3 = (math.log(v / strike_conv) + (r + 0.5 * sigma**2) * tau) / vol
    d4 = d3 - vol

    return (
        v * norm.cdf(-d1)
        + M * math.exp(-r * tau) * (norm.cdf(d2) - norm.cdf(d4))
        + q * v * norm.cdf(d3)
    )


def price_zero_coupon_debt_tree(
    v,
    M,
    T,
    r,
    sigma,
    n_steps=500,
    callable=False,
    convertible=False,
    q=0.6,
    redemption_fn=None,
    theta=1.0,
):
    """Unified CRR valuation for all combinations of callable/convertible."""
    if v <= 0.0 or M <= 0.0:
        raise ValueError("v and M must be strictly positive.")
    if T <= 0.0:
        raise ValueError("T must be strictly positive.")
    if sigma <= 0.0:
        raise ValueError("sigma must be strictly positive.")
    if n_steps < 2:
        raise ValueError("n_steps must be at least 2.")
    if convertible and not (0.0 < q <= 1.0):
        raise ValueError("q must satisfy 0 < q <= 1 when convertible=True.")

    if redemption_fn is None:
        def redemption_fn(t_now, face):
            return face

    dt = T / n_steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(r * dt) - d) / (u - d)

    if not (0.0 <= p <= 1.0):
        raise ValueError("Risk-neutral probability is outside [0, 1].")

    k_penult = n_steps - 1
    t_penult = k_penult * dt
    K_penult = redemption_fn(t_penult, M)

    D_next = [0.0] * n_steps
    for j in range(n_steps):
        v_node = v * (u ** j) * (d ** (k_penult - j))

        continuation_merton = _merton_debt_one_maturity(v_node, M, r, sigma, dt, theta=theta)

        if not callable and not convertible:
            D_next[j] = continuation_merton
            continue

        conv_value = q * v_node if convertible else float("-inf")
        cont_for_conv = (
            _convertible_debt_european(v_node, M, q, r, sigma, dt)
            if convertible
            else continuation_merton
        )

        if callable and not convertible:
            D_next[j] = min(K_penult, continuation_merton)
        elif (not callable) and convertible:
            D_next[j] = max(conv_value, cont_for_conv)
        else:
            D_next[j] = min(max(conv_value, cont_for_conv), max(conv_value, K_penult))

    for k in range(n_steps - 2, -1, -1):
        t_k = k * dt
        K_k = redemption_fn(t_k, M)
        D_curr = [0.0] * (k + 1)

        for j in range(k + 1):
            v_node = v * (u ** j) * (d ** (k - j))
            continuation = math.exp(-r * dt) * (p * D_next[j + 1] + (1.0 - p) * D_next[j])

            if not callable and not convertible:
                D_curr[j] = continuation
            elif callable and not convertible:
                D_curr[j] = min(K_k, continuation)
            elif (not callable) and convertible:
                D_curr[j] = max(q * v_node, continuation)
            else:
                D_curr[j] = min(max(q * v_node, continuation), max(q * v_node, K_k))

        D_next = D_curr

    debt_price = float(D_next[0])

    if callable and convertible:
        scenario = "callable_convertible"
    elif callable:
        scenario = "callable"
    elif convertible:
        scenario = "convertible"
    else:
        scenario = "plain"

    return {
        "scenario": scenario,
        "debt": debt_price,
        "credit_spread_bps": _credit_spread_bps(debt_price, M, T, r),
        "u": u,
        "d": d,
        "p": p,
        "dt": dt,
        "n_steps": n_steps,
    }
