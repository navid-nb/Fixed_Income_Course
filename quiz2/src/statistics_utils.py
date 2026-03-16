"""Descriptive statistics helpers for simulated rates."""

from __future__ import annotations

import numpy as np
import pandas as pd


def terminal_rate_statistics(rates: np.ndarray) -> pd.Series:
    """Return descriptive statistics for the terminal short-rate distribution."""
    if rates.ndim != 2:
        raise ValueError("rates must be a 2D array shaped (n_paths, n_steps)")

    terminal = rates[:, -1]
    q1, q2, q3 = np.percentile(terminal, [25, 50, 75])

    return pd.Series(
        {
            "mean": float(np.mean(terminal)),
            "std": float(np.std(terminal, ddof=1)),
            "q1": float(q1),
            "q2_median": float(q2),
            "q3": float(q3),
        }
    )
