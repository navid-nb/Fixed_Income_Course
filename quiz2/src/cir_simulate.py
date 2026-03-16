"""Fast NumPy-based CIR short-rate simulation utilities."""

import numpy as np


def _box_muller_normals(size: int, seed: int) -> np.ndarray:
    """Generate standard normal draws with NumPy's fast random generator."""
    if size <= 0:
        raise ValueError("size must be positive")

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    rng = np.random.default_rng(seed)
    return rng.standard_normal(size)



def simulate_cir_paths(
    kappa: float,
    theta: float,
    sigma: float,
    r0: float,
    seed: int,
    n_paths: int = 1000,
    n_steps: int = 365,
    dt: float = 1.0 / 365.0,
    n_jobs: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate many CIR paths using NumPy vectorization across paths.

    The n_jobs argument is retained for backward compatibility and is ignored,
    because this vectorized implementation is typically faster than process-based
    parallelism for the intended workload.
    """
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    times = np.arange(n_steps + 1, dtype=float) * dt
    shocks = _box_muller_normals(n_paths * n_steps, seed=seed).reshape(n_paths, n_steps)

    rates = np.empty((n_paths, n_steps + 1), dtype=float)
    rates[:, 0] = r0

    sqrt_dt = np.sqrt(dt)
    for t in range(n_steps):
        current_r = rates[:, t]
        drift = kappa * (theta - current_r) * dt
        if np.any(current_r < 0.0):
            print(f"Warning: negative short rates at time step {t}, clipping diffusion to zero for those paths")
            current_r = np.maximum(current_r, 0.0)
        diffusion = sigma * np.sqrt(current_r) * sqrt_dt * shocks[:, t]
        rates[:, t + 1] = current_r + drift + diffusion

    return times, rates


def plot_cir_paths(
    rates: np.ndarray,
    times: np.ndarray,
    reset_day: int | None = None,
    settlement_day: int | None = None,
    n_plot_paths: int = 40,
) -> None:
    """Plot a sample of simulated CIR paths and the terminal rate histogram."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    days = np.arange(rates.shape[1])
    for i in range(min(n_plot_paths, rates.shape[0])):
        axes[0].plot(days, rates[i], alpha=0.6, linewidth=1.0)
    if reset_day is not None:
        axes[0].axvline(reset_day, color='black', linestyle='--', linewidth=1.2, label='reset day')
    if settlement_day is not None:
        axes[0].axvline(settlement_day, color='gray', linestyle='--', linewidth=1.2, label='settlement day')
    axes[0].set_title('Sample CIR Paths')
    axes[0].set_xlabel('Day')
    axes[0].set_ylabel('Short Rate')
    if reset_day is not None or settlement_day is not None:
        axes[0].legend()

    axes[1].hist(rates[:, -1], bins=40, alpha=0.85, edgecolor='black')
    axes[1].set_title(f'Terminal Rate Distribution (Day {rates.shape[1] - 1})')
    axes[1].set_xlabel('Short Rate')
    axes[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()
