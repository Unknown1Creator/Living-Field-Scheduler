#!/usr/bin/env python3
"""Exploratory simulation of the core Living Field Scheduler pressure dynamics.

This is intentionally a lightweight sanity check, not a full OS scheduler.
It focuses on whether the core pressure update rules produce stable behavior,
which is the main risk in a design like this.
"""

from __future__ import annotations

import numpy as np

WEIGHTS = np.array([1.40, 1.15, 1.05, 0.85, 0.75, 0.90], dtype=float)


def clamp(v: np.ndarray, low: float = 0.0, high: float = 9.5) -> np.ndarray:
    return np.clip(v, low, high)


def global_pressure(P: np.ndarray) -> float:
    return float(WEIGHTS @ P)


def simulate(n_tasks: int = 12, steps: int = 80, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    # Tasks maintain non-negative demand vectors normalized to sum 1, as described.
    tasks = rng.dirichlet(np.ones(6), size=n_tasks)

    # Pressure starts in a low but non-zero regime.
    P = np.full(6, 1.0, dtype=float)
    S_history = []
    progress_history = []

    for step in range(steps):
        S = global_pressure(P)
        S_history.append(S)

        mu = P.mean()
        alpha = np.clip(0.78 + 0.004 * S, 0.78, 0.85)
        P = alpha * P + (1.0 - alpha) * mu

        # QEP: tasks are pulled toward high-pressure regions.
        lam_ent = np.clip(0.4 + 0.005 * S, 0.4, 0.6)
        P = P + lam_ent * S * tasks.mean(axis=0)

        # Interference is modeled as aggregate demand overlap.
        overlap = tasks.mean(axis=0)
        P = P + 0.035 * S * overlap

        # Decoherence / noise is mild but helps keep the system from becoming rigid.
        noise = rng.normal(0.0, 0.05, size=6)
        P = P + 0.02 * noise

        # Self-organizing clustering adds more pressure when tasks are aligned.
        P = P + 0.03 * (tasks.mean(axis=0) * (1.0 + 0.1 * np.sin(step / 5.0)))

        # Emergency venting protocol is triggered when the global pressure is high.
        if S > 25.0:
            P = 0.78 * P

        P = clamp(P)

        # Progress model: higher global pressure suppresses progress, but cannot go negative.
        cos_theta = float(np.dot(tasks.mean(axis=0), P) / (np.linalg.norm(tasks.mean(axis=0)) * np.linalg.norm(P)))
        progress = max(0.32, 2.45 - 0.155 * S) * (1.0 + 0.18 * cos_theta)
        progress_history.append(progress)

    return np.array(S_history), np.array(progress_history)


def main() -> None:
    try:
        S_hist, progress_hist = simulate()
        print("Living Field Scheduler pressure sanity check")
        print(f"Initial pressure: {global_pressure(np.full(6, 1.0))}")
        print(f"Peak global pressure: {S_hist.max():.3f}")
        print(f"Final global pressure: {S_hist[-1]:.3f}")
        print(f"Average progress factor: {progress_hist.mean():.3f}")
    except Exception as exc:  # pragma: no cover
        print(f"Simulation failed: {exc}")
        raise


if __name__ == "__main__":
    main()
