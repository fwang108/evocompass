from __future__ import annotations

import math

import numpy as np


def negative_log_likelihood(probabilities: np.ndarray) -> float:
    values = np.asarray(probabilities, dtype=float)
    if values.size == 0:
        raise ValueError("At least one probability is required")
    return float(-np.log(np.clip(values, 1e-12, 1.0)).sum())


def family_bootstrap(
    effects: dict[str, float], draws: int = 10_000, seed: int = 0
) -> dict[str, float | int]:
    values = np.asarray(list(effects.values()), dtype=float)
    if values.size < 2:
        raise ValueError("Family-level inference requires at least two families")
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(draws)])
    return {
        "mean": float(values.mean()),
        "low_95": float(np.quantile(means, 0.025)),
        "high_95": float(np.quantile(means, 0.975)),
        "families": int(values.size),
        "positive_family_fraction": float((values > 0).mean()),
    }


def sign_flip_test(effects: dict[str, float], draws: int = 20_000, seed: int = 0) -> float:
    values = np.asarray(list(effects.values()), dtype=float)
    observed = values.mean()
    rng = np.random.default_rng(seed)
    null = np.array(
        [(values * rng.choice((-1.0, 1.0), len(values))).mean() for _ in range(draws)]
    )
    return float((1 + (null >= observed).sum()) / (draws + 1))


def trajectory_nll(event_mass: float, identity_mass: float) -> float:
    return -math.log(max(event_mass, 1e-12)) - math.log(max(identity_mass, 1e-12))

