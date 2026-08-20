from __future__ import annotations

import numpy as np

from .features import normalize_profiles


def shuffle_correspondence(profiles: np.ndarray, seed: int = 0) -> np.ndarray:
    values = normalize_profiles(profiles).numpy()
    return values[np.random.default_rng(seed).permutation(len(values))]


def mix_correspondence(
    true_profiles: np.ndarray,
    control_profiles: np.ndarray,
    fraction: float,
) -> np.ndarray:
    if not 0 <= fraction <= 1:
        raise ValueError("Correspondence fraction must lie in [0, 1]")
    true_values = normalize_profiles(true_profiles).numpy()
    control_values = normalize_profiles(control_profiles).numpy()
    if true_values.shape != control_values.shape:
        raise ValueError("Profile arrays must have equal shape")
    mixed = fraction * true_values + (1 - fraction) * control_values
    return mixed / mixed.sum(axis=1, keepdims=True)

