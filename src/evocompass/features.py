from __future__ import annotations

import numpy as np
import torch

from .constants import FEATURE_DIM


def normalize_profiles(profiles: np.ndarray | torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(profiles, dtype=torch.float32)
    if values.ndim != 2 or values.shape[1] != 20:
        raise ValueError("MSA profiles must have shape [rows, 20]")
    if not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError("MSA profiles must be finite and non-negative")
    totals = values.sum(dim=1, keepdim=True)
    if (totals <= 0).any():
        raise ValueError("Every MSA profile must have positive mass")
    return values / totals


def source_one_hot(source: np.ndarray | torch.Tensor) -> torch.Tensor:
    indices = torch.as_tensor(source, dtype=torch.long)
    if indices.ndim != 1 or (indices < 0).any() or (indices >= 20).any():
        raise ValueError("Source indices must be a one-dimensional array in [0, 19]")
    return torch.nn.functional.one_hot(indices, num_classes=20).float()


def build_features(
    source: np.ndarray | torch.Tensor,
    profiles: np.ndarray | torch.Tensor,
    time: float | np.ndarray | torch.Tensor,
) -> torch.Tensor:
    source_features = source_one_hot(source)
    profile_features = normalize_profiles(profiles)
    if source_features.shape[0] != profile_features.shape[0]:
        raise ValueError("Source and profile row counts differ")
    if np.isscalar(time):
        time_features = torch.full((source_features.shape[0], 1), float(time))
    else:
        time_features = torch.as_tensor(time, dtype=torch.float32).reshape(-1, 1)
    if time_features.shape[0] != source_features.shape[0]:
        raise ValueError("Time and source row counts differ")
    if ((time_features < 0) | (time_features > 1)).any():
        raise ValueError("Normalized time must lie in [0, 1]")
    features = torch.cat([source_features, profile_features, time_features], dim=1)
    if features.shape[1] != FEATURE_DIM:
        raise RuntimeError("Unexpected feature width")
    return features

