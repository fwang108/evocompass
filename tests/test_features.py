import numpy as np
import pytest

from evocompass.controls import mix_correspondence, shuffle_correspondence
from evocompass.features import build_features, normalize_profiles


def test_feature_contract():
    profiles = np.ones((3, 20), dtype=np.float32)
    features = build_features(np.array([0, 1, 2]), profiles, 0.5)
    assert features.shape == (3, 41)
    assert np.allclose(features[:, 20:40].sum(1).numpy(), 1.0)
    assert np.allclose(features[:, -1].numpy(), 0.5)


def test_invalid_profiles_are_rejected():
    with pytest.raises(ValueError):
        normalize_profiles(np.zeros((2, 20)))


def test_correspondence_controls_preserve_probability_mass():
    profiles = np.arange(1, 61, dtype=float).reshape(3, 20)
    shuffled = shuffle_correspondence(profiles, seed=4)
    mixed = mix_correspondence(profiles, shuffled, 0.25)
    assert np.allclose(shuffled.sum(1), 1.0)
    assert np.allclose(mixed.sum(1), 1.0)

