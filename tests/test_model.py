import numpy as np
import torch

from evocompass import EvoCompassModel, build_features
from evocompass.constants import OPERATIONS


def test_prediction_shapes_and_probabilities():
    model = EvoCompassModel()
    features = build_features(np.array([0, 1]), np.ones((2, 20)), 0.5)
    for operation in OPERATIONS:
        prediction = model.predict(operation, features)
        assert prediction["event_probability"].shape == (2,)
        assert prediction["identity_probability"].shape == (2, 20)
        assert torch.allclose(prediction["identity_probability"].sum(-1), torch.ones(2))


def test_checkpoint_round_trip(tmp_path):
    model = EvoCompassModel()
    path = tmp_path / "model.pt"
    model.save(path, {"purpose": "unit-test"})
    loaded = EvoCompassModel.load(path)
    assert sum(parameter.numel() for parameter in loaded.parameters()) == 30303

