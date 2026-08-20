from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .constants import FEATURE_DIM, HIDDEN_DIM, OPERATIONS


class EventHead(torch.nn.Module):
    def __init__(self, input_dim: int = FEATURE_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class IdentityHead(torch.nn.Module):
    def __init__(self, input_dim: int = FEATURE_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.0),
            torch.nn.Linear(hidden_dim, 20),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


@dataclass(frozen=True)
class LogisticCalibration:
    coefficient: float
    intercept: float

    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.coefficient * logits + self.intercept)


class EvoCompassModel(torch.nn.Module):
    def __init__(self, input_dim: int = FEATURE_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.event_heads = torch.nn.ModuleDict(
            {operation: EventHead(input_dim, hidden_dim) for operation in OPERATIONS}
        )
        self.identity_heads = torch.nn.ModuleDict(
            {operation: IdentityHead(input_dim, hidden_dim) for operation in OPERATIONS}
        )
        self.calibration = {
            operation: LogisticCalibration(1.0, 0.0) for operation in OPERATIONS
        }

    def predict(self, operation: str, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if operation not in OPERATIONS:
            raise ValueError(f"Unknown operation: {operation}")
        event_logits = self.event_heads[operation](features)
        identity_logits = self.identity_heads[operation](features)
        return {
            "event_logits": event_logits,
            "event_probability": self.calibration[operation].probabilities(event_logits),
            "identity_logits": identity_logits,
            "identity_probability": identity_logits.softmax(dim=-1),
        }

    def checkpoint(self, metadata: dict | None = None) -> dict:
        return {
            "format": "evocompass-v1",
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "operations": {
                operation: {
                    "event_state_dict": self.event_heads[operation].state_dict(),
                    "identity_state_dict": self.identity_heads[operation].state_dict(),
                    "calibration": {
                        "coefficient": self.calibration[operation].coefficient,
                        "intercept": self.calibration[operation].intercept,
                    },
                }
                for operation in OPERATIONS
            },
            "metadata": metadata or {},
        }

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        torch.save(self.checkpoint(metadata), Path(path))

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu") -> "EvoCompassModel":
        record = torch.load(Path(path), map_location=map_location, weights_only=False)
        if record.get("format") != "evocompass-v1":
            raise ValueError("Unsupported checkpoint format")
        model = cls(record["input_dim"], record["hidden_dim"])
        for operation in OPERATIONS:
            component = record["operations"][operation]
            model.event_heads[operation].load_state_dict(component["event_state_dict"])
            model.identity_heads[operation].load_state_dict(component["identity_state_dict"])
            model.calibration[operation] = LogisticCalibration(**component["calibration"])
        return model

