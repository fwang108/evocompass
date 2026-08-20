from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .calibration import probability_to_hazard
from .constants import AMINO_ACIDS


@dataclass(frozen=True)
class EditCandidate:
    operation: str
    position: int
    event_probability: float
    identity_probability: np.ndarray | None = None


@dataclass(frozen=True)
class EditEvent:
    operation: str
    position: int
    residue: str | None
    waiting_time: float


def normalized_rates(candidates: Iterable[EditCandidate]) -> tuple[list[EditCandidate], np.ndarray]:
    items = list(candidates)
    if not items:
        return [], np.empty(0, dtype=float)
    rates = probability_to_hazard(np.array([item.event_probability for item in items]))
    total = rates.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Candidate rates must have positive finite mass")
    return items, rates / total


def sample_event(candidates: Iterable[EditCandidate], rng: np.random.Generator) -> EditEvent:
    items, probabilities = normalized_rates(candidates)
    if not items:
        raise ValueError("Cannot sample without candidates")
    rates = probability_to_hazard(np.array([item.event_probability for item in items]))
    waiting_time = float(rng.exponential(1.0 / rates.sum()))
    item = items[int(rng.choice(len(items), p=probabilities))]
    residue = None
    if item.operation != "deletion":
        if item.identity_probability is None:
            raise ValueError(f"{item.operation} requires an identity distribution")
        identity = np.asarray(item.identity_probability, dtype=float)
        if identity.shape != (20,) or identity.sum() <= 0:
            raise ValueError("Identity probabilities must have shape [20] and positive mass")
        identity = identity / identity.sum()
        residue = AMINO_ACIDS[int(rng.choice(20, p=identity))]
    return EditEvent(item.operation, item.position, residue, waiting_time)


def apply_event(sequence: str, event: EditEvent) -> str:
    if event.operation == "substitution":
        if event.residue is None or not 0 <= event.position < len(sequence):
            raise ValueError("Invalid substitution")
        return sequence[: event.position] + event.residue + sequence[event.position + 1 :]
    if event.operation == "deletion":
        if not 0 <= event.position < len(sequence):
            raise ValueError("Invalid deletion")
        return sequence[: event.position] + sequence[event.position + 1 :]
    if event.operation == "insertion":
        if event.residue is None or not 0 <= event.position <= len(sequence):
            raise ValueError("Invalid insertion")
        return sequence[: event.position] + event.residue + sequence[event.position :]
    raise ValueError(f"Unknown operation: {event.operation}")

