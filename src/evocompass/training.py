from __future__ import annotations

import random

import torch
import torch.nn.functional as functional

from .calibration import fit_logistic_calibration
from .models import EvoCompassModel


def _fit_event(model, features, labels, rows, steps, seed, device):
    torch.manual_seed(seed)
    random.seed(seed)
    features, labels = features.to(device), labels.float().to(device)
    index = torch.as_tensor(rows, dtype=torch.long, device=device)
    positives = labels[index].sum().clamp_min(1.0)
    positive_weight = ((len(index) - positives) / positives).clamp(1.0, 200.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    batch_size = min(1024, len(index))
    for _ in range(steps):
        selected = index[torch.randint(len(index), (batch_size,), device=device)]
        loss = functional.binary_cross_entropy_with_logits(
            model(features[selected]), labels[selected], pos_weight=positive_weight
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def _fit_identity(model, features, labels, rows, steps, seed, device):
    torch.manual_seed(seed)
    random.seed(seed)
    features, labels = features.to(device), labels.long().to(device)
    index = torch.as_tensor(rows, dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
    batch_size = min(512, len(index))
    for _ in range(steps):
        selected = index[torch.randint(len(index), (batch_size,), device=device)]
        loss = functional.cross_entropy(model(features[selected]), labels[selected])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def fit_operation(
    model: EvoCompassModel,
    operation: str,
    features: torch.Tensor,
    event_labels: torch.Tensor,
    target_labels: torch.Tensor,
    train_rows: list[int],
    validation_rows: list[int],
    event_steps: int = 900,
    identity_steps: int = 800,
    seed: int = 0,
    device: str = "cpu",
) -> None:
    if not train_rows or not validation_rows:
        raise ValueError("Training and validation rows are required")
    features = features.float()
    _fit_event(
        model.event_heads[operation], features, event_labels, train_rows,
        event_steps, seed, device,
    )
    event_rows = [row for row in train_rows if event_labels[row].item() > 0.5]
    if not event_rows:
        raise ValueError(f"No positive {operation} rows in the training split")
    _fit_identity(
        model.identity_heads[operation], features, target_labels.clamp_min(0),
        event_rows, identity_steps, seed + 1, device,
    )
    calibration_rows = train_rows + validation_rows
    with torch.no_grad():
        logits = model.event_heads[operation](features.to(device)).cpu().numpy()
    model.calibration[operation] = fit_logistic_calibration(
        logits[calibration_rows], event_labels.numpy()[calibration_rows]
    )

