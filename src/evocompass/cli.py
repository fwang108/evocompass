from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from .constants import AA_TO_INDEX, OPERATIONS
from .data import read_records, records_to_arrays, save_arrays
from .features import build_features
from .models import EvoCompassModel
from .sampling import EditCandidate, apply_event, sample_event
from .training import fit_operation


def _load_operation_data(directory: Path, operation: str) -> dict[str, np.ndarray]:
    path = directory / f"{operation}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as record:
        return {key: record[key] for key in record.files}


def _split_rows(families: np.ndarray, split: dict) -> tuple[list[int], list[int], list[int]]:
    family_values = families.astype(str)
    return tuple(
        np.flatnonzero(np.isin(family_values, split[name])).tolist()
        for name in ("train", "validation", "test")
    )


def prepare_main(argv=None):
    parser = argparse.ArgumentParser(description="Convert EvoCompass JSONL rows to operation arrays")
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args(argv)
    records = read_records(args.input)
    arrays = records_to_arrays(records)
    save_arrays(arrays, args.output)
    print(json.dumps({operation: len(values["event"]) for operation, values in arrays.items()}, indent=2))


def train_main(argv=None):
    parser = argparse.ArgumentParser(description="Train EvoCompass event and identity heads")
    parser.add_argument("data")
    parser.add_argument("split")
    parser.add_argument("output")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--event-steps", type=int, default=900)
    parser.add_argument("--identity-steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args(argv)
    split = json.loads(Path(args.split).read_text())
    model = EvoCompassModel().to(args.device)
    counts = {}
    for operation_index, operation in enumerate(OPERATIONS):
        data = _load_operation_data(Path(args.data), operation)
        train, validation, test = _split_rows(data["family"], split)
        features = build_features(data["source"], data["profile"], data["time"])
        fit_operation(
            model, operation, features, torch.from_numpy(data["event"]),
            torch.from_numpy(data["target"]), train, validation,
            args.event_steps, args.identity_steps,
            args.seed + 100 * operation_index, args.device,
        )
        counts[operation] = {"train": len(train), "validation": len(validation), "test": len(test)}
    model.cpu().save(args.output, {"split": split, "row_counts": counts})
    print(json.dumps(counts, indent=2))


def evaluate_main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a frozen EvoCompass checkpoint")
    parser.add_argument("checkpoint")
    parser.add_argument("data")
    parser.add_argument("split")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    split = json.loads(Path(args.split).read_text())
    model = EvoCompassModel.load(args.checkpoint)
    report = {}
    for operation in OPERATIONS:
        data = _load_operation_data(Path(args.data), operation)
        _, _, test = _split_rows(data["family"], split)
        features = build_features(data["source"], data["profile"], data["time"])
        labels = torch.from_numpy(data["event"]).float()
        target = torch.from_numpy(data["target"]).long()
        with torch.no_grad():
            output = model.predict(operation, features)
        probabilities = output["event_probability"].clamp(1e-8, 1 - 1e-8)
        index = torch.as_tensor(test)
        event_rows = index[labels[index] > 0.5]
        report[operation] = {
            "rows": len(test),
            "events": int(len(event_rows)),
            "event_bce": float(functional.binary_cross_entropy(probabilities[index], labels[index])),
            "identity_nll": float(functional.cross_entropy(output["identity_logits"][event_rows], target[event_rows]))
            if len(event_rows) else None,
            "identity_top1": float((output["identity_logits"][event_rows].argmax(-1) == target[event_rows]).float().mean())
            if len(event_rows) else None,
        }
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text)
    print(text, end="")


def sample_main(argv=None):
    parser = argparse.ArgumentParser(description="Sample one EvoCompass edit from candidate rows")
    parser.add_argument("checkpoint")
    parser.add_argument("request", help="JSON file with sequence, time, and candidates")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    request = json.loads(Path(args.request).read_text())
    model = EvoCompassModel.load(args.checkpoint)
    candidates = []
    for record in request["candidates"]:
        source = np.asarray([AA_TO_INDEX[record["source"]]])
        features = build_features(source, np.asarray([record["msa_profile"]]), request["time"])
        with torch.no_grad():
            prediction = model.predict(record["operation"], features)
        candidates.append(EditCandidate(
            operation=record["operation"],
            position=int(record["position"]),
            event_probability=float(prediction["event_probability"][0]),
            identity_probability=prediction["identity_probability"][0].numpy(),
        ))
    event = sample_event(candidates, np.random.default_rng(args.seed))
    edited = apply_event(request["sequence"], event)
    print(json.dumps({"event": event.__dict__, "sequence": edited}, indent=2))

