from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .constants import AA_TO_INDEX, OPERATIONS


REQUIRED_FIELDS = {
    "family", "group", "operation", "source", "msa_profile", "time", "event", "target"
}


def read_records(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = REQUIRED_FIELDS - record.keys()
            if missing:
                raise ValueError(f"Line {line_number} is missing {sorted(missing)}")
            if record["operation"] not in OPERATIONS:
                raise ValueError(f"Line {line_number} has an unsupported operation")
            records.append(record)
    return records


def records_to_arrays(records: list[dict]) -> dict[str, dict[str, np.ndarray]]:
    output = {}
    for operation in OPERATIONS:
        rows = [record for record in records if record["operation"] == operation]
        if not rows:
            continue
        output[operation] = {
            "family": np.asarray([row["family"] for row in rows]),
            "group": np.asarray([row["group"] for row in rows]),
            "source": np.asarray([AA_TO_INDEX[row["source"]] for row in rows], dtype=np.int64),
            "profile": np.asarray([row["msa_profile"] for row in rows], dtype=np.float32),
            "time": np.asarray([row["time"] for row in rows], dtype=np.float32),
            "event": np.asarray([row["event"] for row in rows], dtype=np.float32),
            "target": np.asarray(
                [AA_TO_INDEX.get(row["target"], -1) for row in rows], dtype=np.int64
            ),
        }
    return output


def save_arrays(arrays: dict[str, dict[str, np.ndarray]], directory: str | Path) -> None:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    for operation, values in arrays.items():
        np.savez_compressed(target / f"{operation}.npz", **values)

