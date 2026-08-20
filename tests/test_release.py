import json
from pathlib import Path

from evocompass import EvoCompassModel


ROOT = Path(__file__).resolve().parents[1]


def test_release_checkpoint_loads():
    model = EvoCompassModel.load(ROOT / "artifacts/checkpoints/evocompass.pt")
    assert sum(parameter.numel() for parameter in model.parameters()) == 30303


def test_aggregate_reports_have_expected_scope():
    results = json.loads((ROOT / "artifacts/reports/trajectory_results.json").read_text())
    specificity = json.loads((ROOT / "artifacts/reports/correspondence_specificity.json").read_text())
    assert results["dataset"]["held_out_families"] == 15
    assert results["dataset"]["branches"] == 1557
    assert specificity["dose_response"]["monotonic"] is True

