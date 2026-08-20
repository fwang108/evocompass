import numpy as np

from evocompass.sampling import EditCandidate, EditEvent, apply_event, sample_event


def test_event_mechanics():
    assert apply_event("ACD", EditEvent("substitution", 1, "V", 0.1)) == "AVD"
    assert apply_event("ACD", EditEvent("deletion", 1, None, 0.1)) == "AD"
    assert apply_event("ACD", EditEvent("insertion", 1, "V", 0.1)) == "AVCD"


def test_sampling_returns_valid_event():
    identity = np.zeros(20)
    identity[1] = 1.0
    candidates = [
        EditCandidate("substitution", 0, 0.8, identity),
        EditCandidate("deletion", 1, 0.2),
    ]
    event = sample_event(candidates, np.random.default_rng(2))
    assert event.operation in {"substitution", "deletion"}
    assert event.waiting_time >= 0

