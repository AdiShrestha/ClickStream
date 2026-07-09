import numpy as np
from src import victim_attacker_pairing as vap


def test_pairing_has_no_self_pairs(tmp_path, monkeypatch):
    monkeypatch.setattr(vap, "PAIRING_PATH", tmp_path / "pairs.json")
    fake_subjects = np.array([f"s{i:03d}" for i in range(10)])
    pairs = vap.create_or_load_pairing(fake_subjects)
    for victim, attacker in pairs.items():
        assert victim != attacker, f"{victim} is paired with themselves"
    assert set(pairs.keys()) == set(fake_subjects.tolist())


def test_pairing_is_deterministic_across_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(vap, "PAIRING_PATH", tmp_path / "pairs.json")
    fake_subjects = np.array([f"s{i:03d}" for i in range(10)])
    pairs1 = vap.create_or_load_pairing(fake_subjects)
    pairs2 = vap.create_or_load_pairing(fake_subjects)
    assert pairs1 == pairs2
