import numpy as np
import pytest
from src import subject_split as ss


def test_split_creates_no_overlap_and_correct_proportions(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "SPLIT_PATH", tmp_path / "split.json")
    fake_subjects = np.array([f"s{i:03d}" for i in range(51) for _ in range(8)])
    background, held_out = ss.create_or_load_subject_split(fake_subjects)
    assert len(background) + len(held_out) == 51
    assert len(set(background) & set(held_out)) == 0
    assert len(background) == round(51 * 0.70)


def test_split_is_deterministic_across_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "SPLIT_PATH", tmp_path / "split.json")
    fake_subjects = np.array([f"s{i:03d}" for i in range(51)])
    bg1, ho1 = ss.create_or_load_subject_split(fake_subjects)
    bg2, ho2 = ss.create_or_load_subject_split(fake_subjects)  # loads from disk
    assert bg1 == bg2
    assert ho1 == ho2


def test_split_detects_stale_file(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "SPLIT_PATH", tmp_path / "split.json")
    original_subjects = np.array([f"s{i:03d}" for i in range(51)])
    ss.create_or_load_subject_split(original_subjects)
    different_subjects = np.array([f"s{i:03d}" for i in range(45)])
    with pytest.raises(AssertionError):
        ss.create_or_load_subject_split(different_subjects)
