"""Resume budgets and source compatibility must preserve the trained experiment."""

import json

import pytest
import torch
from tools.train_go2w import loop_budget_target, nested_digest, validate_resume_manifest


def test_additional_budget_starts_at_restored_effective_time():
    previous = 14400.848666046979
    assert loop_budget_target(previous, 12600) == previous + 12600
    assert loop_budget_target(previous) == 14400


@pytest.mark.parametrize("seconds", [0, -1, float("inf"), float("nan")])
def test_additional_budget_rejects_invalid_values(seconds):
    with pytest.raises(ValueError):
        loop_budget_target(14400, seconds)


def test_resume_source_allows_tools_but_rejects_training_changes(tmp_path):
    old = {"src/task.py": "unchanged", "tools/evaluate.py": "old"}
    (tmp_path / "source_manifest.json").write_text(json.dumps({"sha256": "parent", "files": old}))
    current = old | {"tools/evaluate.py": "fixed", "tests/test_resume.py": "new"}
    assert validate_resume_manifest(tmp_path, "parent", current) == [
        "tests/test_resume.py",
        "tools/evaluate.py",
    ]
    with pytest.raises(ValueError, match="source changed"):
        validate_resume_manifest(tmp_path, "parent", current | {"src/task.py": "changed"})
    with pytest.raises(ValueError, match="identity differs"):
        validate_resume_manifest(tmp_path, "wrong", current)


def test_optimizer_digest_compares_nested_tensor_values():
    state = {"state": {0: {"step": torch.tensor(100), "exp_avg": torch.ones(3)}}, "lr": 4e-5}
    assert nested_digest(state) == nested_digest(
        {"lr": 4e-5, "state": {0: {"exp_avg": torch.ones(3), "step": torch.tensor(100)}}}
    )
    state["state"][0]["exp_avg"][0] = 0
    assert nested_digest(state) != nested_digest(
        {"state": {0: {"step": torch.tensor(100), "exp_avg": torch.ones(3)}}, "lr": 4e-5}
    )
