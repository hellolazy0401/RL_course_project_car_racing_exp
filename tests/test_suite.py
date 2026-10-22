import json
from unittest.mock import Mock

import pytest

from racing import suite


CURRENT_VARIANT = "pixel_motion_global_loo_mixture_v2"


def completed_run(root, algo, actor_lr, variant=CURRENT_VARIANT):
    run = root / "smoke" / f"{algo}_s0"
    run.mkdir(parents=True)
    config = dict(completion_mode="lap95", actor_lr=actor_lr)
    if variant is not None:
        config["grpo_c_variant"] = variant
    (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run / "finished.json").write_text("{}", encoding="utf-8")
    for label in ("test", "test_best"):
        (run / label).mkdir()
        (run / label / "summary.json").write_text("{}", encoding="utf-8")
    return run


def suite_arguments(root, algo, requested_lr):
    args = ["racing.suite", "--preset", "smoke", "--algos", algo, "--seeds", "0",
            "--completion-mode", "lap95", "--out", str(root), "--skip-completed"]
    if requested_lr is not None:
        args += ["--actor-lr", str(requested_lr)]
    return args


@pytest.mark.parametrize("algo,prior_lr,requested_lr", [
    ("grpo-c", 3e-4, None),
    ("grpo-c", 1e-4, 3e-4),
    ("ppo", 3e-4, 1e-4),
])
def test_skip_completed_rejects_wrong_learning_rate(tmp_path, monkeypatch, capsys,
                                                  algo, prior_lr, requested_lr):
    completed_run(tmp_path, algo, prior_lr)
    called = Mock()
    monkeypatch.setattr(suite.sys, "argv", suite_arguments(tmp_path, algo, requested_lr))
    monkeypatch.setattr(suite.subprocess, "run", called)
    with pytest.raises(SystemExit) as error:
        suite.main()
    assert error.value.code == 2
    assert "different actor learning rate" in capsys.readouterr().err
    called.assert_not_called()


@pytest.mark.parametrize("variant", [None, "original_v1"])
def test_skip_completed_rejects_old_or_missing_grpo_c_variant(tmp_path, monkeypatch, capsys, variant):
    completed_run(tmp_path, "grpo-c", 1e-4, variant=variant)
    called = Mock()
    monkeypatch.setattr(suite.sys, "argv", suite_arguments(tmp_path, "grpo-c", None))
    monkeypatch.setattr(suite.subprocess, "run", called)
    with pytest.raises(SystemExit) as error:
        suite.main()
    assert error.value.code == 2
    assert "older or unknown implementation" in capsys.readouterr().err
    called.assert_not_called()


@pytest.mark.parametrize("algo,actor_lr,requested_lr", [
    ("grpo-c", 1e-4, None),
    ("grpo-c", 2e-4, 2e-4),
    ("ppo", 3e-4, None),
    ("ppo", 1e-4, 1e-4),
])
def test_matching_completed_run_skips_train_and_evaluation(tmp_path, monkeypatch,
                                                         algo, actor_lr, requested_lr):
    completed_run(tmp_path, algo, actor_lr)
    called = Mock()
    monkeypatch.setattr(suite.sys, "argv", suite_arguments(tmp_path, algo, requested_lr))
    monkeypatch.setattr(suite.subprocess, "run", called)
    suite.main()
    called.assert_called_once()
    assert called.call_args.args[0][1:3] == ["-m", "racing.report"]


def test_skip_completed_rejects_different_episode_batch(tmp_path, monkeypatch, capsys):
    completed_run(tmp_path, "grpo-c", 1e-4)
    called = Mock()
    monkeypatch.setattr(suite.sys, "argv", suite_arguments(tmp_path, "grpo-c", None)
                        + ["--grpo-c-episodes-per-update", "4"])
    monkeypatch.setattr(suite.subprocess, "run", called)
    with pytest.raises(SystemExit):
        suite.main()
    assert "different episode batch" in capsys.readouterr().err
    called.assert_not_called()


@pytest.mark.parametrize("field,value", [("grpo_c_selection", "completion"),
                                         ("grpo_c_completion_bonus", 200.),
                                         ("grpo_c_warm_start", "ppo.pt"),
                                         ("lineage_file", True)])
def test_standard_suite_rejects_hybrid_or_changed_objective(tmp_path, monkeypatch, capsys, field, value):
    run = completed_run(tmp_path, "grpo-c", 1e-4)
    if field == "lineage_file":
        (run/"warm_start.json").write_text('{}', encoding='utf-8')
    else:
        config = json.loads((run/"config.json").read_text())
        config[field] = value
        (run/"config.json").write_text(json.dumps(config))
    called = Mock()
    monkeypatch.setattr(suite.sys, "argv", suite_arguments(tmp_path, "grpo-c", None))
    monkeypatch.setattr(suite.subprocess, "run", called)
    with pytest.raises(SystemExit):
        suite.main()
    assert 'different initialization or objective' in capsys.readouterr().err
    called.assert_not_called()
