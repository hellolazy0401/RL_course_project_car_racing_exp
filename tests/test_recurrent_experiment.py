import hashlib
import json
from pathlib import Path
import subprocess
import sys
import pytest
from racing.recurrent_experiment import evaluate_checkpoint, confirm_candidates


def test_cached_evaluation_checks_model_and_tracks(tmp_path):
    model = tmp_path / "model.pt"
    model.write_bytes(b"trusted dummy")
    out = tmp_path / "eval"
    out.mkdir()
    result = dict(checkpoint_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                  track_seed_start=10, track_seed_stop=20, completion_rate=.9)
    (out / "summary.json").write_text(json.dumps(result))
    assert evaluate_checkpoint(model, out, range(10, 20)) == result
    with pytest.raises(ValueError, match="Stale"):
        evaluate_checkpoint(model, out, range(11, 21))
    model.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Stale"):
        evaluate_checkpoint(model, out, range(10, 20))


def test_confirmation_selects_completed_segment_by_completion_before_speed(tmp_path, monkeypatch):
    from racing import recurrent_experiment as study
    base = tmp_path / "training" / "candidate"
    segment = base / "segment_1"
    segment.mkdir(parents=True)
    for folder, completion, seconds in ((base, .8, 10), (segment, .9, 15)):
        (folder / "finished.json").write_text("{}")
        (folder / "best.pt").write_bytes(b"test")
        (folder / "best_metrics.json").write_text(json.dumps(dict(
            steps=100, completion_rate=completion, mean_finish_seconds=seconds, mean_coverage=.99, mean_return=900)))
    seen = []
    def evaluate(path, out, seeds):
        seen.append((path, seeds))
        return dict(completion_rate=.9, mean_finish_seconds=15)
    monkeypatch.setattr(study, "evaluate_checkpoint", evaluate)
    confirm_candidates(tmp_path, ["candidate"])
    assert seen == [(segment / "best.pt", range(1100000, 1100100))]


def test_windows_spawn_launcher_does_not_load_torch():
    result = subprocess.run([sys.executable, "-c",
        "import runpy,sys; runpy.run_module('racing.train',run_name='__mp_main__'); assert 'torch' not in sys.modules"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_reproduction_remaps_paths_and_preserves_protocol(tmp_path):
    from racing.reproduce_recurrent import training_command
    config = dict(device="cpu", seed=1, num_envs=8, rollout=256, normalize_adv=True,
                  rp_observation="state", rp_control="direct", target_kl=.03,
                  eval_episodes=10, rp_split_head=False, rp_finish_assist=False,
                  out="old_root", resume="old_absolute_checkpoint.pt")
    resume = tmp_path / "training" / "run" / "last.pt"
    cmd = training_command(config, tmp_path, "run/segment_1", 393216, resume)
    assert cmd[cmd.index("--resume")+1] == str(resume)
    assert cmd[cmd.index("--steps")+1] == "393216"
    assert cmd[cmd.index("--target-kl")+1] == "0.03"
    assert cmd[cmd.index("--eval-episodes")+1] == "10"
    assert "old_root" not in cmd and "old_absolute_checkpoint.pt" not in cmd


def test_parallel_evaluation_matches_serial_physics_and_memory(tmp_path):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    import torch
    from racing.recurrent_ppo import RecurrentAgent
    torch.set_num_threads(2)
    agent = RecurrentAgent(hidden_size=16, control_mode="residual")
    checkpoint = tmp_path / "fixture.pt"
    torch.save(dict(model=agent.state_dict(),
        config=dict(rp_observation="state", rp_hidden=16, rp_split_head=False, rp_control="residual"),
        env_config=dict(recurrent=True, observation_mode="state", control_mode="residual",
                        completion_mode="lap95", speed_scale=2.2),
        steps=1), checkpoint)
    serial = evaluate_checkpoint(checkpoint, tmp_path / "serial", range(101, 102))
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as pool:
        first = pool.submit(evaluate_checkpoint, checkpoint, tmp_path / "parallel_a", range(101, 102))
        second = pool.submit(evaluate_checkpoint, checkpoint, tmp_path / "parallel_b", range(102, 103))
        assert first.result() == serial
        assert second.result()["episodes"] == 1
