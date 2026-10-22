"""Replay the actual recorded experiment configurations into a fresh directory.

Unlike a new hyperparameter sweep, this preserves each run's budget, seed,
validation protocol and continuation boundary. Old absolute paths are remapped.
"""
import argparse
from pathlib import Path
import sys
from .recurrent_experiment import read, frozen_source, run_command, finalize
from .recurrent_ppo import selection_key
from .utils import save_json


VALUE_KEYS = (
    "seed", "num_envs", "rollout", "batch_size", "actor_lr", "critic_lr", "gamma",
    "gae_lambda", "reward_scale", "ppo_epochs", "clip", "entropy_coef", "target_kl",
    "max_grad_norm", "completion_mode", "rp_observation", "rp_control", "rp_controller",
    "rp_hidden", "rp_sequence", "rp_speed_scale", "rp_residual_scale",
    "rp_completion_bonus", "rp_offroad_penalty", "eval_every", "eval_episodes",
    "eval_seed_start", "threads",
)


def training_command(config, root, name, steps, resume=None, device=None):
    cmd = [sys.executable, "-m", "racing.train", "--algo", "recurrent-ppo",
           "--out", str(root / "training"), "--name", name, "--steps", str(steps),
           "--device", device or config["device"]]
    for key in VALUE_KEYS:
        if config.get(key) is not None:
            cmd += ["--" + key.replace("_", "-"), str(config[key])]
    cmd.append("--normalize-adv" if config["normalize_adv"] else "--no-normalize-adv")
    for key in ("rp_split_head", "rp_finish_assist"):
        if config.get(key, False):
            cmd.append("--" + key.replace("_", "-"))
    if resume is not None:
        cmd += ["--resume", str(resume)]
    return cmd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--records", required=True, help="Experiment root, or extracted records directory with training/")
    p.add_argument("--out", required=True, help="New output directory; never overwrites an earlier experiment")
    p.add_argument("--device", choices=["cpu", "cuda"], help="Optional device override; default follows each recorded run")
    p.add_argument("--finalize", action="store_true", help="Also confirm, select and run the independent tests")
    args = p.parse_args()
    records, root = Path(args.records).resolve(), Path(args.out).resolve()
    if not (records / "training").is_dir():
        p.error("records must contain training/")
    root.mkdir(parents=True, exist_ok=False)
    source = frozen_source(root)
    candidates, derived = [], []
    save_json(root / "reproduction.json", dict(records=str(records), device_override=args.device,
              training_track_resets="Environment restarts at recorded continuation boundaries",
              pixel_discarded_rollout="Uncheckpointed original work is not replayed",
              numerical_note="Same configuration, not a guarantee of bitwise equivalence across machines"))
    for run in sorted((records / "training").iterdir()):
        if not (run / "config.json").exists():
            continue
        candidates.append(run.name)
        if (run / "derivation.json").exists():
            derived.append(run)
            continue
        segments = [run] + sorted(run.glob("segment_*"), key=lambda q: int(q.name.split("_")[-1]))
        previous = None
        for segment in segments:
            config = read(segment / "config.json")
            name = segment.relative_to(records / "training").as_posix()
            steps = config["steps"]
            if (segment / "intentional_restart.json").exists():
                steps = read(segment / "intentional_restart.json")["checkpoint_steps"]
            command = training_command(config, root, name, steps, previous, args.device)
            run_command(root, source, name.replace("/", "_"), command)
            previous = root / "training" / name / "last.pt"
    for run in derived:
        # This study's finish variant derives from the validation-best state
        # direct seed 0 checkpoint; it does not use confirmation/test results.
        original = root / "training" / "state_direct_s0"
        paths = [original / "best.pt", *original.glob("segment_*/best.pt")]
        checkpoint = max(paths, key=lambda q: selection_key(read(q.parent / "best_metrics.json")))
        run_command(root, source, run.name, [
            sys.executable, "-m", "scripts.derive_finish_assist", "--checkpoint", str(checkpoint),
            "--out", str(root / "training" / run.name)])
    if args.finalize:
        finalize(root, candidates, test_episodes=100)
        run_command(root, source, "report", [sys.executable, "-m", "scripts.report_recurrent", "--root", str(root)])
    save_json(root / "reproduction_complete.json", dict(candidates=candidates, tested=args.finalize))
    save_json(root / "status.json", dict(status="complete", candidates=candidates, tested=args.finalize))


if __name__ == "__main__":
    main()
