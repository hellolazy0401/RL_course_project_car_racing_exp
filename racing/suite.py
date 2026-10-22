"""Sequential, fail-fast experiment matrix; avoids multiple GPU jobs competing."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=["smoke", "pilot", "main"], default="pilot")
    p.add_argument("--algos", nargs="+", choices=["a2c", "ppo", "vpg50", "grpo", "grpo-c", "sac", "recurrent-ppo"], default=["a2c", "ppo", "vpg50"])
    p.add_argument("--rp-observation", choices=["pixels", "state"], default="pixels")
    p.add_argument("--rp-control", choices=["direct", "residual", "adaptive"], default="direct")
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--out", default="runs")
    p.add_argument("--steps", type=int)
    p.add_argument("--actor-lr", type=float, default=None,
                   help="Actor learning rate for all selected algorithms; default: grpo-c 1e-4, others 3e-4")
    p.add_argument("--completion-mode", choices=["original", "lap95"], default="original")
    p.add_argument("--grpo-c-episodes-per-update", type=int, default=1)
    p.add_argument("--skip-completed", action="store_true")
    args = p.parse_args()
    if args.grpo_c_episodes_per_update < 1:
        p.error("--grpo-c-episodes-per-update must be positive")
    settings = dict(smoke=(512, [0], 1, 2, 16, 128, 1),
                    pilot=(100_000, [0], 5, 4, 128, 25_000, 10),
                    main=(1_000_000, [0,1,2], 10, 4, 128, 50_000, 50))
    steps, seeds, val_eps, envs, rollout, eval_every, test_eps = settings[args.preset]
    root = Path(args.out) / args.preset
    root.mkdir(parents=True, exist_ok=True)
    for algo in args.algos:
        effective_actor_lr = args.actor_lr if args.actor_lr is not None else (1e-4 if algo == "grpo-c" else 3e-4)
        for seed in args.seeds or seeds:
            effective_steps = args.steps or steps
            if algo == "recurrent-ppo":
                quantum = envs * rollout
                effective_steps = ((effective_steps + quantum - 1) // quantum) * quantum
                if effective_steps != (args.steps or steps):
                    print(f"Recurrent PPO sequence budget: requested {args.steps or steps}, actual {effective_steps}", flush=True)
            run = root / f"{algo}_s{seed}"
            if args.skip_completed and (run / "finished.json").exists() and not (run / "config.json").exists():
                p.error("Completed run is missing config.json; cannot confirm compatibility. Use a new --out")
            if args.skip_completed and (run / "config.json").exists():
                prior = json.loads((run / "config.json").read_text())
                if prior.get("completion_mode", "original") != args.completion_mode:
                    p.error("Existing run uses a different completion mode; use a new --out")
                if prior.get("actor_lr") != effective_actor_lr:
                    p.error("Existing run uses a different actor learning rate; use a new --out")
                if algo == "recurrent-ppo" and (prior.get("rp_observation") != args.rp_observation
                        or prior.get("rp_control") != args.rp_control or prior.get("steps") != effective_steps):
                    p.error("Existing recurrent run has a different input/control/budget; use a new --out")
                if algo == "grpo-c" and prior.get("grpo_c_variant") != "pixel_motion_global_loo_mixture_v2":
                    p.error("Existing GRPO-C run uses an older or unknown implementation; use a new --out")
                if algo == "grpo-c" and prior.get("grpo_c_episodes_per_update", 1) != args.grpo_c_episodes_per_update:
                    p.error("Existing GRPO-C run uses a different episode batch; use a new --out")
                if algo == "grpo-c" and (prior.get("grpo_c_warm_start") or (run/"warm_start.json").exists()
                        or prior.get("grpo_c_selection", "return") != "return"
                        or prior.get("grpo_c_trainable", "full") != "full"
                        or prior.get("grpo_c_completion_bonus", 0.) != 0.):
                    p.error("Existing GRPO-C run uses a different initialization or objective; use a new --out")
            if not (args.skip_completed and (run / "finished.json").exists()):
                cmd = [sys.executable, "-m", "racing.train", "--algo", algo, "--seed", str(seed),
                    "--steps", str(effective_steps), "--device", args.device,
                    "--out", str(root), "--num-envs", str(envs), "--rollout", str(rollout),
                    "--eval-every", str(eval_every), "--eval-episodes", str(val_eps),
                    "--completion-mode", args.completion_mode]
                # Smoke still exercises the real K=50 update, not a different algorithm.
                if algo == "sac" and args.preset == "smoke":
                    cmd += ["--sac-learning-starts", "128", "--sac-buffer-size", "1024"]
                if algo == "recurrent-ppo":
                    cmd += ["--rp-observation", args.rp_observation, "--rp-control", args.rp_control,
                            "--rp-sequence", str(min(32, rollout))]
                if algo == "grpo":
                    cmd += ["--grpo-group-size", str(envs)]
                if algo == "grpo-c":
                    cmd += ["--grpo-c-episodes-per-update", str(args.grpo_c_episodes_per_update)]
                if args.actor_lr is not None:
                    cmd += ["--actor-lr", str(args.actor_lr)]
                elif algo == "grpo-c":
                    cmd += ["--actor-lr", "1e-4"]
                subprocess.run(cmd, check=True)
            # Primary: last checkpoint at equal sample budget. Secondary: validation-selected best.
            for checkpoint, label in (("last", "test"), ("best", "test_best")):
                dest = run / label
                if args.skip_completed and (dest / "summary.json").exists():
                    continue
                subprocess.run([sys.executable,"-m","racing.evaluate","--checkpoint",str(run/f"{checkpoint}.pt"),
                    "--out",str(dest),"--device",args.device,"--episodes",str(test_eps),
                    "--videos", "1" if args.preset=="smoke" else "3"], check=True)
    report_cmd=[sys.executable,"-m","racing.report","--root",str(root)]
    if "grpo" in args.algos or "grpo-c" in args.algos or "recurrent-ppo" in args.algos:
        report_cmd += ["--allow-unequal-budgets"]
    subprocess.run(report_cmd, check=True)


if __name__ == "__main__":
    main()
