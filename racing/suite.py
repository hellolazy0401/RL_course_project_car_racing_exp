"""Sequential, fail-fast experiment matrix; avoids multiple GPU jobs competing."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=["smoke", "pilot", "main"], default="pilot")
    p.add_argument("--algos", nargs="+", choices=["a2c", "ppo", "vpg50"], default=["a2c", "ppo", "vpg50"])
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--out", default="runs")
    p.add_argument("--steps", type=int)
    p.add_argument("--skip-completed", action="store_true")
    args = p.parse_args()
    settings = dict(smoke=(512, [0], 1, 2, 16, 128, 1),
                    pilot=(100_000, [0], 5, 4, 128, 25_000, 10),
                    main=(1_000_000, [0,1,2], 10, 4, 128, 50_000, 50))
    steps, seeds, val_eps, envs, rollout, eval_every, test_eps = settings[args.preset]
    root = Path(args.out) / args.preset
    root.mkdir(parents=True, exist_ok=True)
    for algo in args.algos:
        for seed in args.seeds or seeds:
            run = root / f"{algo}_s{seed}"
            if not (args.skip_completed and (run / "finished.json").exists()):
                cmd = [sys.executable, "-m", "racing.train", "--algo", algo, "--seed", str(seed),
                    "--steps", str(args.steps or steps), "--device", args.device,
                    "--out", str(root), "--num-envs", str(envs), "--rollout", str(rollout),
                    "--eval-every", str(eval_every), "--eval-episodes", str(val_eps)]
                # Smoke still exercises the real K=50 update, not a different algorithm.
                subprocess.run(cmd, check=True)
            # Primary: last checkpoint at equal sample budget. Secondary: validation-selected best.
            for checkpoint, label in (("last", "test"), ("best", "test_best")):
                dest = run / label
                if args.skip_completed and (dest / "summary.json").exists():
                    continue
                subprocess.run([sys.executable,"-m","racing.evaluate","--checkpoint",str(run/f"{checkpoint}.pt"),
                    "--out",str(dest),"--device",args.device,"--episodes",str(test_eps),
                    "--videos", "1" if args.preset=="smoke" else "3"], check=True)
    subprocess.run([sys.executable,"-m","racing.report","--root",str(root)], check=True)


if __name__ == "__main__":
    main()
