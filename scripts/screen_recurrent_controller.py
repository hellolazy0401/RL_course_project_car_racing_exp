"""Validation-only controller screen, with zero learned residual."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from racing.recurrent_env import RecurrentRacing
from racing.utils import append_csv, save_json, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scales", type=float, nargs="+", default=[.8, 1., 1.2])
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed-start", type=int, default=1000000)
    p.add_argument("--profile", choices=["v1", "v2"], default="v1")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    results = []
    for scale in args.scales:
        env = RecurrentRacing(completion_mode="lap95", control_mode="residual", speed_scale=scale,
                             controller_profile=args.profile)
        rows = []
        started = time.perf_counter()
        try:
            for seed in range(args.seed_start, args.seed_start + args.episodes):
                obs, _ = env.reset(seed=seed)
                while True:
                    obs, _, term, trunc, info = env.step(np.zeros(2, np.float32))
                    if term or trunc:
                        row = dict(speed_scale=scale, track_seed=seed, **info["episode_metrics"])
                        rows.append(row)
                        append_csv(out / "episodes.csv", row)
                        print(json.dumps(row), flush=True)
                        break
        finally:
            env.close()
        completed = [r["raw_frames"] / 50 for r in rows if r["completed"]]
        result = dict(speed_scale=scale, controller_profile=args.profile, **summary(rows), mean_finish_seconds=float(np.mean(completed)) if completed else None,
                      seconds=time.perf_counter() - started)
        results.append(result)
        save_json(out / "summary.json", results)
        print(result, flush=True)


if __name__ == "__main__":
    main()
