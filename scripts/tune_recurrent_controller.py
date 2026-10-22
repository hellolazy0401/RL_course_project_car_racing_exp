"""Small validation-only steering/preview search. Never reads test tracks."""
import argparse
import functools
import itertools
from pathlib import Path
import numpy as np
from racing import recurrent_env
from racing.recurrent_env import geometry, RecurrentRacing
from racing.recurrent_ppo import completion_summary, selection_key
from racing.utils import save_json, append_csv


def candidate_controller(base, speed_scale=2.2, profile="v1", *, steering_gain=1.0, yaw_gain=.12, preview=.32):
    centers, pos, right, forward, velocity, index, tangent = geometry(base)
    speed = float(velocity @ forward)
    lookahead = np.clip(5.0 + preview * max(speed, 0), 6, 45)
    distances = np.linalg.norm(centers - pos, axis=1)
    target = centers[(index + 1) % len(centers)]
    for offset in range(1, 25):
        k = (index + offset) % len(centers)
        target = centers[k]
        if distances[k] >= lookahead:
            break
    delta = target - pos
    heading_error = np.arctan2(delta @ right, delta @ forward)
    steer = np.clip(steering_gain * heading_error + yaw_gain * base.car.hull.angularVelocity, -1, 1)
    curvature = 0.
    for k in (0, 3, 6, 10):
        a, b, c = (centers[(index + k + j) % len(centers)] for j in (0, 3, 6))
        ab, bc = b-a, c-b
        turn = abs(np.arctan2(ab[0]*bc[1]-ab[1]*bc[0], ab@bc))
        curvature = max(curvature, turn/max(np.linalg.norm(ab), 1e-6))
    target_speed = np.clip(np.sqrt(16/max(curvature, .001)), 24, 78)*speed_scale
    if abs(heading_error) > .65:
        target_speed = min(target_speed, 28)
    return np.array([steer, np.clip((target_speed-speed)*.1, -.8, 1)], np.float32), target_speed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=30)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    results = []
    # Explicit small, declared grid; no optimizing on the final test set.
    grid = list(itertools.product((.7, 1.0, 1.4), (.08, .16), (.25, .4)))
    save_json(out/"grid.json", dict(steering_gain=[.7,1,1.4], yaw_gain=[.08,.16],
                                   preview=[.25,.4], speed_scale=2.2,
                                   validation_seeds=[1000000,1000000+args.episodes-1]))
    for number, (gain, yaw, preview) in enumerate(grid):
        recurrent_env.controller = functools.partial(candidate_controller, steering_gain=gain, yaw_gain=yaw, preview=preview)
        env = RecurrentRacing(completion_mode="lap95", control_mode="residual", speed_scale=2.2)
        rows = []
        try:
            for seed in range(1000000, 1000000+args.episodes):
                env.reset(seed=seed)
                while True:
                    _, _, term, trunc, info = env.step(np.zeros(2, np.float32))
                    if term or trunc:
                        row = dict(candidate=number, track_seed=seed, **info["episode_metrics"])
                        rows.append(row); append_csv(out/"episodes.csv", row)
                        break
        finally:
            env.close()
        result = dict(candidate=number, steering_gain=gain, yaw_gain=yaw, preview=preview,
                      **completion_summary(rows))
        results.append(result)
        save_json(out/"results.json", results)
        print(result, flush=True)
    save_json(out/"selected.json", max(results, key=selection_key))


if __name__ == "__main__":
    main()
