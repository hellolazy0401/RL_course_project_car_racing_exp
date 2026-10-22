"""Record actual recurrent driving telemetry on a requested validation track."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from racing.recurrent_env import make_env
from racing.recurrent_ppo import agent_from_config
from racing.utils import append_csv, save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    agent = agent_from_config(ck["config"]).eval()
    agent.load_state_dict(ck["model"])
    env = make_env(ck["env_config"])
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    rows = []
    try:
        obs, _ = env.reset(seed=args.seed)
        gate_rows = []
        original_gate_update = env.finish_gate.update
        def trace_gate(on_start, forward, visited, total):
            entered = bool(on_start and not env.finish_gate.was_on_start)
            finished = original_gate_update(on_start, forward, visited, total)
            gate_row = dict(raw_frame=len(gate_rows)+1, on_start=bool(on_start),
                            entered=entered, forward=bool(forward), coverage=visited/total,
                            gate_finish=finished)
            gate_rows.append(gate_row)
            append_csv(out/"gate_trace.csv", gate_row)
            return finished
        env.finish_gate.update = trace_gate
        state = agent.initial_state(1)
        starts = torch.ones(1, dtype=torch.bool)
        centers = env.track_centers
        while True:
            with torch.no_grad():
                action, _, _, _, _, state = agent.step(torch.as_tensor(obs[None]), state, starts, deterministic=True)
            starts.fill_(False)
            obs, _, term, trunc, info = env.step(action[0].numpy())
            base = env.unwrapped
            position = np.asarray(base.car.hull.position)
            nearest = int(np.square(centers-position).sum(1).argmin())
            row = dict(seconds=env.raw_steps/50, x=float(position[0]), y=float(position[1]),
                       speed=float(base.car.hull.linearVelocity.length), track_index=nearest,
                       on_start=int(env.on_start()), coverage=base.tile_visited_count/len(centers),
                       start_distance=float(np.linalg.norm(position-centers[0])),
                       steering=float(info["applied_command"][0]),
                       longitudinal=float(info["applied_command"][1]))
            rows.append(row); append_csv(out/"telemetry.csv", row)
            if term or trunc:
                save_json(out/"summary.json", dict(track_seed=args.seed, **info["episode_metrics"],
                          final=row, track_tiles=len(centers), checkpoint=args.checkpoint))
                save_json(out/"gate_trace_summary.json", dict(
                    physics_frames=len(gate_rows),
                    start_contact_frames_after_95=sum(r["on_start"] and r["coverage"] > .95 for r in gate_rows),
                    forward_start_entries_after_95=sum(r["entered"] and r["forward"] and r["coverage"] > .95 for r in gate_rows),
                    finish_gate_triggered=any(r["gate_finish"] for r in gate_rows)))
                break
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        axes[0, 0].plot(centers[:, 0], centers[:, 1], color="#aab4c2", lw=8, alpha=.6)
        axes[0, 0].plot([r["x"] for r in rows], [r["y"] for r in rows], color="#168d85", lw=1.5)
        axes[0, 0].scatter(*centers[0], color="red", label="Start/finish", zorder=5)
        axes[0, 0].scatter(rows[-1]["x"], rows[-1]["y"], color="black", label="Final car", zorder=5)
        axes[0, 0].axis("equal"); axes[0, 0].legend(); axes[0, 0].set_title(f"Track {args.seed}")
        t = [r["seconds"] for r in rows]
        axes[0, 1].plot(t, [r["speed"] for r in rows]); axes[0, 1].set_ylabel("Speed (world units/s)")
        axes[1, 0].plot(t, [r["steering"] for r in rows], label="Steer")
        axes[1, 0].plot(t, [r["longitudinal"] for r in rows], label="Gas / negative brake")
        axes[1, 0].legend()
        axes[1, 1].plot(t, [r["coverage"] for r in rows], label="Coverage")
        axes[1, 1].plot(t, [r["track_index"]/len(centers) for r in rows], label="Nearest track fraction")
        axes[1, 1].legend()
        for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
            ax.set_xlabel("Seconds"); ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(out/"telemetry.png", dpi=160); plt.close(fig)
        print(json.dumps(info["episode_metrics"]), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
